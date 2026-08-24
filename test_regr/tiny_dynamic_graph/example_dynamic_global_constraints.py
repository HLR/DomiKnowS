"""Train a tiny MLP with executable and dynamic global-constraint losses.

The graph is declared once with a union vocabulary. Each row activates only the
concepts needed by its example, which also deactivates unrelated sensors and
global constraints. Training uses the regular DomiKnowS ``program.train`` path.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from domiknows.graph import Concept, Graph, ifL
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import ModuleLearner
from domiknows.sensor.pytorch.sensors import FunctionalReaderSensor

from test_regr.tiny_dynamic_graph.example import CandidateClassifier, reset_domiknows_state


CONCEPT_VOCABULARY = ("red", "colored", "dog", "animal", "tree", "plant")
MOCK_DATA_PATH = Path(__file__).with_name("mock_global_constraint_samples.json")


@dataclass(frozen=True)
class MockObject:
    object_id: str
    features: tuple
    labels: dict


@dataclass(frozen=True)
class MockConstraintExample:
    example_id: str
    active_concepts: tuple
    rule: str
    objects: tuple


@dataclass
class DynamicConstraintContext:
    graph: Graph
    concepts: dict
    candidates: tuple
    rules: dict
    examples: tuple
    entries: tuple
    program: InferenceProgram
    shared_model: torch.nn.Module
    optimizer: torch.optim.Optimizer


class TinyConceptMLP(torch.nn.Module):
    """One small shared predictor for all six atomic concepts."""

    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(len(CONCEPT_VOCABULARY), 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, len(CONCEPT_VOCABULARY)),
        )
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.network[0].weight)
            self.network[0].bias.zero_()
            torch.nn.init.xavier_uniform_(self.network[-1].weight)
            # Every implication starts violated: antecedent true, consequent false.
            self.network[-1].bias.copy_(
                torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
            )

    def concept_logits(self, features, concept_index):
        score = self.network(features.float())[:, concept_index]
        return torch.stack((-score, score), dim=-1)


class ConceptView(torch.nn.Module):
    """Expose one shared MLP output as a binary DomiKnowS learner."""

    def __init__(self, shared_model, concept_index):
        super().__init__()
        self.shared_model = shared_model
        self.concept_index = concept_index

    def forward(self, features):
        return self.shared_model.concept_logits(features, self.concept_index)


class ActiveConceptDataset:
    """Switch active concepts immediately before yielding a compiled row.

    Consume this dataset sequentially with ``batch_size=1``. The graph state is
    restored even if iteration exits through an exception.
    """

    def __init__(self, graph, entries):
        self.graph = graph
        self.entries = tuple(entries)

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        try:
            for active_concepts, row in self.entries:
                self.graph.set_active_concepts(active_concepts)
                yield row
        finally:
            self.graph.set_active_concepts(None)


def load_mock_examples(path=MOCK_DATA_PATH):
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(
        MockConstraintExample(
            example_id=record["example_id"],
            active_concepts=tuple(record["active_concepts"]),
            rule=record["rule"],
            objects=tuple(
                MockObject(
                    object_id=obj["object_id"],
                    features=tuple(obj["features"]),
                    labels=dict(obj["labels"]),
                )
                for obj in record["objects"]
            ),
        )
        for record in records
    )


def _compiled_rows(examples, device):
    rows = []
    active_names_by_row = []
    for example in examples:
        features = torch.tensor(
            [obj.features for obj in example.objects], dtype=torch.float32, device=device
        )
        indices = torch.arange(len(example.objects), device=device)
        for concept_name in example.active_concepts:
            for candidate_index, obj in enumerate(example.objects):
                rows.append(
                    {
                        "features": features,
                        "indices": indices,
                        "logic_str": (
                            "existsL(andL("
                            f'{concept_name}("o"), '
                            f'candidate_{candidate_index}(path="o")'
                            "))"
                        ),
                        "logic_label": torch.tensor(
                            [obj.labels[concept_name]], dtype=torch.long, device=device
                        ),
                    }
                )
                active_names_by_row.append(example.active_concepts)
    return rows, active_names_by_row


def build_dynamic_constraint_example(
    device="cpu",
    learning_rate=0.05,
    beta=1.0,
    executable_weight=1.0,
    global_weight=1.0,
):
    """Build one reusable graph and its combined-loss ``InferenceProgram``."""

    torch.manual_seed(7)
    reset_domiknows_state()
    examples = load_mock_examples()
    shared_model = TinyConceptMLP().to(device)

    with Graph("tiny_dynamic_global_constraints") as graph:
        obj = Concept(name="object")
        concepts = {name: obj(name=name) for name in CONCEPT_VOCABULARY}
        max_objects = max(len(example.objects) for example in examples)
        candidates = tuple(obj(name=f"candidate_{index}") for index in range(max_objects))
        rules = {
            "red_implies_colored": ifL(
                concepts["red"]("x"),
                concepts["colored"](path="x"),
                name="red_implies_colored",
            ),
            "dog_implies_animal": ifL(
                concepts["dog"]("x"),
                concepts["animal"](path="x"),
                name="dog_implies_animal",
            ),
            "tree_implies_plant": ifL(
                concepts["tree"]("x"),
                concepts["plant"](path="x"),
                name="tree_implies_plant",
            ),
        }

    obj["features"] = FunctionalReaderSensor(
        keyword="features",
        forward=lambda data: torch.as_tensor(data, dtype=torch.float32, device=device),
    )
    obj["index"] = FunctionalReaderSensor(
        keyword="indices",
        forward=lambda data: torch.as_tensor(data, dtype=torch.long, device=device),
    )
    for concept_index, concept in enumerate(concepts.values()):
        obj[concept] = ModuleLearner(
            "features", module=ConceptView(shared_model, concept_index).to(device)
        )
    for candidate_index, candidate in enumerate(candidates):
        obj[candidate] = ModuleLearner(
            "index", module=CandidateClassifier(candidate_index).to(device)
        )

    rows, active_names_by_row = _compiled_rows(examples, device)
    namespace = dict(concepts)
    namespace.update({candidate.name: candidate for candidate in candidates})
    compiled = graph.compile_executable(
        rows,
        logic_keyword="logic_str",
        logic_label_keyword="logic_label",
        extra_namespace_values=namespace,
    )
    entries = tuple(
        (
            tuple(concepts[name] for name in active_names) + candidates,
            compiled[index],
        )
        for index, active_names in enumerate(active_names_by_row)
    )

    program = InferenceProgram(
        graph,
        SolverModel,
        poi=[obj, *concepts.values(), *candidates, graph.constraint],
        device=device,
        inferTypes=["local/argmax"],
        beta=beta,
        tnorm="P",
        include_global_constraint_loss=True,
        executable_constraint_loss_weight=executable_weight,
        global_constraint_loss_weight=global_weight,
    )
    optimizer = torch.optim.Adam(shared_model.parameters(), lr=learning_rate)
    program.opt = optimizer
    return DynamicConstraintContext(
        graph=graph,
        concepts=concepts,
        candidates=candidates,
        rules=rules,
        examples=examples,
        entries=entries,
        program=program,
        shared_model=shared_model,
        optimizer=optimizer,
    )


def active_rule_names(context, example):
    """Return the sole global rule enabled by one dynamic example."""

    context.graph.set_active_concepts(
        [context.concepts[name] for name in example.active_concepts]
        + list(context.candidates)
    )
    try:
        return tuple(name for name, rule in context.rules.items() if rule.active)
    finally:
        context.graph.set_active_concepts(None)


def measure_constraint_losses(context):
    """Read executable/global components from DomiKnowS's constraint model."""

    components = {"executable": [], "global": [], "combined": []}
    context.program.model.eval()
    try:
        for active_concepts, row in context.entries:
            context.graph.set_active_concepts(active_concepts)
            with torch.no_grad():
                _, _, _, builder = context.program.model(row)
                loss, *_ = context.program.cmodel(builder)
            components["executable"].append(
                float(context.program.cmodel.last_executable_loss)
            )
            components["global"].append(float(context.program.cmodel.last_global_loss))
            components["combined"].append(float(loss))
    finally:
        context.graph.set_active_concepts(None)
    return {name: sum(values) / len(values) for name, values in components.items()}


def global_constraint_gradient_norm(context):
    """Prove that graph-global loss reaches the shared MLP parameters."""

    active_concepts, row = context.entries[0]
    context.graph.set_active_concepts(active_concepts)
    cmodel = context.program.cmodel
    old_executable_weight = cmodel.executable_constraint_loss_weight
    old_global_weight = cmodel.global_constraint_loss_weight
    context.program.model.zero_grad(set_to_none=True)
    try:
        cmodel.executable_constraint_loss_weight = 0.0
        cmodel.global_constraint_loss_weight = 1.0
        _, _, _, builder = context.program.model(row)
        global_loss, *_ = cmodel(builder)
        global_loss.backward()
        return sum(
            float(parameter.grad.detach().abs().sum())
            for parameter in context.shared_model.parameters()
            if parameter.grad is not None
        )
    finally:
        cmodel.executable_constraint_loss_weight = old_executable_weight
        cmodel.global_constraint_loss_weight = old_global_weight
        context.program.model.zero_grad(set_to_none=True)
        context.graph.set_active_concepts(None)


def concept_accuracy(context):
    """Evaluate the MLP against every mock atomic-concept label."""

    correct = 0
    total = 0
    context.shared_model.eval()
    device = next(context.shared_model.parameters()).device
    with torch.no_grad():
        for example in context.examples:
            features = torch.tensor(
                [obj.features for obj in example.objects], dtype=torch.float32, device=device
            )
            scores = context.shared_model.network(features)
            for concept_name in example.active_concepts:
                predicted = (
                    scores[:, CONCEPT_VOCABULARY.index(concept_name)] >= 0
                ).long().cpu().tolist()
                expected = [obj.labels[concept_name] for obj in example.objects]
                correct += sum(int(left == right) for left, right in zip(predicted, expected))
                total += len(expected)
    return 100.0 * correct / total


def global_constraint_accuracy(context):
    """Measure hard implication satisfaction on all six mock objects."""

    satisfied = 0
    total = 0
    context.shared_model.eval()
    device = next(context.shared_model.parameters()).device
    with torch.no_grad():
        for example in context.examples:
            source_name, target_name = example.active_concepts
            features = torch.tensor(
                [obj.features for obj in example.objects], dtype=torch.float32, device=device
            )
            scores = context.shared_model.network(features)
            source = scores[:, CONCEPT_VOCABULARY.index(source_name)] >= 0
            target = scores[:, CONCEPT_VOCABULARY.index(target_name)] >= 0
            satisfied += int((~source | target).sum().item())
            total += source.numel()
    return 100.0 * satisfied / total


def train_to_overfit(context=None, epochs=30, device="cpu"):
    """Train through ``program.train`` and return before/after diagnostics."""

    if context is None:
        context = build_dynamic_constraint_example(device=device)
    before_parameters = tuple(
        parameter.detach().clone() for parameter in context.shared_model.parameters()
    )
    before = {
        "loss": measure_constraint_losses(context),
        "concept_accuracy": concept_accuracy(context),
        "constraint_accuracy": global_constraint_accuracy(context),
    }
    context.program.opt = context.optimizer
    context.program.train(
        ActiveConceptDataset(context.graph, context.entries),
        warmup_epochs=0,
        constraint_epochs=epochs,
        batch_size=1,
        device=device,
    )
    after = {
        "loss": measure_constraint_losses(context),
        "concept_accuracy": concept_accuracy(context),
        "constraint_accuracy": global_constraint_accuracy(context),
    }
    parameters_changed = any(
        not torch.equal(old.cpu(), parameter.detach().cpu())
        for old, parameter in zip(before_parameters, context.shared_model.parameters())
    )
    context.graph.set_active_concepts(None)
    return context, before, after, parameters_changed


if __name__ == "__main__":
    built = build_dynamic_constraint_example()
    active = {
        example.example_id: active_rule_names(built, example)
        for example in built.examples
    }
    print(f"active_rules={active}")
    print(f"global_gradient_norm={global_constraint_gradient_norm(built):.6f}")
    built, before_metrics, after_metrics, changed = train_to_overfit(built)
    print(f"parameters_changed={changed}")
    print(f"before={before_metrics}")
    print(f"after={after_metrics}")
