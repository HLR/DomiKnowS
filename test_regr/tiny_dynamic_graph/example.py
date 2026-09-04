"""Rebuild a small graph per instance while sharing one neural learner."""

from dataclasses import dataclass

import torch

from domiknows.graph import Concept, Graph, Relation
from domiknows.graph.dataNode import DataNode, DataNodeBuilder
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import ModuleLearner
from domiknows.sensor.pytorch.sensors import FunctionalReaderSensor
from domiknows.solver.ilpOntSolverFactory import ilpOntSolverFactory


CONCEPT_VOCABULARY = ("red", "dog", "cat", "tree")


@dataclass(frozen=True)
class DynamicExampleSpec:
    example_id: str
    active_concepts: tuple
    query_concepts: tuple
    features: tuple
    answer_index: int


EXAMPLES = (
    DynamicExampleSpec(
        "red_dog",
        ("red", "dog", "cat"),
        ("red", "dog"),
        ((1, 1, 0, 0), (1, 0, 1, 0), (0, 1, 0, 1)),
        0,
    ),
    DynamicExampleSpec(
        "cat_tree",
        ("red", "cat", "tree"),
        ("cat", "tree"),
        ((1, 0, 0, 1), (0, 0, 1, 1), (0, 1, 1, 0)),
        1,
    ),
)


class SharedConceptClassifier(torch.nn.Module):
    """One small parameter matrix shared by every dynamically created graph."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(len(CONCEPT_VOCABULARY), len(CONCEPT_VOCABULARY))
        with torch.no_grad():
            self.linear.weight.copy_(2.0 * torch.eye(len(CONCEPT_VOCABULARY)))
            self.linear.bias.fill_(-0.5)

    def concept_logits(self, features, concept_index):
        score = self.linear(features.float())[:, concept_index]
        return torch.stack((-score, score), dim=-1)


class ConceptView(torch.nn.Module):
    """Expose one output of the shared classifier as a DomiKnowS predicate."""

    def __init__(self, shared_model, concept_index):
        super().__init__()
        self.shared_model = shared_model
        self.concept_index = concept_index

    def forward(self, features):
        return self.shared_model.concept_logits(features, self.concept_index)


class CandidateClassifier(torch.nn.Module):
    def __init__(self, candidate_index):
        super().__init__()
        self.candidate_index = candidate_index

    def forward(self, indices):
        score = (indices.view(-1) == self.candidate_index).float() * 8.0 - 4.0
        return torch.stack((-score, score), dim=-1)


@dataclass
class DynamicGraphContext:
    graph_name: str
    active_concepts: tuple
    dataset: object
    program: InferenceProgram


def reset_domiknows_state():
    Graph.clear()
    Concept.clear()
    Relation.clear()
    DataNode.clear()
    DataNodeBuilder.clear()
    ilpOntSolverFactory.clear()


def build_graph_for_example(spec, shared_model, device="cpu"):
    """Create a fresh schema containing only the concepts needed by ``spec``."""

    reset_domiknows_state()
    graph_name = f"tiny_dynamic_{spec.example_id}"
    with Graph(graph_name) as graph:
        obj = Concept(name="obj")
        concepts = {name: obj(name=name) for name in spec.active_concepts}
        candidates = [obj(name=f"candidate_{index}") for index in range(len(spec.features))]

    obj["features"] = FunctionalReaderSensor(
        keyword="features",
        forward=lambda data: torch.as_tensor(data, dtype=torch.float32, device=device),
    )
    obj["index"] = FunctionalReaderSensor(
        keyword="indices",
        forward=lambda data: torch.as_tensor(data, dtype=torch.long, device=device),
    )
    for name, concept in concepts.items():
        obj[concept] = ModuleLearner(
            "features",
            module=ConceptView(shared_model, CONCEPT_VOCABULARY.index(name)).to(device),
        )
    for index, candidate in enumerate(candidates):
        obj[candidate] = ModuleLearner("index", module=CandidateClassifier(index).to(device))

    namespace = dict(concepts)
    namespace.update({candidate.name: candidate for candidate in candidates})
    features = torch.tensor(spec.features, dtype=torch.float32, device=device)
    rows = []
    for candidate_index in range(len(spec.features)):
        predicates = [f'{spec.query_concepts[0]}("o")']
        predicates.extend(f'{name}(path="o")' for name in spec.query_concepts[1:])
        predicates.append(f'candidate_{candidate_index}(path="o")')
        rows.append(
            {
                "features": features,
                "indices": torch.arange(len(spec.features), device=device),
                "logic_str": f'existsL(andL({", ".join(predicates)}))',
                "logic_label": torch.tensor(
                    [int(candidate_index == spec.answer_index)], dtype=torch.long, device=device
                ),
            }
        )

    dataset = graph.compile_executable(
        rows,
        logic_keyword="logic_str",
        logic_label_keyword="logic_label",
        extra_namespace_values=namespace,
    )
    program = InferenceProgram(
        graph,
        SolverModel,
        poi=[obj, *concepts.values(), *candidates, graph.constraint],
        device=device,
        inferTypes=["local/argmax"],
        beta=1.0,
    )
    return DynamicGraphContext(graph_name, tuple(concepts), dataset, program)


def train_each_dynamic_graph(specs=EXAMPLES, device="cpu", learning_rate=1e-3):
    """Train each fresh graph while preserving shared parameters and optimizer state."""

    shared_model = SharedConceptClassifier().to(device)
    optimizer = torch.optim.Adam(shared_model.parameters(), lr=learning_rate)
    before = shared_model.linear.weight.detach().clone()
    graph_summaries = []
    for spec in specs:
        context = build_graph_for_example(spec, shared_model, device=device)
        context.program.opt = optimizer
        context.program.train(
            context.dataset,
            warmup_epochs=0,
            constraint_epochs=1,
            device=device,
            c_lr=learning_rate,
        )
        graph_summaries.append((context.graph_name, context.active_concepts))
    parameters_changed = not torch.equal(before.cpu(), shared_model.linear.weight.detach().cpu())
    return shared_model, tuple(graph_summaries), parameters_changed


def infer_each_dynamic_graph(shared_model, specs=EXAMPLES, device="cpu"):
    accuracies = {}
    graph_summaries = []
    for spec in specs:
        context = build_graph_for_example(spec, shared_model, device=device)
        accuracies[spec.example_id] = context.program.evaluate_condition(
            context.dataset, device=device
        )
        graph_summaries.append((context.graph_name, context.active_concepts))
    return accuracies, tuple(graph_summaries)


if __name__ == "__main__":
    model, trained_graphs, changed = train_each_dynamic_graph()
    results, inferred_graphs = infer_each_dynamic_graph(model)
    print(f"parameters_changed={changed}")
    print(f"trained_graphs={trained_graphs}")
    print(f"inferred_graphs={inferred_graphs}")
    print(f"accuracies={results}")
