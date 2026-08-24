"""Unexecuted GraphQA-scale stress workload for dynamic global constraints.

This module is intentionally excluded from tests. Its CLI exits before graph
construction unless ``--confirm-run`` is supplied. See
``TO_RUN_large_dynamic_graphqa.md`` before attempting a run.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from domiknows.graph import Concept, Graph, ifL
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import ModuleLearner
from domiknows.sensor.pytorch.sensors import FunctionalReaderSensor

from test_regr.tiny_dynamic_graph.example import CandidateClassifier, reset_domiknows_state
from test_regr.tiny_dynamic_graph.example_dynamic_global_constraints import (
    ActiveConceptDataset,
)


DEFAULT_CONFIG = Path(__file__).with_name("mock_graphqa_stress_config.json")


@dataclass(frozen=True)
class StressConfig:
    seed: int
    feature_dim: int
    hidden_dim: int
    name_concepts: int
    attribute_concepts: int
    semantic_concepts: int
    capability_concepts: int
    kb_rules: int
    examples: int
    objects_per_example: int
    active_distractors_per_example: int
    epochs: int
    learning_rate: float
    beta: float
    executable_weight: float
    global_weight: float

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def learned_concepts(self):
        return (
            self.name_concepts
            + self.attribute_concepts
            + self.semantic_concepts
            + self.capability_concepts
        )

    @property
    def executable_rows(self):
        # Name labels + attribute labels for every object, then one set query.
        return self.examples * (2 * self.objects_per_example + 1)


@dataclass(frozen=True)
class RuleSpec:
    name: str
    source: str
    target: str
    family: str


@dataclass(frozen=True)
class MockGraphQAExample:
    example_id: str
    features: torch.Tensor
    active_concepts: tuple
    name_concept: str
    attribute_concept: str
    semantic_concept: str
    capability_concept: str
    answer_indices: tuple


@dataclass
class GraphQAStressContext:
    config: StressConfig
    graph: Graph
    concepts: dict
    candidates: tuple
    rules: dict
    rule_specs: tuple
    examples: tuple
    entries: tuple
    program: InferenceProgram
    shared_model: torch.nn.Module
    optimizer: torch.optim.Optimizer


def family_names(prefix, count):
    return tuple(f"{prefix}_{index:04d}" for index in range(count))


def concept_families(config):
    return {
        "name": family_names("name", config.name_concepts),
        "attribute": family_names("attribute", config.attribute_concepts),
        "semantic": family_names("semantic", config.semantic_concepts),
        "capability": family_names("capability", config.capability_concepts),
    }


def build_rule_specs(config, families):
    """Create mandatory proof chains, then deterministic KB distractor rules."""

    specs = []
    seen_edges = set()

    def add(source, target, family):
        edge = (source, target)
        if source == target or edge in seen_edges:
            return False
        seen_edges.add(edge)
        specs.append(
            RuleSpec(
                name=f"kb_{family}_{len(specs):05d}",
                source=source,
                target=target,
                family=family,
            )
        )
        return True

    semantic = families["semantic"]
    capability = families["capability"]
    for index, source in enumerate(families["name"]):
        add(source, semantic[index % len(semantic)], "name_to_semantic")
    for index, source in enumerate(semantic):
        add(source, capability[index % len(capability)], "semantic_to_capability")
        if index:
            add(source, semantic[(index - 1) // 2], "semantic_hierarchy")
    for index, source in enumerate(families["attribute"]):
        add(source, semantic[(index * 11 + 3) % len(semantic)], "attribute_to_semantic")

    mandatory_count = len(specs)
    if config.kb_rules < mandatory_count:
        raise ValueError(
            f"kb_rules={config.kb_rules} is smaller than {mandatory_count} mandatory rules"
        )

    source_pool = families["name"] + families["attribute"] + semantic
    target_pool = semantic + capability
    attempt = 0
    max_attempts = config.kb_rules * 20
    while len(specs) < config.kb_rules and attempt < max_attempts:
        source_index = attempt % len(source_pool)
        target_block = attempt // len(source_pool)
        source = source_pool[source_index]
        target = target_pool[
            (source_index * 37 + target_block * 101 + 29) % len(target_pool)
        ]
        add(source, target, "distractor")
        attempt += 1
    if len(specs) != config.kb_rules:
        raise ValueError(
            f"could only create {len(specs)} unique rules; requested {config.kb_rules}"
        )
    return tuple(specs)


class SharedGraphQAMLP(torch.nn.Module):
    """One shared local scorer whose outputs back every learned concept."""

    def __init__(self, config):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(config.feature_dim, config.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(config.hidden_dim, config.learned_concepts),
        )

    def concept_logits(self, features, concept_index):
        score = self.network(features.float())[:, concept_index]
        return torch.stack((-score, score), dim=-1)


class ConceptView(torch.nn.Module):
    def __init__(self, shared_model, concept_index):
        super().__init__()
        self.shared_model = shared_model
        self.concept_index = concept_index

    def forward(self, features):
        return self.shared_model.concept_logits(features, self.concept_index)


def build_mock_examples(config, families, rule_specs):
    """Generate GraphQA-like scenes and dynamically active proof neighborhoods."""

    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    semantic = families["semantic"]
    capability = families["capability"]
    examples = []
    for index in range(config.examples):
        name_index = index % len(families["name"])
        semantic_index = name_index % len(semantic)
        name = families["name"][name_index]
        attribute = families["attribute"][(index * 7 + 1) % len(families["attribute"])]
        semantic_name = semantic[semantic_index]
        capability_name = capability[semantic_index % len(capability)]
        first_answer = index % config.objects_per_example
        second_answer = (first_answer + config.objects_per_example // 2) % config.objects_per_example
        answer_indices = tuple(sorted({first_answer, second_answer}))

        active = {name, attribute, semantic_name, capability_name}
        rule_offset = (index * 53) % len(rule_specs)
        distractor_index = 0
        while len(active) < 4 + config.active_distractors_per_example:
            rule = rule_specs[(rule_offset + distractor_index) % len(rule_specs)]
            active.add(rule.source)
            active.add(rule.target)
            distractor_index += 1

        examples.append(
            MockGraphQAExample(
                example_id=f"mock_graphqa_{index:06d}",
                features=torch.randn(
                    config.objects_per_example,
                    config.feature_dim,
                    generator=generator,
                ),
                active_concepts=tuple(sorted(active)),
                name_concept=name,
                attribute_concept=attribute,
                semantic_concept=semantic_name,
                capability_concept=capability_name,
                answer_indices=answer_indices,
            )
        )
    return tuple(examples)


def rows_for_examples(examples, config, device):
    rows = []
    active_names_by_row = []
    for example in examples:
        features = example.features.to(device)
        indices = torch.arange(config.objects_per_example, device=device)
        positives = set(example.answer_indices)
        for concept_name in (example.name_concept, example.attribute_concept):
            for candidate_index in range(config.objects_per_example):
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
                            [int(candidate_index in positives)],
                            dtype=torch.long,
                            device=device,
                        ),
                    }
                )
                active_names_by_row.append(example.active_concepts)

        rows.append(
            {
                "features": features,
                "indices": indices,
                "logic_str": (
                    "miotaL(andL("
                    f'{example.capability_concept}("o"), '
                    f'{example.attribute_concept}(path="o")'
                    "), threshold=0.5, hard=False)"
                ),
                "logic_label": torch.tensor(
                    [int(i in positives) for i in range(config.objects_per_example)],
                    dtype=torch.float32,
                    device=device,
                ),
            }
        )
        active_names_by_row.append(example.active_concepts)
    return rows, active_names_by_row


def build_stress_workload(config, device="cpu"):
    """Construct the large union graph. This is intentionally not called on import."""

    torch.manual_seed(config.seed)
    reset_domiknows_state()
    families = concept_families(config)
    all_names = tuple(name for family in families.values() for name in family)
    rule_specs = build_rule_specs(config, families)
    examples = build_mock_examples(config, families, rule_specs)
    shared_model = SharedGraphQAMLP(config).to(device)

    with Graph("to_run_dynamic_graphqa_global_constraints") as graph:
        obj = Concept(name="object")
        concepts = {name: obj(name=name) for name in all_names}
        candidates = tuple(
            obj(name=f"candidate_{index}")
            for index in range(config.objects_per_example)
        )
        rules = {
            spec.name: ifL(
                concepts[spec.source]("o"),
                concepts[spec.target](path="o"),
                name=spec.name,
            )
            for spec in rule_specs
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

    rows, active_names_by_row = rows_for_examples(examples, config, device)
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
        beta=config.beta,
        tnorm="P",
        include_global_constraint_loss=True,
        executable_constraint_loss_weight=config.executable_weight,
        global_constraint_loss_weight=config.global_weight,
    )
    optimizer = torch.optim.Adam(shared_model.parameters(), lr=config.learning_rate)
    program.opt = optimizer
    return GraphQAStressContext(
        config=config,
        graph=graph,
        concepts=concepts,
        candidates=candidates,
        rules=rules,
        rule_specs=rule_specs,
        examples=examples,
        entries=entries,
        program=program,
        shared_model=shared_model,
        optimizer=optimizer,
    )


def train_stress_workload(context, device="cpu", output=None):
    """Run regular DomiKnowS training after the caller passes the CLI safety gate."""

    context.program.opt = context.optimizer
    context.program.train(
        ActiveConceptDataset(context.graph, context.entries),
        warmup_epochs=0,
        constraint_epochs=context.config.epochs,
        batch_size=1,
        device=device,
    )
    context.graph.set_active_concepts(None)
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": context.shared_model.state_dict(),
                "config": asdict(context.config),
            },
            output_path,
        )


def workload_summary(config):
    return {
        "learned_concepts": config.learned_concepts,
        "global_rules": config.kb_rules,
        "mock_examples": config.examples,
        "objects_per_example": config.objects_per_example,
        "compiled_executable_rows": config.executable_rows,
        "active_concepts_per_example_upper_bound": (
            4 + config.active_distractors_per_example + config.objects_per_example
        ),
        "epochs": config.epochs,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    parser.add_argument(
        "--confirm-run",
        action="store_true",
        help="Required: construct the large graph and start program.train.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = StressConfig.load(args.config)
    print(json.dumps(workload_summary(config), indent=2, sort_keys=True))
    if not args.confirm_run:
        print("NOT RUN: pass --confirm-run only after reviewing TO_RUN_large_dynamic_graphqa.md")
        return 0

    context = build_stress_workload(config, device=args.device)
    train_stress_workload(context, device=args.device, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
