"""Unexecuted GraphQA-scale stress workload for dynamic global constraints.

This module is intentionally excluded from tests. Its CLI exits before graph
construction unless ``--confirm-run`` is supplied. See
``TO_RUN_large_dynamic_graphqa.md`` before attempting a run.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import torch

from domiknows.graph import Concept, Graph, ifL
from domiknows.graph.executable import LogicDataset
from domiknows.program.lossprogram import InferenceProgram, PrimalDualProgram
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
    program: object
    program_profile: str
    shared_model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    compiled_executable_formulas: int
    reused_executable_rows: int
    parameterized_executable_templates: bool


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


def group_compiled_rows_by_example(compiled, config):
    """Combine one scene's executable labels into one model input.

    ``compile_executable`` intentionally creates one logical constraint per
    label.  The old workload then treated every one of those labels as a new
    optimizer item, repeating the shared MLP forward pass 25 times per scene.
    A grouped item retains all label-reader keys and selects all of its
    executable constraints together through LogicDataset's tuple-aware switch.
    """
    rows_per_example = 2 * config.objects_per_example + 1
    if len(compiled) != config.examples * rows_per_example:
        raise ValueError(
            f"Expected {config.examples * rows_per_example} compiled rows, "
            f"got {len(compiled)}"
        )

    grouped = []
    for start in range(0, len(compiled), rows_per_example):
        items = [compiled[index] for index in range(start, start + rows_per_example)]
        first = items[0]
        payload = {
            key: value
            for key, value in first.items()
            if key not in (LogicDataset.curr_lc_key, LogicDataset.do_switch_key)
            and not (isinstance(key, str) and key.startswith("_constraint_"))
            and key not in ("logic_str", "logic_label")
        }
        selected = []
        labels_by_name = {}
        bindings_by_name = {}
        for item in items:
            lc_name = item[LogicDataset.curr_lc_key]
            if lc_name not in selected:
                selected.append(lc_name)
            label_key = LogicDataset.KEYWORD_FMT.format(lc_name=lc_name)
            labels_by_name.setdefault(lc_name, []).append(item[label_key])
            item_bindings = item.get(LogicDataset.BINDINGS_KEY, {})
            bindings_by_name.setdefault(lc_name, []).extend(
                item_bindings.get(lc_name, ()))

        for lc_name, labels in labels_by_name.items():
            label_key = LogicDataset.KEYWORD_FMT.format(lc_name=lc_name)
            if len(labels) == 1:
                payload[label_key] = labels[0]
            else:
                normalized_labels = []
                for label in labels:
                    tensor = torch.as_tensor(label)
                    if tensor.dim() > 0 and tensor.shape[0] == 1:
                        tensor = tensor.squeeze(0)
                    normalized_labels.append(tensor)
                # The leading axis is DomiKnowS's data-item/batch axis. Keep
                # all template instances inside one constraint DataNode.
                payload[label_key] = torch.stack(
                    normalized_labels, dim=0).unsqueeze(0)
        if any(bindings_by_name.values()):
            payload[LogicDataset.BINDINGS_KEY] = {
                name: tuple(bindings)
                for name, bindings in bindings_by_name.items()
                if bindings
            }
        payload[LogicDataset.curr_lc_key] = tuple(selected)
        payload[LogicDataset.do_switch_key] = None
        grouped.append(payload)
    return tuple(grouped)


def build_stress_workload(
    config,
    device="cpu",
    program_profile="inference",
    parameterize_executable=True,
):
    """Construct the large union graph. This is intentionally not called on import."""

    if program_profile not in ("inference", "primal-dual-amortized"):
        raise ValueError(
            "program_profile must be 'inference' or 'primal-dual-amortized'"
        )

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
        deduplicate=True,
        parameterize=parameterize_executable,
    )
    grouped = group_compiled_rows_by_example(compiled, config)
    rows_per_example = 2 * config.objects_per_example + 1
    entries = tuple(
        (
            tuple(concepts[name] for name in active_names_by_row[index * rows_per_example])
            + candidates,
            grouped[index],
        )
        for index in range(config.examples)
    )
    program_kwargs = dict(
        poi=[obj, *concepts.values(), *candidates, graph.constraint],
        device=device,
        inferTypes=["local/argmax"],
        beta=config.beta,
        tnorm="P",
        compile_lc=True,
    )
    if program_profile == "inference":
        program = InferenceProgram(
            graph,
            SolverModel,
            include_global_constraint_loss=True,
            executable_constraint_loss_weight=config.executable_weight,
            global_constraint_loss_weight=config.global_weight,
            **program_kwargs,
        )
    else:
        # This profile directly exercises compiled groundingFeatures through a
        # per-grounding DualCritic. Executable constraints remain in each scene
        # as labels/model inputs, but are excluded from the global dual system.
        global_constraint_names = {rule.lcName for rule in rules.values()}
        excluded_constraint_names = tuple(
            name
            for key, lc in graph.allLogicalConstrainsRecursive
            if key not in global_constraint_names
            and getattr(lc, "lcName", key) not in global_constraint_names
            for name in (key, getattr(lc, "lcName", None))
            if name is not None
        )
        program = PrimalDualProgram(
            graph,
            SolverModel,
            dual_granularity="amortized",
            exclude_constraints=excluded_constraint_names,
            **program_kwargs,
        )
    # LearningBasedProgram places the prediction model from its constructor
    # arguments; LossProgram.to additionally moves constraint-side parameters
    # such as the amortized DualCritic and its multiplier bounds.
    program.to(device)
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
        program_profile=program_profile,
        shared_model=shared_model,
        optimizer=optimizer,
        compiled_executable_formulas=compiled.unique_constraint_count,
        reused_executable_rows=compiled.reused_constraint_count,
        parameterized_executable_templates=compiled.parameterized,
    )


def train_stress_workload(context, device="cpu", output=None):
    """Run regular DomiKnowS training after the caller passes the CLI safety gate."""

    context.program.opt = context.optimizer
    train_kwargs = {}
    if context.program_profile == "primal-dual-amortized":
        # Ensure even a one-epoch smoke profile executes the critic rather than
        # spending all of its few items in primal-only warmup.
        train_kwargs.update(c_warmup_iters=0, c_freq=1)
    context.program.train(
        ActiveConceptDataset(context.graph, context.entries),
        warmup_epochs=0,
        constraint_epochs=context.config.epochs,
        batch_size=1,
        device=device,
        **train_kwargs,
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
        "optimizer_items_per_epoch": config.examples,
        "executable_rows_per_item": 2 * config.objects_per_example + 1,
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
        "--program-profile",
        choices=("inference", "primal-dual-amortized"),
        default="inference",
        help=(
            "inference keeps the combined executable/global objective; "
            "primal-dual-amortized directly exercises compiled groundingFeatures "
            "with a per-grounding DualCritic"
        ),
    )
    parser.add_argument(
        "--confirm-run",
        action="store_true",
        help="Required: construct the large graph and start program.train.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = StressConfig.load(args.config)
    summary = workload_summary(config)
    summary["program_profile"] = args.program_profile
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not args.confirm_run:
        print("NOT RUN: pass --confirm-run only after reviewing TO_RUN_large_dynamic_graphqa.md")
        return 0

    construction_start = time.perf_counter()
    context = build_stress_workload(
        config, device=args.device, program_profile=args.program_profile
    )
    construction_seconds = time.perf_counter() - construction_start
    training_start = time.perf_counter()
    train_stress_workload(context, device=args.device, output=args.output)
    training_seconds = time.perf_counter() - training_start
    cmodel = context.program.cmodel
    critic = getattr(cmodel, "dual_critic", None)
    diagnostics = {
        "construction_seconds": construction_seconds,
        "training_seconds": training_seconds,
        "optimizer_items_per_epoch": len(context.entries),
        "compiled_executable_formulas": context.compiled_executable_formulas,
        "reused_executable_rows": context.reused_executable_rows,
        "parameterized_executable_templates": (
            context.parameterized_executable_templates
        ),
        "compile_lc": bool(getattr(cmodel, "compile_lc", False)),
        "dual_granularity": getattr(cmodel, "dual_granularity", None),
        "dual_critic_parameters": (
            sum(parameter.numel() for parameter in critic.parameters())
            if critic is not None else 0
        ),
        "last_executable_loss": (
            float(cmodel.last_executable_loss)
            if hasattr(cmodel, "last_executable_loss") else None
        ),
        "last_global_loss": (
            float(cmodel.last_global_loss)
            if hasattr(cmodel, "last_global_loss") else None
        ),
        "graph_reset": context.graph._active_concepts is None,
    }
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
