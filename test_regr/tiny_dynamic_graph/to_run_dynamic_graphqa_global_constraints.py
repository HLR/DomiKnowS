"""Unexecuted GraphQA-scale stress workload for dynamic global constraints.

This module is intentionally excluded from tests. Its CLI exits before graph
construction unless ``--confirm-run`` is supplied. See
``TO_RUN_large_dynamic_graphqa.md`` before attempting a run.
"""

import argparse
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import threading
import time

import psutil
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
from test_regr.GraphQA.dataset import load_kb_facts


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
    predicate: str
    source_symbol: str
    target_symbol: str
    projection: str


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
    anchor_fact: tuple
    neighborhood_facts: tuple


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
    kb_facts: tuple
    epoch_timings: list = field(default_factory=list)
    item_records: list = field(default_factory=list)
    evaluation_records: dict = field(default_factory=dict)
    compiled_loss_calculator: object = None


def _json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")


def dataset_fingerprint(context):
    """Hash every generated datum except training-only configuration values."""

    digest = hashlib.sha256()
    digest.update(_json_bytes([asdict(rule) for rule in context.rule_specs]))
    digest.update(_json_bytes(context.kb_facts))
    for example in context.examples:
        digest.update(_json_bytes({
            "example_id": example.example_id,
            "active_concepts": example.active_concepts,
            "name_concept": example.name_concept,
            "attribute_concept": example.attribute_concept,
            "semantic_concept": example.semantic_concept,
            "capability_concept": example.capability_concept,
            "answer_indices": example.answer_indices,
            "anchor_fact": example.anchor_fact,
            "neighborhood_facts": example.neighborhood_facts,
        }))
        features = example.features.detach().cpu().contiguous()
        digest.update(str(features.dtype).encode("ascii"))
        digest.update(_json_bytes(tuple(features.shape)))
        digest.update(features.numpy().tobytes())
    return digest.hexdigest()


def _summary_stats(values):
    values = [float(value) for value in values]
    finite = [value for value in values if math.isfinite(value)]
    return {
        "count": len(values),
        "finite_count": len(finite),
        "nonfinite_count": len(values) - len(finite),
        "zero_count": sum(abs(value) <= 1e-12 for value in finite),
        "min": min(finite) if finite else None,
        "mean": sum(finite) / len(finite) if finite else None,
        "max": max(finite) if finite else None,
    }


def summarize_item_records(records):
    zero_loss_records = [
        record for record in records
        if math.isfinite(float(record["global_loss"]))
        and abs(float(record["global_loss"])) <= 1e-12
    ]
    return {
        "items": len(records),
        "executable_loss": _summary_stats(
            record["executable_loss"] for record in records),
        "global_loss": _summary_stats(
            record["global_loss"] for record in records),
        "active_rules": _summary_stats(
            record["active_rule_count"] for record in records),
        "compiled_active_rules": _summary_stats(
            record["compiled_active_rule_count"] for record in records),
        "inactive_rule_evaluations": sum(
            int(record["compiled_active_rule_count"] != record["active_rule_count"])
            for record in records
        ),
        # A zero fuzzy loss is valid when all active implications are
        # satisfied.  These intersections identify the two cases where zero
        # would instead be evidence of missing or incorrectly filtered work.
        "zero_global_loss_with_no_active_rules": sum(
            int(record["active_rule_count"] <= 0)
            for record in zero_loss_records
        ),
        "zero_global_loss_with_rule_count_mismatch": sum(
            int(record["compiled_active_rule_count"]
                != record["active_rule_count"])
            for record in zero_loss_records
        ),
    }


def global_loss_validation_gate(item_summary, gradient_diagnostic):
    """Distinguish legitimate satisfied-rule zeros from skipped evaluation."""

    probe_loss = float(gradient_diagnostic.get("global_loss", math.nan))
    probe_gradient = float(
        gradient_diagnostic.get("shared_mlp_gradient_norm", math.nan))
    probe_zero = math.isfinite(probe_loss) and abs(probe_loss) <= 1e-12
    probe_nonfinite = not math.isfinite(probe_loss)
    gradient_missing = (
        not math.isfinite(probe_gradient) or probe_gradient <= 0.0)
    no_active = int(item_summary["zero_global_loss_with_no_active_rules"])
    count_mismatch = int(
        item_summary["zero_global_loss_with_rule_count_mismatch"])
    nonfinite_items = int(item_summary["global_loss"]["nonfinite_count"])
    return {
        "zero_global_loss_items": int(item_summary["global_loss"]["zero_count"]),
        "zero_global_loss_with_no_active_rules": no_active,
        "zero_global_loss_with_rule_count_mismatch": count_mismatch,
        "violated_probe_zero_loss": bool(probe_zero),
        "violated_probe_nonfinite_loss": bool(probe_nonfinite),
        "violated_probe_missing_gradient": bool(gradient_missing),
        "nonfinite_global_loss_items": nonfinite_items,
        "passed": not any((
            no_active,
            count_mismatch,
            probe_zero,
            probe_nonfinite,
            gradient_missing,
            nonfinite_items,
        )),
    }


def evaluate_workload_predictions(context, threshold=0.5):
    """Evaluate synthetic labels and hard active-rule satisfaction."""

    positions = {
        name: index for index, name in enumerate(context.concepts)
    }
    rules_by_source = defaultdict(list)
    for rule in context.rule_specs:
        rules_by_source[rule.source].append(rule)

    concept_correct = 0
    concept_total = 0
    exact_set_correct = 0
    kb_satisfied = 0
    kb_total = 0
    active_rule_counts = []
    device = next(context.shared_model.parameters()).device
    was_training = context.shared_model.training
    context.shared_model.eval()
    context.shared_model.clear_cached_logits()
    with torch.no_grad():
        for example in context.examples:
            scores = context.shared_model.network(
                example.features.to(device).float())
            probabilities = torch.sigmoid(2.0 * scores)
            positives = torch.zeros(
                context.config.objects_per_example,
                dtype=torch.bool,
                device=device,
            )
            positives[list(example.answer_indices)] = True

            for concept_name in (
                    example.name_concept, example.attribute_concept):
                predicted = probabilities[:, positions[concept_name]] >= threshold
                concept_correct += int((predicted == positives).sum().item())
                concept_total += int(positives.numel())

            answer_probability = (
                probabilities[:, positions[example.capability_concept]]
                * probabilities[:, positions[example.attribute_concept]]
            )
            predicted_set = tuple(
                torch.nonzero(answer_probability >= threshold, as_tuple=False)
                .flatten().cpu().tolist()
            )
            exact_set_correct += int(predicted_set == example.answer_indices)

            active = set(example.active_concepts)
            active_rules = {
                rule.name: rule
                for source in active
                for rule in rules_by_source.get(source, ())
                if rule.target in active
            }
            active_rule_counts.append(len(active_rules))
            hard = probabilities >= threshold
            for rule in active_rules.values():
                source = hard[:, positions[rule.source]]
                target = hard[:, positions[rule.target]]
                kb_satisfied += int(((~source) | target).sum().item())
                kb_total += int(source.numel())

    if was_training:
        context.shared_model.train()
    return {
        "concept_correct": concept_correct,
        "concept_total": concept_total,
        "concept_accuracy": concept_correct / concept_total if concept_total else 0.0,
        "miota_exact_set_correct": exact_set_correct,
        "miota_exact_set_total": len(context.examples),
        "miota_exact_set_accuracy": (
            exact_set_correct / len(context.examples)
            if context.examples else 0.0
        ),
        "kb_rule_satisfied_groundings": kb_satisfied,
        "kb_rule_total_groundings": kb_total,
        "kb_rule_satisfaction": kb_satisfied / kb_total if kb_total else 0.0,
        "active_rules": _summary_stats(active_rule_counts),
        "active_rule_counts": active_rule_counts,
        "threshold": float(threshold),
    }


class ProcessResourceMonitor:
    """Sample process RSS and retain PyTorch CUDA allocator peaks."""

    def __init__(self, device, interval_seconds=0.05):
        self.device = str(device)
        self.interval_seconds = float(interval_seconds)
        self.process = psutil.Process()
        self.peak_rss_bytes = 0
        self._stop = threading.Event()
        self._thread = None

    @property
    def uses_cuda(self):
        return self.device.startswith("cuda") and torch.cuda.is_available()

    def _sample(self):
        while not self._stop.is_set():
            try:
                self.peak_rss_bytes = max(
                    self.peak_rss_bytes, self.process.memory_info().rss)
            except psutil.Error:
                pass
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        self.peak_rss_bytes = self.process.memory_info().rss
        if self.uses_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._thread = threading.Thread(
            target=self._sample, name="graphqa-resource-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2 * self.interval_seconds))
        try:
            self.peak_rss_bytes = max(
                self.peak_rss_bytes, self.process.memory_info().rss)
        except psutil.Error:
            pass
        if self.uses_cuda:
            torch.cuda.synchronize(self.device)

    def summary(self):
        result = {
            "peak_cpu_rss_bytes": int(self.peak_rss_bytes),
            "peak_cpu_rss_mib": self.peak_rss_bytes / (1024 ** 2),
            "peak_cuda_allocated_bytes": 0,
            "peak_cuda_allocated_mib": 0.0,
            "peak_cuda_reserved_bytes": 0,
            "peak_cuda_reserved_mib": 0.0,
        }
        if self.uses_cuda:
            allocated = torch.cuda.max_memory_allocated(self.device)
            reserved = torch.cuda.max_memory_reserved(self.device)
            result.update({
                "peak_cuda_allocated_bytes": int(allocated),
                "peak_cuda_allocated_mib": allocated / (1024 ** 2),
                "peak_cuda_reserved_bytes": int(reserved),
                "peak_cuda_reserved_mib": reserved / (1024 ** 2),
            })
        return result


def _atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def family_names(prefix, count):
    return tuple(f"{prefix}_{index:04d}" for index in range(count))


def concept_families(config):
    return {
        "name": family_names("name", config.name_concepts),
        "attribute": family_names("attribute", config.attribute_concepts),
        "semantic": family_names("semantic", config.semantic_concepts),
        "capability": family_names("capability", config.capability_concepts),
    }


def _stable_family_concept(families, family, symbol):
    """Map a real KB symbol to a stable learned-predicate slot."""

    names = families[family]
    digest = hashlib.sha256(f"{family}\0{symbol}".encode("utf-8")).digest()
    return names[int.from_bytes(digest[:8], "big") % len(names)]


def _fact_projections(families, fact):
    """Project one real KB edge into the stress model's predicate layers."""

    predicate, left, right = fact
    return (
        (
            "name_to_semantic",
            _stable_family_concept(families, "name", left),
            _stable_family_concept(families, "semantic", right),
        ),
        (
            "attribute_to_semantic",
            _stable_family_concept(families, "attribute", left),
            _stable_family_concept(families, "semantic", right),
        ),
        (
            "semantic_to_capability",
            _stable_family_concept(families, "semantic", right),
            _stable_family_concept(families, "capability", right),
        ),
        (
            "name_to_capability",
            _stable_family_concept(families, "name", left),
            _stable_family_concept(families, "capability", right),
        ),
    )


def _proof_ordered_facts(kb_facts):
    """Order real facts by deterministic breadth-first proof neighborhoods."""

    facts = tuple(kb_facts)
    facts_by_symbol = defaultdict(list)
    for index, (_predicate, left, right) in enumerate(facts):
        facts_by_symbol[left].append(index)
        facts_by_symbol[right].append(index)
    degree = {symbol: len(indices) for symbol, indices in facts_by_symbol.items()}
    seeds = sorted(
        range(len(facts)),
        key=lambda index: (
            -(degree[facts[index][1]] + degree[facts[index][2]]),
            facts[index],
            index,
        ),
    )

    ordered = []
    visited = set()
    for seed in seeds:
        if seed in visited:
            continue
        frontier = deque((seed,))
        while frontier:
            index = frontier.popleft()
            if index in visited:
                continue
            visited.add(index)
            fact = facts[index]
            ordered.append(fact)
            for symbol in fact[1:]:
                frontier.extend(
                    neighbor
                    for neighbor in facts_by_symbol[symbol]
                    if neighbor not in visited
                )
    return tuple(ordered)


def build_rule_specs(config, families, kb_facts=None):
    """Create constraints only from normalized facts in the real VQAR KB."""

    if kb_facts is None:
        kb_facts = load_kb_facts()
    kb_facts = tuple(dict.fromkeys(kb_facts))
    if not kb_facts:
        raise ValueError("No GraphQA/VQAR KB facts were found")

    specs = []
    seen_edges = set()

    def add(fact, projection, source, target):
        edge = (source, target)
        if source == target or edge in seen_edges:
            return False
        seen_edges.add(edge)
        predicate, source_symbol, target_symbol = fact
        specs.append(
            RuleSpec(
                name=f"kb_{projection}_{len(specs):05d}",
                source=source,
                target=target,
                family=predicate,
                predicate=predicate,
                source_symbol=source_symbol,
                target_symbol=target_symbol,
                projection=projection,
            )
        )
        return True

    for fact in _proof_ordered_facts(kb_facts):
        for projection, source, target in _fact_projections(families, fact):
            add(fact, projection, source, target)
            if len(specs) == config.kb_rules:
                return tuple(specs)
    if len(specs) != config.kb_rules:
        raise ValueError(
            f"real KB produced {len(specs)} unique projected rules; "
            f"requested {config.kb_rules}"
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
        self._cached_feature_key = None
        self._cached_feature_source = None
        self._cached_logits = None
        self._cache_generation = 0
        self.forward_calls = 0

    @staticmethod
    def _feature_key(features):
        return (
            features.data_ptr(),
            getattr(features, '_version', 0),
            tuple(features.shape),
            features.device,
            features.dtype,
        )

    def clear_cached_logits(self):
        self._cached_feature_key = None
        self._cached_feature_source = None
        self._cached_logits = None

    def logits(self, features):
        key = self._feature_key(features)
        if self._cached_feature_key == key and self._cached_logits is not None:
            return self._cached_logits

        logits = self.network(features.float())
        self.forward_calls += 1
        self._cache_generation += 1
        generation = self._cache_generation
        self._cached_feature_key = key
        # Retaining the source tensor prevents allocator pointer reuse from
        # making a later scene look like the currently cached one.
        self._cached_feature_source = features
        self._cached_logits = logits

        if logits.requires_grad:
            def clear_after_backward(gradient):
                if self._cache_generation == generation:
                    self.clear_cached_logits()
                return gradient

            logits.register_hook(clear_after_backward)
        return logits

    def concept_logits(self, features, concept_index):
        score = self.logits(features)[:, concept_index]
        return torch.stack((-score, score), dim=-1)


class ConceptView(torch.nn.Module):
    def __init__(self, shared_model, concept_index):
        super().__init__()
        self.shared_model = shared_model
        self.concept_index = concept_index

    def forward(self, features):
        return self.shared_model.concept_logits(features, self.concept_index)


def build_mock_examples(config, families, rule_specs):
    """Generate scenes whose active proof neighborhoods come from real KB facts."""

    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    rules_by_fact = defaultdict(list)
    fact_order = []
    for rule in rule_specs:
        fact = (rule.predicate, rule.source_symbol, rule.target_symbol)
        if fact not in rules_by_fact:
            fact_order.append(fact)
        rules_by_fact[fact].append(rule)

    required_projections = {
        "name_to_semantic",
        "attribute_to_semantic",
        "semantic_to_capability",
    }
    anchors = [
        fact for fact in fact_order
        if required_projections.issubset(
            {rule.projection for rule in rules_by_fact[fact]})
    ]
    if not anchors:
        raise ValueError("Real KB rules produced no complete proof-chain anchors")

    facts_by_symbol = defaultdict(list)
    for fact in fact_order:
        _predicate, left, right = fact
        facts_by_symbol[left].append(fact)
        facts_by_symbol[right].append(fact)

    def extract_neighborhood(anchor, target_concepts):
        active = {
            _stable_family_concept(families, "name", anchor[1]),
            _stable_family_concept(families, "attribute", anchor[1]),
            _stable_family_concept(families, "semantic", anchor[2]),
            _stable_family_concept(families, "capability", anchor[2]),
        }
        selected_facts = [anchor]
        selected_set = {anchor}
        visited_symbols = set()
        frontier = deque((anchor[1], anchor[2]))

        while frontier and len(active) < target_concepts:
            symbol = frontier.popleft()
            if symbol in visited_symbols:
                continue
            visited_symbols.add(symbol)
            for fact in facts_by_symbol[symbol]:
                contributed = False
                for rule in rules_by_fact[fact]:
                    additions = {rule.source, rule.target} - active
                    if not additions or len(active) + len(additions) <= target_concepts:
                        active.update((rule.source, rule.target))
                        contributed = True
                    if len(active) == target_concepts:
                        break
                if contributed and fact not in selected_set:
                    selected_set.add(fact)
                    selected_facts.append(fact)
                    frontier.extend((fact[1], fact[2]))
                if len(active) == target_concepts:
                    break
        return active, tuple(selected_facts)

    target_active_concepts = 4 + config.active_distractors_per_example
    anchor_catalog = []
    for anchor in anchors:
        active, neighborhood_facts = extract_neighborhood(
            anchor, target_active_concepts)
        if len(active) == target_active_concepts and len(neighborhood_facts) > 1:
            anchor_catalog.append((anchor, tuple(sorted(active)), neighborhood_facts))
    if not anchor_catalog:
        raise ValueError(
            "Real KB has no connected proof neighborhood large enough for "
            f"{target_active_concepts} active concepts"
        )

    examples = []
    for index in range(config.examples):
        anchor, active, neighborhood_facts = anchor_catalog[
            (index * 53) % len(anchor_catalog)]
        _predicate, left, right = anchor
        name = _stable_family_concept(families, "name", left)
        attribute = _stable_family_concept(families, "attribute", left)
        semantic_name = _stable_family_concept(families, "semantic", right)
        capability_name = _stable_family_concept(families, "capability", right)
        first_answer = index % config.objects_per_example
        second_answer = (first_answer + config.objects_per_example // 2) % config.objects_per_example
        answer_indices = tuple(sorted({first_answer, second_answer}))

        examples.append(
            MockGraphQAExample(
                example_id=f"kb_graphqa_{index:06d}",
                features=torch.randn(
                    config.objects_per_example,
                    config.feature_dim,
                    generator=generator,
                ),
                active_concepts=active,
                name_concept=name,
                attribute_concept=attribute,
                semantic_concept=semantic_name,
                capability_concept=capability_name,
                answer_indices=answer_indices,
                anchor_fact=anchor,
                neighborhood_facts=neighborhood_facts,
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
    kb_dir=None,
    kb_facts=None,
):
    """Construct the large union graph. This is intentionally not called on import."""

    if program_profile not in ("inference", "primal-dual-amortized"):
        raise ValueError(
            "program_profile must be 'inference' or 'primal-dual-amortized'"
        )

    torch.manual_seed(config.seed)
    reset_domiknows_state()
    kb_facts = tuple(dict.fromkeys(
        kb_facts if kb_facts is not None else load_kb_facts(kb_dir=kb_dir)
    ))
    families = concept_families(config)
    all_names = tuple(name for family in families.values() for name in family)
    rule_specs = build_rule_specs(config, families, kb_facts=kb_facts)
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
        kb_facts=kb_facts,
    )


def global_gradient_diagnostic(context, device="cpu"):
    """Backpropagate one deliberately violated real-KB implication."""

    example = context.examples[0]
    active = set(example.active_concepts)
    rule = next(
        rule for rule in context.rule_specs
        if rule.source in active and rule.target in active
    )
    positions = {name: index for index, name in enumerate(context.concepts)}
    output_layer = context.shared_model.network[-1]
    saved_weight = output_layer.weight.detach().clone()
    saved_bias = output_layer.bias.detach().clone()
    saved_forward_calls = context.shared_model.forward_calls
    context.program.model.zero_grad(set_to_none=True)
    context.shared_model.clear_cached_logits()
    try:
        with torch.no_grad():
            output_layer.weight.zero_()
            output_layer.bias.zero_()
            output_layer.bias[positions[rule.source]] = 6.0
            output_layer.bias[positions[rule.target]] = -6.0

        active_concepts, row = context.entries[0]
        context.graph.set_active_concepts(active_concepts)
        _loss, _metric, datanode, _builder = context.program.model(row)
        calculate_global = getattr(
            context.program.cmodel, "_calculate_global_constraint_loss", None)
        if calculate_global is not None:
            global_loss = calculate_global(datanode)
        else:
            constraint_losses = datanode.calculateLcLoss(
                tnorm="P", compiled=True, sampleGlobalLoss=False)
            global_terms = [
                result["lossTensor"].clamp(min=0).sum()
                for result in constraint_losses.values()
                if isinstance(result, dict)
                and result.get("executableName") is None
                and torch.is_tensor(result.get("lossTensor"))
            ]
            global_loss = sum(global_terms)
        global_loss.backward()
        squared_norm = sum(
            float(parameter.grad.detach().float().square().sum().item())
            for parameter in context.shared_model.parameters()
            if parameter.grad is not None
        )
        solver, _ = datanode.getILPSolver(
            conceptsRelations=datanode.collectConceptsAndRelations())
        context.compiled_loss_calculator = getattr(
            solver, "_compiled_loss_calculator", None)
        return {
            "rule": rule.name,
            "source": rule.source,
            "target": rule.target,
            "global_loss": float(global_loss.detach()),
            "shared_mlp_gradient_norm": math.sqrt(squared_norm),
            "source_probability": float(torch.sigmoid(torch.tensor(12.0))),
            "target_probability": float(torch.sigmoid(torch.tensor(-12.0))),
            "deliberately_violated": True,
        }
    finally:
        with torch.no_grad():
            output_layer.weight.copy_(saved_weight)
            output_layer.bias.copy_(saved_bias)
        context.program.model.zero_grad(set_to_none=True)
        context.shared_model.clear_cached_logits()
        context.shared_model.forward_calls = saved_forward_calls
        context.graph.set_active_concepts(None)


def _compiled_cache_info(context):
    calculator = context.compiled_loss_calculator
    return calculator.cache_info() if calculator is not None else {}


class _InstrumentedActiveConceptDataset(ActiveConceptDataset):
    def __init__(self, graph, entries, after_item):
        super().__init__(graph, entries)
        self.after_item = after_item

    def __iter__(self):
        try:
            for index, (active_concepts, row) in enumerate(self.entries):
                self.graph.set_active_concepts(active_concepts)
                yield row
                self.after_item(index)
        finally:
            self.graph.set_active_concepts(None)


def _checkpoint_payload(context, completed_epoch, c_session):
    copt = getattr(context.program, "copt", None)
    return {
        "schema_version": 2,
        "model": context.shared_model.state_dict(),
        "optimizer": context.optimizer.state_dict(),
        "constraint_model": context.program.cmodel.state_dict(),
        "constraint_optimizer": copt.state_dict() if copt is not None else None,
        "config": asdict(context.config),
        "program_profile": context.program_profile,
        "dataset_fingerprint": dataset_fingerprint(context),
        "completed_epoch": int(completed_epoch),
        "c_session": dict(c_session),
        "epoch_timings": tuple(context.epoch_timings),
        "item_records": tuple(context.item_records),
        "evaluation_records": context.evaluation_records,
        "shared_model_forward_calls": context.shared_model.forward_calls,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "python_rng_state": random.getstate(),
    }


def _restore_checkpoint(context, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved_config = dict(checkpoint.get("config", {}))
    current_config = asdict(context.config)
    saved_epochs = int(saved_config.pop("epochs", 0))
    current_epochs = int(current_config.pop("epochs"))
    if saved_config != current_config or current_epochs < saved_epochs:
        raise ValueError("Resume checkpoint configuration does not match this run")
    if checkpoint.get("program_profile") != context.program_profile:
        raise ValueError("Resume checkpoint program profile does not match this run")
    fingerprint = dataset_fingerprint(context)
    if checkpoint.get("dataset_fingerprint") != fingerprint:
        raise ValueError("Resume checkpoint dataset fingerprint does not match")

    context.shared_model.load_state_dict(checkpoint["model"])
    context.optimizer.load_state_dict(checkpoint["optimizer"])
    context.program.cmodel.load_state_dict(checkpoint["constraint_model"])
    context.epoch_timings[:] = list(checkpoint.get("epoch_timings", ()))
    context.item_records[:] = list(checkpoint.get("item_records", ()))
    context.evaluation_records.clear()
    context.evaluation_records.update(checkpoint.get("evaluation_records", {}))
    context.shared_model.forward_calls = int(
        checkpoint.get("shared_model_forward_calls", 0))
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    cuda_states = checkpoint.get("cuda_rng_state_all")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
    if checkpoint.get("python_rng_state") is not None:
        random.setstate(checkpoint["python_rng_state"])
    return {
        "completed_epoch": int(checkpoint.get("completed_epoch", 0)),
        "c_session": dict(checkpoint.get("c_session", {})),
        "constraint_optimizer": checkpoint.get("constraint_optimizer"),
    }


def train_stress_workload(context, device="cpu", output=None, resume=None):
    """Run instrumented, resumable DomiKnowS training."""

    context.program.opt = context.optimizer
    train_kwargs = {}
    resume_state = None
    if resume is not None:
        resume_state = _restore_checkpoint(context, resume, device)
    else:
        context.epoch_timings.clear()
        context.item_records.clear()
        context.evaluation_records.clear()

    if context.compiled_loss_calculator is None:
        gradient_diagnostic = global_gradient_diagnostic(context, device=device)
        context.evaluation_records.setdefault(
            "gradient_diagnostic", gradient_diagnostic)
    if "pre_training" not in context.evaluation_records:
        context.evaluation_records["pre_training"] = (
            evaluate_workload_predictions(context))

    completed_epoch = (
        resume_state["completed_epoch"] if resume_state is not None else 0)
    c_session = resume_state["c_session"] if resume_state is not None else {}
    constraint_optimizer_state = (
        resume_state["constraint_optimizer"] if resume_state is not None else None)
    elapsed_before = sum(
        float(record["epoch_seconds"]) for record in context.epoch_timings)
    training_start = time.perf_counter()
    epoch_start = training_start
    last_c_session = dict(c_session)
    previous_batched_constraints = int(
        _compiled_cache_info(context).get("batched_formula_constraints", 0))
    expected_active_counts = context.evaluation_records[
        "pre_training"]["active_rule_counts"]

    def after_item(item_index):
        nonlocal previous_batched_constraints
        cache_info = _compiled_cache_info(context)
        current_batched_constraints = int(
            cache_info.get("batched_formula_constraints", 0))
        compiled_active = current_batched_constraints - previous_batched_constraints
        previous_batched_constraints = current_batched_constraints
        cmodel = context.program.cmodel
        executable_loss = getattr(cmodel, "last_executable_loss", None)
        global_loss = getattr(cmodel, "last_global_loss", None)
        context.item_records.append({
            "epoch": len(context.item_records) // context.config.examples + 1,
            "item": item_index + 1,
            "executable_loss": (
                float(executable_loss) if executable_loss is not None else math.nan),
            "global_loss": (
                float(global_loss) if global_loss is not None else math.nan),
            "active_rule_count": int(expected_active_counts[item_index]),
            "compiled_active_rule_count": int(compiled_active),
        })

    checkpoint_path = output if output is not None else resume

    def report_epoch(program, epoch, phase, current_c_session):
        """Keep long stress runs observable and checkpoint every epoch."""

        nonlocal epoch_start, last_c_session
        now = time.perf_counter()
        cmodel = program.cmodel
        executable_loss = getattr(cmodel, "last_executable_loss", None)
        global_loss = getattr(cmodel, "last_global_loss", None)
        record = {
            "event": "epoch_complete",
            "epoch": epoch,
            "epochs": context.config.epochs,
            "phase": phase,
            "epoch_seconds": now - epoch_start,
            "elapsed_seconds": elapsed_before + now - training_start,
            "last_executable_loss": (
                float(executable_loss) if executable_loss is not None else None),
            "last_global_loss": (
                float(global_loss) if global_loss is not None else None),
        }
        context.epoch_timings.append(record)
        last_c_session = dict(current_c_session)
        if checkpoint_path:
            _atomic_torch_save(
                _checkpoint_payload(context, epoch, last_c_session),
                checkpoint_path,
            )
        print(json.dumps(record, sort_keys=True), flush=True)
        epoch_start = now

    if context.program_profile == "primal-dual-amortized":
        train_kwargs.update(c_warmup_iters=0, c_freq=1)
    if completed_epoch < context.config.epochs:
        context.program.train(
            _InstrumentedActiveConceptDataset(
                context.graph, context.entries, after_item),
            warmup_epochs=0,
            constraint_epochs=context.config.epochs,
            batch_size=1,
            device=device,
            epoch_end_callback=report_epoch,
            start_epoch=completed_epoch,
            resume_c_session=c_session if resume_state is not None else None,
            resume_copt_state=constraint_optimizer_state,
            persist_c_session=True,
            **train_kwargs,
        )
        completed_epoch = context.config.epochs
    context.graph.set_active_concepts(None)
    context.evaluation_records["post_training"] = (
        evaluate_workload_predictions(context))
    context.evaluation_records["item_summary"] = summarize_item_records(
        context.item_records)
    context.evaluation_records["global_loss_validation_gate"] = (
        global_loss_validation_gate(
            context.evaluation_records["item_summary"],
            context.evaluation_records["gradient_diagnostic"],
        )
    )
    context.evaluation_records["compiled_cache"] = _compiled_cache_info(context)
    if checkpoint_path:
        _atomic_torch_save(
            _checkpoint_payload(context, completed_epoch, last_c_session),
            checkpoint_path,
        )
    return context.evaluation_records


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
    parser.add_argument(
        "--kb-dir",
        help=(
            "VQAR knowledge_base directory containing is_a.facts and "
            "in_oa_rel.facts; defaults to the bundled real C2-C6 KB files"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    parser.add_argument(
        "--resume",
        help="Resume an epoch checkpoint written by --output.",
    )
    parser.add_argument(
        "--results-json",
        help="Atomically save the final machine-readable diagnostics.",
    )
    parser.add_argument(
        "--global-weight",
        type=float,
        help="Override global_weight without changing generated data.",
    )
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
    if args.global_weight is not None:
        config = replace(config, global_weight=args.global_weight)
    summary = workload_summary(config)
    summary["program_profile"] = args.program_profile
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not args.confirm_run:
        print("NOT RUN: pass --confirm-run only after reviewing TO_RUN_large_dynamic_graphqa.md")
        return 0

    construction_start = time.perf_counter()
    with ProcessResourceMonitor(args.device) as construction_monitor:
        context = build_stress_workload(
            config,
            device=args.device,
            program_profile=args.program_profile,
            kb_dir=args.kb_dir,
        )
    construction_seconds = time.perf_counter() - construction_start
    training_start = time.perf_counter()
    with ProcessResourceMonitor(args.device) as training_monitor:
        train_stress_workload(
            context,
            device=args.device,
            output=args.output,
            resume=args.resume,
        )
    training_seconds = time.perf_counter() - training_start
    cmodel = context.program.cmodel
    critic = getattr(cmodel, "dual_critic", None)
    diagnostics = {
        "construction_seconds": construction_seconds,
        "training_seconds": training_seconds,
        "epoch_training_seconds": sum(
            record["epoch_seconds"] for record in context.epoch_timings),
        "examples_per_second": (
            config.examples * config.epochs
            / sum(record["epoch_seconds"] for record in context.epoch_timings)
            if context.epoch_timings else None
        ),
        "construction_resources": construction_monitor.summary(),
        "training_resources": training_monitor.summary(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": args.device,
            "cuda_device": (
                torch.cuda.get_device_name(args.device)
                if str(args.device).startswith("cuda")
                and torch.cuda.is_available() else None
            ),
        },
        "config": asdict(config),
        "dataset_fingerprint": dataset_fingerprint(context),
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
        "shared_model_forward_calls": context.shared_model.forward_calls,
        "kb_fact_count": len(context.kb_facts),
        "kb_predicates": sorted({fact[0] for fact in context.kb_facts}),
        "proof_neighborhood_count": len(context.examples),
        "mean_neighborhood_facts": (
            sum(len(example.neighborhood_facts) for example in context.examples)
            / len(context.examples)
        ),
        "epoch_timings": context.epoch_timings,
        "mean_epoch_seconds": (
            sum(record["epoch_seconds"] for record in context.epoch_timings)
            / len(context.epoch_timings)
            if context.epoch_timings else None
        ),
        "evaluation": context.evaluation_records,
    }
    if args.results_json:
        _atomic_json_save(diagnostics, args.results_json)
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
