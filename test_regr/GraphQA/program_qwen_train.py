"""Train GraphQA predicate families through DomiKnowS InferenceProgram.

This is the CLEVR-style path: VQAR task -> bounded GraphQA facts ->
graph.compile_executable -> InferenceProgram(..., SolverModel) -> program.train.
The learned Qwen heads predict relation-family labels; concrete GraphQA
predicates are exposed as DomiKnowS child predicates through FunctionalSensor
slices, so executable iota/query logic can reuse the same predictions.
"""

from __future__ import annotations

import argparse
import functools
import json
import random
from pathlib import Path

import torch

from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.metric import MacroAverageTracker
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import EdgeSensor, ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.sensors import FunctionalReaderSensor, FunctionalSensor, JointReaderSensor

from .dataset import DEFAULT_VQAR_ROOT, discover_vqar_dataset, load_kb_facts, load_vqar_tasks, vqar_task_to_graphqa_instance
from .execution import create_candidate_membership_instance, create_executable_instance, materialize_bounded_facts
from .oracle import answer_object, answer_objects
from .graph import OBJECT_SYMBOL_RELATIONS, alias_values, canonical_relation, collect_kb_relations, collect_object_relations, create_graphqa_graph, safe_name
from .modules import GraphQAPredicateClassifier, label_spaces
from .train_predicate_classifier import choose_default_task_path, filter_kb_facts_for_instance

DEFAULT_MODEL = "/localscratch/premsrit/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DEFAULT_OUTPUT = Path("/egr/research-hlr2/premsrit/GraphQA/models/qwen3_8b_graphqa_domiknows_program.pt")
NO_RELATION_LABEL = "NoRelation"


class GraphQASolverModel(SolverModel):
    """SolverModel with supervised CE warmup for DomiKnowS program.train."""

    def __init__(self, *args, **kwargs):
        # Historical GraphQA commands passed this flag through InferenceProgram,
        # but DomiKnowS forwards unknown kwargs into SolverModel before CModel.
        # Consume it here so old commands fail safe without requiring a new API.
        kwargs.pop("include_global_constraint_loss", None)
        kwargs.pop("global_constraint_weight", None)
        kwargs.setdefault("loss", MacroAverageTracker(NBCrossEntropyLoss()))
        super().__init__(*args, **kwargs)


class BinaryOracleLearner(torch.nn.Module):
    """Turn a 0/1 tensor into two-column logits for fixed concepts."""

    def __init__(self, confidence=8.0):
        super().__init__()
        self.confidence = float(confidence)

    def forward(self, labels):
        labels = torch.as_tensor(labels, dtype=torch.long, device=labels.device if torch.is_tensor(labels) else None).view(-1)
        yes = labels.float() * self.confidence
        no = (1 - labels).float() * self.confidence
        return torch.stack([no, yes], dim=-1)


class MulticlassOracleLearner(torch.nn.Module):
    """High-confidence deterministic logits for fixed KB predicate families."""

    def __init__(self, num_labels, confidence=8.0):
        super().__init__()
        self.num_labels = int(num_labels)
        self.confidence = float(confidence)

    def forward(self, labels):
        labels = torch.as_tensor(labels, dtype=torch.long, device=labels.device if torch.is_tensor(labels) else None).view(-1)
        logits = torch.full((labels.numel(), self.num_labels), -self.confidence, dtype=torch.float32, device=labels.device)
        valid = (labels >= 0) & (labels < self.num_labels)
        if valid.any():
            logits[valid, labels[valid]] = self.confidence
        return logits


class GraphQAFamilyLearner(torch.nn.Module):
    """View one shared GraphQAPredicateClassifier as one relation-family learner."""

    def __init__(self, shared: GraphQAPredicateClassifier, kind: str):
        super().__init__()
        self.shared = shared
        self.kind = kind

    def forward(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = list(prompts)
        if not prompts:
            label_count = {
                "object_symbol": len(self.shared.object_symbol_labels),
                "symbol_pair": len(self.shared.symbol_pair_labels),
                "object_pair": max(1, len(self.shared.object_pair_labels)),
            }[self.kind]
            return torch.empty((0, label_count), dtype=torch.float32, device=self.shared.device_name)
        logits = self.shared.forward_examples([{"kind": self.kind, "prompt": prompt} for prompt in prompts])
        return _mask_logits_to_allowed_labels(logits, prompts, _labels_for_kind(self.shared, self.kind))


class GraphQAAtomicPredicateLearner(torch.nn.Module):
    """Expose one family-logit column as a CLEVR-style binary learner."""

    def __init__(self, index):
        super().__init__()
        self.index = int(index)

    def forward(self, logits):
        return _binary_logits_from_family(logits, self.index)


def _labels_for_kind(shared, kind):
    if kind == "object_symbol":
        return shared.object_symbol_labels
    if kind == "symbol_pair":
        return shared.symbol_pair_labels
    if kind == "object_pair":
        return shared.object_pair_labels
    raise ValueError(f"Unknown GraphQA learner kind: {kind!r}")


def _allowed_labels_from_prompt(prompt):
    marker = "Allowed labels:"
    for line in str(prompt).splitlines():
        if line.startswith(marker):
            text = line[len(marker):].strip().rstrip(".")
            return [item.strip() for item in text.split(",") if item.strip()]
    return None


def _mask_logits_to_allowed_labels(logits, prompts, labels):
    if logits.numel() == 0:
        return logits
    label_to_index = {label: index for index, label in enumerate(labels)}
    masked = logits.clone()
    for row, prompt in enumerate(prompts):
        allowed = _allowed_labels_from_prompt(prompt)
        if not allowed:
            continue
        allowed_indices = [label_to_index[label] for label in allowed if label in label_to_index]
        if not allowed_indices:
            continue
        row_mask = torch.full_like(masked[row], -1.0e4)
        row_mask[allowed_indices] = 0.0
        masked[row] = masked[row] + row_mask
    return masked


def build_graphqa_context(instances, args):
    schema = _load_label_schema(getattr(args, "schema_path", None))
    graph_instances = _augment_instances_for_schema(instances, schema)
    ctx = create_graphqa_graph(graph_instances, include_global_constraints=args.global_consistency)
    spaces = _ensure_no_relation_labels(schema or label_spaces(graph_instances))
    spaces["answer_object"] = [str(value) for value in ctx.object_values]
    spaces["_require_oracle_clean"] = not args.allow_oracle_inconsistent_executables
    spaces["_enable_set_answer_execution"] = args.enable_set_answer_execution
    spaces["_single_answer_only"] = args.single_answer_only
    spaces["_boolean_answer_execution"] = args.boolean_answer_execution
    spaces["_max_set_answer_candidates"] = args.max_set_answer_candidates
    spaces["_max_set_answer_negatives"] = args.max_set_answer_negatives
    spaces["_max_object_symbol_candidates"] = args.max_object_symbol_candidates
    spaces["_max_object_pair_candidates"] = args.max_object_pair_candidates
    spaces["_oracle_kb_predicates"] = args.oracle_kb_predicates
    spaces["_include_global_constraints"] = args.global_consistency
    attach_program_train_sensors(ctx, spaces, args)
    return ctx, spaces


def _ensure_no_relation_labels(spaces):
    spaces = {key: list(value) for key, value in spaces.items()}
    for key in ("object_symbol", "symbol_pair", "object_pair"):
        labels = spaces.setdefault(key, [])
        if labels and NO_RELATION_LABEL not in labels:
            labels.append(NO_RELATION_LABEL)
    return spaces


def create_graphqa_program(ctx, args):
    poi = [
        ctx.scene,
        ctx.obj,
        ctx.answer_object,
        ctx.symbol,
        ctx.object_symbol_pair,
        ctx.symbol_pair,
        ctx.object_pair,
        ctx.object_symbol_relation,
        ctx.symbol_pair_relation,
        ctx.object_pair_relation,
        *ctx.object_symbol_relations.values(),
        *ctx.symbol_relations.values(),
        *ctx.object_relations.values(),
        *ctx.object_concepts.values(),
        ctx.graph.constraint,
    ]
    program = InferenceProgram(
        ctx.graph,
        GraphQASolverModel,
        poi=poi,
        device=args.device,
        inferTypes=["local/argmax"],
        beta=args.beta,
        include_global_constraint_loss=args.global_consistency,
        global_constraint_weight=args.beta_global,
    )
    # Match the CLEVR/Temporal executable-training path. SolverModel does not
    # accept tnorm as a constructor kwarg on this DomiKnowS branch, so apply it
    # to the constraint model after InferenceProgram construction. Product and
    # Lukasiewicz iotaL are differentiable; Goedel uses a hard argmax.
    if hasattr(program, "cmodel"):
        program.cmodel.tnorm = args.tnorm
        program.cmodel.counting_tnorm = getattr(program.cmodel, "counting_tnorm", None) or args.tnorm
    return program


def build_graphqa_program(instances, args):
    """Build a GraphQA program for legacy callers.

    Training code should compile executable datasets before calling
    create_graphqa_program, matching CLEVR/Temporal.
    """
    ctx, spaces = build_graphqa_context(instances, args)
    return None, ctx, create_graphqa_program(ctx, args), spaces


def attach_program_train_sensors(ctx, spaces, args):
    device = args.device
    ctx.scene["index"] = FunctionalReaderSensor(keyword="scene_indices", forward=lambda data: _tensor(data, device=device))
    ctx.obj["index"] = FunctionalReaderSensor(keyword="object_indices", forward=lambda data: _tensor(data, device=device))
    ctx.symbol["index"] = FunctionalReaderSensor(keyword="symbol_indices", forward=lambda data: _tensor(data, device=device))
    ctx.obj["ids"] = FunctionalReaderSensor(keyword="object_ids", forward=_safe_list)
    ctx.symbol["ids"] = FunctionalReaderSensor(keyword="symbol_ids", forward=_safe_list)
    ctx.scene["feature"] = FunctionalSensor(ctx.scene["index"], forward=lambda idx: idx.float().unsqueeze(-1))
    ctx.obj["feature"] = FunctionalSensor(ctx.obj["index"], forward=lambda idx: idx.float().unsqueeze(-1))
    ctx.obj["object_domain_label"] = FunctionalSensor(
        ctx.obj["index"], forward=lambda idx: torch.ones_like(idx, dtype=torch.long)
    )
    ctx.obj[ctx.object_domain] = ModuleLearner(
        "object_domain_label", module=BinaryOracleLearner(), device=device
    )
    ctx.symbol["feature"] = FunctionalSensor(ctx.symbol["index"], forward=lambda idx: idx.float().unsqueeze(-1))

    ctx.obj[ctx.scene_contains_obj] = EdgeSensor(
        ctx.obj["index"], ctx.scene["index"], relation=ctx.scene_contains_obj,
        forward=lambda obj_idx, _scene_idx: torch.ones_like(obj_idx).unsqueeze(-1),
    )
    ctx.symbol[ctx.scene_contains_symbol] = EdgeSensor(
        ctx.symbol["index"], ctx.scene["index"], relation=ctx.scene_contains_symbol,
        forward=lambda sym_idx, _scene_idx: torch.ones_like(sym_idx).unsqueeze(-1),
    )

    for raw_name, concept in _constant_concepts(ctx, prefix="object_"):
        ctx.obj[f"is_{raw_name}_label"] = FunctionalSensor(
            ctx.obj["ids"], forward=lambda ids, _name=raw_name, _device=device: _binary_membership(ids, _name, _device)
        )
        ctx.obj[concept] = ModuleLearner(f"is_{raw_name}_label", module=BinaryOracleLearner(), device=device)
    for raw_name, concept in _constant_concepts(ctx, prefix="symbol_"):
        ctx.symbol[f"is_{raw_name}_label"] = FunctionalSensor(
            ctx.symbol["ids"], forward=lambda ids, _name=raw_name, _device=device: _binary_membership(ids, _name, _device)
        )
        ctx.symbol[concept] = ModuleLearner(f"is_{raw_name}_label", module=BinaryOracleLearner(), device=device)

    ctx.object_symbol_pair["index"] = FunctionalReaderSensor(keyword="object_symbol_pair_indices", forward=lambda data: _tensor(data, device=device))
    ctx.symbol_pair["index"] = FunctionalReaderSensor(keyword="symbol_pair_indices", forward=lambda data: _tensor(data, device=device))
    ctx.object_pair["index"] = FunctionalReaderSensor(keyword="object_pair_indices", forward=lambda data: _tensor(data, device=device))

    # Explicit pair rows and joint endpoint maps mirror the working
    # Temporal/CLEVR relation representation. This materializes relationLinks
    # needed by path-based queryL/iotaL constraints without detaching logits.
    ctx.object_symbol_pair[
        ctx.object_symbol_object.reversed,
        ctx.object_symbol_symbol.reversed,
    ] = JointReaderSensor(
        ctx.object_symbol_pair["index"],
        keyword="object_symbol_relation_maps",
        forward=lambda *_args, data: data,
    )
    ctx.symbol_pair[
        ctx.symbol_pair_src.reversed,
        ctx.symbol_pair_dst.reversed,
    ] = JointReaderSensor(
        ctx.symbol_pair["index"],
        keyword="symbol_pair_relation_maps",
        forward=lambda *_args, data: data,
    )
    ctx.object_pair[
        ctx.object_pair_src.reversed,
        ctx.object_pair_dst.reversed,
    ] = JointReaderSensor(
        ctx.object_pair["index"],
        keyword="object_pair_relation_maps",
        forward=lambda *_args, data: data,
    )

    shared = GraphQAPredicateClassifier(
        model_path=args.model_path,
        object_symbol_labels=spaces["object_symbol"],
        symbol_pair_labels=spaces["symbol_pair"],
        object_pair_labels=spaces["object_pair"],
        device=device,
        freeze_backbone=args.freeze_backbone,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=[m.strip() for m in str(args.lora_target_modules).split(",") if m.strip()],
        max_length=args.max_length,
        encode_batch_size=args.encode_batch_size,
    )
    # Execution-only training should learn from compiled queryL/iotaL labels,
    # not from oracle local predicate labels.  Keep labels in the data for
    # oracle KB rows and diagnostics, but do not attach them as supervised
    # DomiKnowS labels when this flag is enabled.
    shared.execution_label_only = bool(getattr(args, "execution_label_only", False))

    _attach_family(ctx.object_symbol_pair, ctx.object_symbol_relation, ctx.object_symbol_relations, spaces["object_symbol"], "object_symbol", shared, device)
    _attach_family(ctx.symbol_pair, ctx.symbol_pair_relation, ctx.symbol_relations, spaces["symbol_pair"], "symbol_pair", shared, device, oracle_family=spaces.get("_oracle_kb_predicates", True))
    _attach_family(ctx.object_pair, ctx.object_pair_relation, ctx.object_relations, spaces["object_pair"], "object_pair", shared, device)


def _attach_family(base_concept, family_concept, child_concepts, labels, kind, shared, device, oracle_family=False):
    if not labels:
        return
    base_concept[f"{kind}_prompts"] = FunctionalReaderSensor(keyword=f"{kind}_prompts", forward=lambda data: list(data))
    base_concept[f"{kind}_label"] = FunctionalReaderSensor(
        keyword=f"{kind}_label",
        forward=lambda data, _device=device: torch.as_tensor(data, dtype=torch.long, device=_device),
    )
    supervise_local = not getattr(shared, "execution_label_only", False) and not oracle_family
    if supervise_local:
        # Supervise the actual multiclass family parent concept used by direct
        # evaluation, while keeping the string-key labels available to child labels.
        base_concept[family_concept] = FunctionalReaderSensor(
            keyword=f"{kind}_label",
            label=True,
            forward=lambda data, _device=device: torch.as_tensor(data, dtype=torch.long, device=_device),
        )
    if oracle_family:
        # KB facts are fixed symbolic evidence.  They are represented as
        # deterministic logits from the per-instance symbol_pair labels, not as
        # trainable ModuleLearner outputs.  Thus executable constraints can use
        # KB predicates to guide learned visual grounding without learning KB.
        base_concept[family_concept] = FunctionalSensor(
            base_concept[f"{kind}_label"],
            forward=lambda labels, n=len(labels): _multiclass_logits_from_labels(labels, n),
        )
    else:
        base_concept[family_concept] = ModuleLearner(
            f"{kind}_prompts",
            module=GraphQAFamilyLearner(shared, kind),
            device=device,
        )
    label_to_index = {label: index for index, label in enumerate(labels)}
    for label, concept in child_concepts.items():
        if label == NO_RELATION_LABEL or label not in label_to_index:
            continue
        index = label_to_index[label]
        if oracle_family:
            base_concept[concept] = FunctionalSensor(
                base_concept[family_concept],
                forward=lambda logits, idx=index: _binary_logits_from_family(logits, idx),
            )
        else:
            base_concept[concept] = ModuleLearner(
                base_concept[family_concept],
                module=GraphQAAtomicPredicateLearner(index),
                device=device,
            )
        if supervise_local:
            # CLEVR-style local supervision is attached to the concrete child
            # predicate, not only to the open-vocabulary family parent.
            base_concept[concept] = FunctionalSensor(
                base_concept[f"{kind}_label"],
                forward=lambda labels, idx=index: (labels.view(-1) == idx).long(),
                label=True,
            )


def _multiclass_logits_from_labels(labels, num_labels, confidence=8.0):
    labels = torch.as_tensor(labels, dtype=torch.long, device=labels.device if torch.is_tensor(labels) else None).view(-1)
    logits = torch.full((labels.numel(), int(num_labels)), -float(confidence), dtype=torch.float32, device=labels.device)
    valid = (labels >= 0) & (labels < int(num_labels))
    if valid.any():
        logits[valid, labels[valid]] = float(confidence)
    return logits


def _binary_logits_from_family(logits, index):
    positive = logits[:, index]
    if logits.shape[1] == 1:
        negative = -positive
    else:
        mask = torch.ones(logits.shape[1], dtype=torch.bool, device=logits.device)
        mask[index] = False
        negative = torch.logsumexp(logits[:, mask], dim=-1)
    return torch.stack([negative, positive], dim=-1)


def compile_program_train_dataset(instances, ctx, spaces, device="cpu"):
    data = []
    skipped_without_answer = 0
    expanded_set_answer_rows = 0
    skipped_multi_answer = 0
    for instance in instances:
        if spaces.get("_single_answer_only") and len(instance.get("expected_answers", []) or []) != 1:
            skipped_multi_answer += 1
            continue
        converted_rows = _to_program_train_data(instance, spaces, device=device)
        if not converted_rows:
            skipped_without_answer += 1
            continue
        for converted in converted_rows:
            if converted.get("logic_label") is None:
                skipped_without_answer += 1
                continue
            if converted.get("answer_mode") == "candidate_membership":
                expanded_set_answer_rows += 1
            data.append(converted)
    if skipped_multi_answer:
        print(f"skipped_multi_answer={skipped_multi_answer}", flush=True)
    if skipped_without_answer:
        print(f"skipped_without_executable_answer={skipped_without_answer}", flush=True)
    if expanded_set_answer_rows:
        print(f"expanded_set_answer_rows={expanded_set_answer_rows}", flush=True)
    if not data:
        raise ValueError("No GraphQA executable examples have a supported answer label")
    return ctx.graph.compile_executable(
        data,
        logic_keyword="logic_str",
        logic_label_keyword="logic_label",
        extra_namespace_values=ctx.namespace,
    )


def _to_program_train_data(instance, spaces, device="cpu"):
    converted_rows = _create_executable_rows(instance, spaces)
    return [_populate_program_train_data(instance, converted, spaces, device=device) for converted in converted_rows]


def _create_executable_rows(instance, spaces):
    expected_answers = [str(answer) for answer in instance.get("expected_answers", [])]
    oracle_answers = [str(answer) for answer in answer_objects(instance)]
    require_clean = spaces.get("_require_oracle_clean", True)

    if len(expected_answers) == 1:
        answer = str(instance.get("expected_answer") or expected_answers[0])
        if require_clean and oracle_answers != [answer]:
            return []
        if spaces.get("_boolean_answer_execution"):
            candidates = [str(obj) for obj in instance.get("objects", [])]
            max_negatives = spaces.get("_max_set_answer_negatives")
            negatives = [candidate for candidate in candidates if candidate != answer]
            if max_negatives is not None and int(max_negatives) >= 0:
                negatives = negatives[: int(max_negatives)]
            return [
                create_candidate_membership_instance(instance, candidate, candidate == answer)
                for candidate in [answer, *negatives]
            ]
        instance_for_execution = dict(instance)
        instance_for_execution["expected_answer"] = answer
        return [create_executable_instance(instance_for_execution, answer_label_space=spaces.get("answer_object"))]

    if not spaces.get("_enable_set_answer_execution", True):
        return []
    if not expected_answers:
        return []
    if require_clean and set(oracle_answers) != set(expected_answers):
        return []

    candidates = [str(obj) for obj in instance.get("objects", [])]
    max_candidates = spaces.get("_max_set_answer_candidates")
    if max_candidates is not None and int(max_candidates) > 0 and len(candidates) > int(max_candidates):
        return []

    gold = set(expected_answers)
    max_negatives = spaces.get("_max_set_answer_negatives")
    if max_negatives is not None and int(max_negatives) >= 0:
        positives = [candidate for candidate in candidates if candidate in gold]
        negatives = [candidate for candidate in candidates if candidate not in gold]
        rng_key = "|".join(
            str(instance.get(key, ""))
            for key in ("source_question_id", "question_id", "source_image_id", "image_id")
        )
        rng = random.Random(rng_key)
        rng.shuffle(negatives)
        candidates = positives + negatives[: int(max_negatives)]

    return [
        create_candidate_membership_instance(instance, candidate, candidate in gold)
        for candidate in candidates
    ]


def _populate_program_train_data(instance, converted, spaces, device="cpu"):
    facts = list(converted.get("facts", [])) or materialize_bounded_facts(instance)
    objects = [str(obj) for obj in instance.get("objects", [])]
    symbols = [str(symbol) for symbol in instance.get("symbols", []) if not str(symbol).startswith("__")]
    object_index = {obj: i for i, obj in enumerate(objects)}
    symbol_index = {symbol: i for i, symbol in enumerate(symbols)}

    true_object_symbol = {}
    true_symbol_pair = {}
    true_object_pair = {}
    for pred, left, right in facts:
        pred = canonical_relation(pred)
        left = str(left)
        right = str(right)
        if pred in OBJECT_SYMBOL_RELATIONS and left in object_index and right in symbol_index and pred in spaces["object_symbol"]:
            true_object_symbol.setdefault((left, right), []).append(pred)
        elif pred in spaces["symbol_pair"] and left in symbol_index and right in symbol_index:
            true_symbol_pair.setdefault((left, right), []).append(pred)
        elif pred in spaces["object_pair"] and left in object_index and right in object_index:
            true_object_pair.setdefault((left, right), []).append(pred)

    object_symbol_rows = _candidate_object_symbol_rows(instance, objects, symbols, true_object_symbol, spaces, device=device)
    symbol_pair_rows = _candidate_symbol_pair_rows(instance, symbols, true_symbol_pair, spaces, device=device)
    object_pair_rows = _candidate_object_pair_rows(instance, objects, true_object_pair, spaces, device=device)

    if not object_symbol_rows and objects and symbols and spaces["object_symbol"]:
        object_symbol_rows.append((0, 0, spaces["object_symbol"][0], _dummy_prompt("object-symbol")))
    if not symbol_pair_rows and symbols and spaces["symbol_pair"]:
        symbol_pair_rows.append((0, 0, spaces["symbol_pair"][0], _dummy_prompt("symbol-pair")))
    if not object_pair_rows and objects and spaces["object_pair"]:
        object_pair_rows.append((0, 0, spaces["object_pair"][0], _dummy_prompt("object-pair")))

    object_symbol_object_link = _link_matrix(len(object_symbol_rows), len(objects), [(i, row[0]) for i, row in enumerate(object_symbol_rows)], device)
    object_symbol_symbol_link = _link_matrix(len(object_symbol_rows), len(symbols), [(i, row[1]) for i, row in enumerate(object_symbol_rows)], device)
    symbol_pair_src_link = _link_matrix(len(symbol_pair_rows), len(symbols), [(i, row[0]) for i, row in enumerate(symbol_pair_rows)], device)
    symbol_pair_dst_link = _link_matrix(len(symbol_pair_rows), len(symbols), [(i, row[1]) for i, row in enumerate(symbol_pair_rows)], device)
    object_pair_src_link = _link_matrix(len(object_pair_rows), len(objects), [(i, row[0]) for i, row in enumerate(object_pair_rows)], device)
    object_pair_dst_link = _link_matrix(len(object_pair_rows), len(objects), [(i, row[1]) for i, row in enumerate(object_pair_rows)], device)

    converted.update({
        "scene_indices": torch.tensor([0], dtype=torch.long, device=device),
        "object_indices": torch.arange(len(objects), dtype=torch.long, device=device),
        "symbol_indices": torch.arange(len(symbols), dtype=torch.long, device=device),
        "object_ids": objects,
        "symbol_ids": symbols,
        "object_symbol_pair_indices": torch.arange(len(object_symbol_rows), dtype=torch.long, device=device),
        "symbol_pair_indices": torch.arange(len(symbol_pair_rows), dtype=torch.long, device=device),
        "object_pair_indices": torch.arange(len(object_pair_rows), dtype=torch.long, device=device),
        "object_symbol_object_link": object_symbol_object_link,
        "object_symbol_symbol_link": object_symbol_symbol_link,
        "symbol_pair_src_link": symbol_pair_src_link,
        "symbol_pair_dst_link": symbol_pair_dst_link,
        "object_pair_src_link": object_pair_src_link,
        "object_pair_dst_link": object_pair_dst_link,
        "object_symbol_relation_maps": (object_symbol_object_link.float(), object_symbol_symbol_link.float()),
        "symbol_pair_relation_maps": (symbol_pair_src_link.float(), symbol_pair_dst_link.float()),
        "object_pair_relation_maps": (object_pair_src_link.float(), object_pair_dst_link.float()),
        "object_symbol_prompts": [row[3] for row in object_symbol_rows],
        "symbol_pair_prompts": [row[3] for row in symbol_pair_rows],
        "object_pair_prompts": [row[3] for row in object_pair_rows],
        "object_symbol_label": torch.tensor([spaces["object_symbol"].index(row[2]) for row in object_symbol_rows], dtype=torch.long, device=device),
        "symbol_pair_label": torch.tensor([spaces["symbol_pair"].index(row[2]) for row in symbol_pair_rows], dtype=torch.long, device=device),
        "object_pair_label": torch.tensor([spaces["object_pair"].index(row[2]) for row in object_pair_rows], dtype=torch.long, device=device),
    })
    if converted.get("logic_label") is not None:
        converted["logic_label"] = torch.LongTensor([int(converted["logic_label"])]).to(device)
    return converted


def _candidate_object_symbol_rows(instance, objects, symbols, true_by_pair, spaces, device="cpu"):
    labels = spaces.get("object_symbol", [])
    if not labels:
        return []
    visual_labels = ["Name", "Attribute"]
    if spaces.get("_include_global_constraints"):
        visual_labels.extend(["ObjectType", "ObjectCategory"])
    allowed_labels = [label for label in (*visual_labels, NO_RELATION_LABEL) if label in labels]
    selected_symbols = _bounded_symbol_candidates(instance, symbols, true_by_pair, spaces.get("_max_object_symbol_candidates"))
    pair_keys = [(obj, sym) for obj in objects for sym in selected_symbols]
    pair_keys.extend(key for key in true_by_pair if key not in pair_keys)
    rows = []
    seen = set()
    for obj, sym in pair_keys:
        if obj not in objects or sym not in symbols or (obj, sym) in seen:
            continue
        seen.add((obj, sym))
        label = _choose_label(true_by_pair.get((obj, sym), []), allowed_labels)
        rows.append((objects.index(obj), symbols.index(sym), label, _object_symbol_prompt(instance, obj, sym, allowed_labels)))
    return rows


def _candidate_symbol_pair_rows(instance, symbols, true_by_pair, spaces, device="cpu"):
    labels = spaces.get("symbol_pair", [])
    if not labels:
        return []
    rows = []
    seen = set()
    for src, dst in true_by_pair:
        if src not in symbols or dst not in symbols:
            continue
        # One graph node per extensional KB fact preserves multiple predicates
        # over the same symbol endpoints without turning any KB relation into a learner.
        for label in true_by_pair.get((src, dst), []):
            fact_key = (src, dst, label)
            if label not in labels or fact_key in seen:
                continue
            seen.add(fact_key)
            rows.append((symbols.index(src), symbols.index(dst), label, _symbol_pair_prompt(instance, src, dst, labels)))
    return rows


def _candidate_object_pair_rows(instance, objects, true_by_pair, spaces, device="cpu"):
    labels = spaces.get("object_pair", [])
    if not labels:
        return []
    allowed_labels = _allowed_object_pair_labels(instance, labels)
    max_pairs = spaces.get("_max_object_pair_candidates")
    pair_keys = [(src, dst) for src in objects for dst in objects if src != dst]
    if max_pairs is not None and int(max_pairs) > 0:
        keep = []
        needed = set(true_by_pair) | _query_object_pair_candidates(instance)
        for key in pair_keys:
            if key in needed:
                keep.append(key)
        for key in pair_keys:
            if len(keep) >= int(max_pairs):
                break
            if key not in keep:
                keep.append(key)
        pair_keys = keep
    pair_keys.extend(key for key in true_by_pair if key not in pair_keys)
    rows = []
    seen = set()
    for src, dst in pair_keys:
        if src not in objects or dst not in objects or (src, dst) in seen:
            continue
        seen.add((src, dst))
        label = _choose_label(true_by_pair.get((src, dst), []), allowed_labels)
        rows.append((objects.index(src), objects.index(dst), label, _object_pair_prompt(instance, src, dst, allowed_labels)))
    return rows


def _bounded_symbol_candidates(instance, symbols, true_by_pair, max_candidates):
    selected = []
    needed = set(_symbols_needed_by_query(instance))
    needed.update(sym for _obj, sym in true_by_pair)
    for sym in symbols:
        if sym in needed and sym not in selected:
            selected.append(sym)
    if max_candidates is None or int(max_candidates) <= 0:
        max_candidates = len(symbols)
    for sym in symbols:
        if len(selected) >= int(max_candidates):
            break
        if sym not in selected:
            selected.append(sym)
    return selected


def _query_relation_labels(instance):
    labels = []
    query = instance.get("query", {})
    for conditions in [query.get("conditions", [])] + list(query.get("alternatives", [])):
        for pred, _left, right in conditions:
            pred = canonical_relation(pred)
            if pred in {"RelationFrom", "RelationTo"} and isinstance(right, (list, tuple)) and len(right) == 2:
                rel = canonical_relation(right[0])
                if rel is not None and rel not in labels:
                    labels.append(rel)
    return labels


def _allowed_object_pair_labels(instance, label_space):
    labels = [label for label in _query_relation_labels(instance) if label in label_space]
    if NO_RELATION_LABEL in label_space and NO_RELATION_LABEL not in labels:
        labels.append(NO_RELATION_LABEL)
    if not labels:
        labels = [NO_RELATION_LABEL] if NO_RELATION_LABEL in label_space else list(label_space)
    return labels


def _query_object_pair_candidates(instance):
    out = set()
    query = instance.get("query", {})
    for conditions in [query.get("conditions", [])] + list(query.get("alternatives", [])):
        for pred, _left, right in conditions:
            pred = canonical_relation(pred)
            if pred in {"RelationFrom", "RelationTo"} and isinstance(right, (list, tuple)) and len(right) == 2:
                _rel, object_ids = right
                for obj in object_ids:
                    obj = str(obj)
                    for candidate in instance.get("objects", []):
                        candidate = str(candidate)
                        if pred == "RelationFrom":
                            out.add((obj, candidate))
                        else:
                            out.add((candidate, obj))
    return out


def _choose_label(true_labels, label_space):
    for label in ("Name", "Attribute", "ObjectType", "ObjectCategory"):
        if label in true_labels and label in label_space:
            return label
    for label in true_labels:
        if label in label_space:
            return label
    return NO_RELATION_LABEL if NO_RELATION_LABEL in label_space else label_space[0]


def evaluate_family_accuracy(dataset, ctx, program):
    metrics = {}
    was_training = program.model.training
    program.model.eval()
    with torch.no_grad():
        for kind, base, family in [
            ("object_symbol", ctx.object_symbol_pair, ctx.object_symbol_relation),
            ("symbol_pair", ctx.symbol_pair, ctx.symbol_pair_relation),
            ("object_pair", ctx.object_pair, ctx.object_pair_relation),
        ]:
            correct = 0
            total = 0
            for row in dataset:
                program.model(row)
                logits = base[family](row)
                labels = row.get(f"{kind}_label")
                if labels is None or logits is None or logits.numel() == 0:
                    continue
                labels = labels.to(logits.device).view(-1)
                preds = logits.argmax(dim=-1).view(-1)
                n = min(preds.numel(), labels.numel())
                if n:
                    correct += int((preds[:n] == labels[:n]).sum().item())
                    total += int(n)
            metrics[f"{kind}_correct"] = correct
            metrics[f"{kind}_total"] = total
            metrics[f"{kind}_acc"] = correct / total if total else 0.0
    if was_training:
        program.model.train()
    return metrics


def load_instances(args):
    if args.task_path is None:
        discovered = discover_vqar_dataset(args.root)
        task_path = choose_default_task_path(discovered)
    else:
        task_path = args.task_path
    tasks = load_vqar_tasks(task_path, limit=args.limit)
    global_kb_facts = [] if args.no_kb else load_kb_facts(kb_dir=args.kb_dir)
    kb_index = _index_kb_facts(global_kb_facts)
    instances = []
    failures = []
    for index, task in enumerate(tasks):
        try:
            instance = vqar_task_to_graphqa_instance(task, kb_facts=[])
            kb_facts = _filter_kb_facts_for_instance_indexed(
                instance,
                kb_index,
                max_depth=args.kb_depth,
                max_extra_kg=args.max_extra_kg_facts,
            )
            instance["kb_facts"] = kb_facts
            # Keep the DomiKnowS graph vocabulary bounded. KB facts are still
            # available for bounded materialization and prompt rows, but we do
            # not create one DomiKnowS concept for every open-vocabulary KG
            # symbol across the full training split.
            scene_symbols = {
                str(right)
                for pred, _left, right in instance.get("visual_facts", [])
                if canonical_relation(pred) in OBJECT_SYMBOL_RELATIONS and right is not None
            }
            instance["symbols"] = sorted(
                scene_symbols
                | _symbols_needed_by_query(instance)
                | _kb_endpoint_symbols(kb_facts)
            )
            instance["facts"] = materialize_bounded_facts(instance)
            instances.append(instance)
        except Exception as exc:
            failures.append((index, type(exc).__name__, str(exc)))
    return task_path, instances, failures




def _index_kb_facts(kb_facts):
    type_by_src = {}
    extra_by_symbol = {}
    for pred, left, right in kb_facts:
        pred = canonical_relation(pred)
        fact = (pred, left, right)
        if pred == "TypeOf":
            type_by_src.setdefault(left, []).append(fact)
        else:
            extra_by_symbol.setdefault(left, []).append(fact)
            extra_by_symbol.setdefault(right, []).append(fact)
    return {"type_by_src": type_by_src, "extra_by_symbol": extra_by_symbol}


def _filter_kb_facts_for_instance_indexed(instance, kb_index, max_depth=2, max_extra_kg=256):
    needed = set()
    direct_kg_conditions = []
    for pred, _obj, symbol in instance.get("visual_facts", []):
        if canonical_relation(pred) in {"Name", "Attribute"} and symbol is not None:
            needed.add(symbol)
    query = instance.get("query", {})
    if query.get("target_type") not in (None, "__any_object__"):
        needed.update(alias_values("SemanticClass", query["target_type"]))
    for conditions in [query.get("conditions", [])] + list(query.get("alternatives", [])):
        for pred, _left, right in conditions:
            pred = canonical_relation(pred)
            if pred in {"Name", "ObjectType", "ObjectCategory"}:
                if right is not None:
                    needed.add(right)
            elif pred == "Attribute":
                if right is not None:
                    needed.update(alias_values("Attribute", right))
            elif pred == "SemanticClass":
                if right is not None:
                    needed.update(alias_values("SemanticClass", right))
            elif pred == "KG":
                _rel, dst = right
                direct_kg_conditions.append((canonical_relation(_rel), dst))
                needed.update(alias_values("SemanticClass", dst))
                needed.update(alias_values("Attribute", dst))

    filtered = []
    frontier = set(needed)
    seen = set()

    # Forward TypeOf closure supports Name -> ObjectType -> ObjectCategory.
    for _depth in range(max_depth):
        next_frontier = set()
        for src in frontier:
            for fact in kb_index["type_by_src"].get(src, []):
                if fact in seen:
                    continue
                _pred, _left, right = fact
                filtered.append(fact)
                seen.add(fact)
                next_frontier.add(right)
        frontier = next_frontier
        needed.update(next_frontier)

    # KG_Find(BLANK, rel, dst) needs the open-attribute edges pointing to the
    # queried destination, e.g. Has(zebra, stripes). Add those before broad
    # reverse TypeOf expansion consumes the max_extra_kg budget.
    extra_kg = 0
    for rel, dst in direct_kg_conditions:
        dst_aliases = alias_values("SemanticClass", dst) + alias_values("Attribute", dst)
        for dst_alias in list(dict.fromkeys(dst_aliases)):
            for fact in kb_index["extra_by_symbol"].get(dst_alias, []):
                if fact in seen:
                    continue
                fact_rel, _left, _right = fact
                if canonical_relation(fact_rel) != rel:
                    continue
                filtered.append(fact)
                seen.add(fact)
                extra_kg += 1
                if max_extra_kg is not None and extra_kg >= max_extra_kg:
                    return filtered

    # Reverse TypeOf closure keeps source symbols needed by Hypernym_Find and
    # KG_Find(BLANK, rel, dst) when the query names a higher-level class. This
    # branch can explode for generic classes such as ``object``, so it shares
    # the same max_extra_kg budget as the remaining loose KG evidence.
    reverse_type = {}
    for facts in kb_index["type_by_src"].values():
        for fact in facts:
            _pred, left, right = fact
            reverse_type.setdefault(right, []).append(fact)
    frontier = set(needed)
    for _depth in range(max_depth):
        next_frontier = set()
        for dst in frontier:
            for fact in reverse_type.get(dst, []):
                if fact in seen:
                    continue
                _pred, left, _right = fact
                filtered.append(fact)
                seen.add(fact)
                next_frontier.add(left)
                extra_kg += 1
                if max_extra_kg is not None and extra_kg >= max_extra_kg:
                    return filtered
        frontier = next_frontier
        needed.update(next_frontier)

    for symbol in list(needed):
        for fact in kb_index["extra_by_symbol"].get(symbol, []):
            if fact in seen:
                continue
            filtered.append(fact)
            seen.add(fact)
            extra_kg += 1
            if extra_kg >= max_extra_kg:
                return filtered
    return filtered

def _kb_endpoint_symbols(kb_facts):
    symbols = set()
    for fact in kb_facts or []:
        if not isinstance(fact, (list, tuple)) or len(fact) != 3:
            continue
        _pred, left, right = fact
        if left is not None:
            symbols.add(str(left))
        if right is not None:
            symbols.add(str(right))
    return symbols


def _symbols_needed_by_query(instance):
    symbols = set()
    query = instance.get("query", {})
    target = query.get("target_type")
    if target and target != "__any_object__":
        symbols.update(alias_values("SemanticClass", target))
    condition_groups = [query.get("conditions", [])]
    condition_groups.extend(query.get("alternatives", []))
    for conditions in condition_groups:
        for pred, _left, right in conditions:
            pred = canonical_relation(pred)
            if pred in {"Name", "ObjectType", "ObjectCategory"}:
                if right is not None:
                    symbols.add(right)
            elif pred == "Attribute":
                if right is not None:
                    symbols.update(alias_values("Attribute", right))
            elif pred == "SemanticClass":
                if right is not None:
                    symbols.update(alias_values("SemanticClass", right))
            elif pred == "KG" and isinstance(right, (list, tuple)) and len(right) == 2:
                _rel, dst = right
                if dst is not None:
                    symbols.update(alias_values("SemanticClass", dst))
                    symbols.update(alias_values("Attribute", dst))
    return symbols

def split_instances(instances, dev_fraction=0.1, seed=13):
    instances = list(instances)
    random.Random(seed).shuffle(instances)
    if len(instances) <= 1 or dev_fraction <= 0:
        return instances, []
    dev_size = min(max(1, int(round(len(instances) * dev_fraction))), len(instances) - 1)
    return instances[dev_size:], instances[:dev_size]


def _link_matrix(n_left, n_right, links, device):
    matrix = torch.zeros((n_left, n_right), dtype=torch.long, device=device)
    for left, right in links:
        matrix[int(left), int(right)] = 1
    return matrix


def _edge_from_matrix(_src_index, _dst_index, matrix):
    return matrix


def _constant_concepts(ctx, prefix):
    # Concept.sup is the owning Graph, not the immediate is_a parent. Use the
    # explicit mappings created by graph.py so answer and symbol subclasses are
    # always attached to their root DataNodes, matching CLEVR.
    if prefix == "object_":
        yield from ctx.object_concepts.items()
    elif prefix == "symbol_":
        yield from ctx.symbol_concepts.items()


def _binary_membership(ids, safe_value, device):
    return torch.tensor([1 if safe_name(value) == safe_value else 0 for value in ids], dtype=torch.long, device=device)


def _tensor(data, device="cpu"):
    if data is None:
        return torch.empty(0, dtype=torch.long, device=device)
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return torch.as_tensor(data, dtype=torch.long, device=device)


def _safe_list(data):
    return [] if data is None else list(data)


def _dummy_prompt(kind):
    return f"Dummy GraphQA {kind} predicate used only to keep DomiKnowS empty relation families buildable."


def _all_evidence_facts(instance):
    facts = []
    seen = set()
    for key in ("facts", "visual_facts", "kb_facts"):
        for fact in instance.get(key, []) or []:
            if not isinstance(fact, (list, tuple)) or len(fact) != 3:
                continue
            pred, left, right = fact
            item = (canonical_relation(pred), str(left), str(right))
            if item in seen:
                continue
            seen.add(item)
            facts.append(item)
    return facts


def _format_fact(fact):
    pred, left, right = fact
    return f"{pred}({left}, {right})"


def _format_relevant_facts(instance, *, left=None, right=None, max_facts=16, max_scan=4096):
    left = None if left is None else str(left)
    right = None if right is None else str(right)
    exact = []
    touching = []
    seen = set()
    scanned = 0

    # Prefer raw visual/KG evidence before the larger materialized closure.
    for key in ("visual_facts", "kb_facts", "facts"):
        for fact in instance.get(key, []) or []:
            if not isinstance(fact, (list, tuple)) or len(fact) != 3:
                continue
            pred, fact_left, fact_right = fact
            item = (canonical_relation(pred), str(fact_left), str(fact_right))
            if item in seen:
                continue
            seen.add(item)
            scanned += 1
            _pred, fact_left, fact_right = item
            matches_left = left is None or fact_left == left or fact_right == left
            matches_right = right is None or fact_left == right or fact_right == right
            if matches_left and matches_right:
                exact.append(item)
            elif matches_left or matches_right:
                touching.append(item)
            if len(exact) + len(touching) >= int(max_facts) or scanned >= int(max_scan):
                break
        if len(exact) + len(touching) >= int(max_facts) or scanned >= int(max_scan):
            break

    selected = (exact + touching)[: int(max_facts)]
    if not selected:
        return "None."
    suffix = "" if scanned < int(max_scan) else "\n... evidence scan capped"
    return "\n".join(_format_fact(fact) for fact in selected) + suffix


def _format_object_metadata(instance, obj):
    metadata = (instance.get("object_metadata") or {}).get(str(obj), {})
    if not metadata:
        return "None."
    lines = []
    if "image_id" in metadata:
        lines.append(f"image_id={metadata['image_id']}")
    if "image_url" in metadata:
        lines.append(f"image_url={metadata['image_url']}")
    if "bbox" in metadata:
        lines.append(f"bbox={metadata['bbox']}")
    feature = metadata.get("feature") or {}
    if feature:
        lines.append(
            "region_feature_summary="
            f"dim={feature.get('dim')} mean={feature.get('mean')} max={feature.get('max')} "
            f"nonzero={feature.get('nonzero')} head={feature.get('head')}"
        )
    return "\n".join(lines) if lines else "None."


def _object_symbol_prompt(instance, obj, symbol, labels=None):
    labels = labels or ["Name", "ObjectType", "ObjectCategory", "Attribute", NO_RELATION_LABEL]
    return "\n".join([
        "Classify one GraphQA visual predicate from the object region only.",
        "Do not use scene-graph labels, gold answers, or the executable query.",
        "Visual input representation for the object region:",
        _format_object_metadata(instance, obj),
        f"Object id: {obj}",
        f"Candidate concept: {symbol}",
        "Predicate question: what relation, if any, holds between this object region and the candidate concept?",
        f"Allowed labels: {', '.join(map(str, labels))}.",
    ])


def _symbol_pair_prompt(instance, src, dst, labels=None):
    labels = labels or ["TypeOf", NO_RELATION_LABEL]
    return "\n".join([
        "Classify the GraphQA symbol-pair KG predicate.",
        "Use the provided bounded scene/KG facts as evidence; do not infer from symbol names alone.",
        f"Source symbol: {src}",
        f"Destination symbol: {dst}",
        f"Query: {instance.get('query', {})}",
        "Relevant facts:",
        _format_relevant_facts(instance, left=src, right=dst),
        f"Labels: {', '.join(map(str, labels))}.",
    ])


def _object_pair_prompt(instance, src, dst, labels=None):
    labels = labels or [NO_RELATION_LABEL]
    return "\n".join([
        "Classify one GraphQA visual relation predicate from two object regions only.",
        "Do not use scene-graph labels, gold answers, or the executable query.",
        "Visual input representation for the source object region:",
        _format_object_metadata(instance, src),
        "Visual input representation for the destination object region:",
        _format_object_metadata(instance, dst),
        f"Source object id: {src}",
        f"Destination object id: {dst}",
        "Predicate question: what visual relation, if any, holds from the source object to the destination object?",
        f"Allowed labels: {', '.join(map(str, labels))}.",
    ])


def parse_args():
    parser = argparse.ArgumentParser(description="Train GraphQA with DomiKnowS program.train and Qwen predicate-family learners.")
    parser.add_argument("--root", type=Path, default=DEFAULT_VQAR_ROOT)
    parser.add_argument("--task-path", type=Path, default=None)
    parser.add_argument("--kb-dir", type=Path, default=None)
    parser.add_argument("--no-kb", action="store_true")
    parser.add_argument("--kb-depth", type=int, default=2)
    parser.add_argument("--max-extra-kg-facts", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--schema-limit",
        type=int,
        default=None,
        help=(
            "Optional number of task instances used only to build the graph/label schema. "
            "Use this when evaluating a checkpoint on a smaller --limit so relation heads "
            "match the training-time schema."
        ),
    )
    parser.add_argument("--schema-path", type=Path, default=None, help="JSON label-space schema saved beside a GraphQA checkpoint.")
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--encode-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1, help="Backward-compatible alias for --warmup-epochs when unset.")
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--constraint-epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tnorm", default="G")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--beta-global", type=float, default=1.0, help="Weight applied to graph-level consistency loss inside closs.")
    parser.add_argument(
        "--global-consistency",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable graph-level grounding/KB consistency loss in addition to executable query loss.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-condition-eval", action="store_true")
    parser.add_argument("--enable-set-answer-execution", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--boolean-answer-execution",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train single-answer questions as candidate-wise Boolean executable assertions while retaining queryL for final inference.",
    )
    parser.add_argument(
        "--single-answer-only",
        action="store_true",
        help="Drop examples whose gold answer set has size other than one; disables multi-answer training/eval rows.",
    )
    parser.add_argument(
        "--save-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a checkpoint and schema after each warmup/constraint epoch.",
    )
    parser.add_argument(
        "--max-set-answer-candidates",
        type=int,
        default=32,
        help="Skip set-answer examples with more than this many candidate objects; use <=0 for no cap.",
    )
    parser.add_argument(
        "--max-set-answer-negatives",
        type=int,
        default=-1,
        help="For set-answer execution, keep all gold objects and at most this many negative candidate objects; use <0 for all negatives.",
    )
    parser.add_argument("--max-object-symbol-candidates", type=int, default=64, help="Maximum symbol candidates per object for explicit negative object-symbol rows; <=0 keeps all symbols.")
    parser.add_argument("--max-object-pair-candidates", type=int, default=128, help="Maximum object-object candidate rows per instance; <=0 keeps all object pairs.")
    parser.add_argument(
        "--oracle-kb-predicates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Represent KB/symbol-pair entries as deterministic DomiKnowS predicate logits instead of learned Qwen predicates.",
    )
    parser.add_argument(
        "--execution-label-only",
        action="store_true",
        help="Do not attach local visual predicate labels; train learned predicates only through compiled executable logic_label/closs.",
    )
    parser.add_argument(
        "--allow-oracle-inconsistent-executables",
        action="store_true",
        help="Keep examples even when perfect bounded predicates do not select exactly the gold object/set.",
    )
    return parser.parse_args()



def _load_label_schema(path):
    if path is None:
        return None
    data = json.loads(Path(path).read_text())
    return {
        "object_symbol": list(data.get("object_symbol", [])),
        "symbol_pair": list(data.get("symbol_pair", [])),
        "object_pair": list(data.get("object_pair", [])),
    }


def _augment_instances_for_schema(instances, schema):
    if not schema:
        return instances
    augmented = list(instances)
    dummy = {
        "objects": ["__schema_o1__", "__schema_o2__"],
        "symbols": ["__schema_s1__", "__schema_s2__"],
        "visual_facts": [],
        "kb_facts": [],
        "facts": [],
        "query": {"target_type": "__any_object__", "conditions": [], "answer_type": "object"},
    }
    for rel in schema.get("symbol_pair", []):
        dummy["kb_facts"].append((rel, "__schema_s1__", "__schema_s2__"))
        dummy["facts"].append((rel, "__schema_s1__", "__schema_s2__"))
    for rel in schema.get("object_pair", []):
        dummy["visual_facts"].append((rel, "__schema_o1__", "__schema_o2__"))
        dummy["facts"].append((rel, "__schema_o1__", "__schema_o2__"))
    augmented.append(dummy)
    return augmented

def _save_epoch_checkpoint(program, spaces, output, stage, epoch):
    output = Path(output)
    epoch_path = output.with_name(f"{output.stem}_{stage}_epoch{epoch}{output.suffix}")
    program.save(epoch_path)
    schema_path = epoch_path.with_suffix(epoch_path.suffix + ".schema.json")
    schema_path.write_text(json.dumps(spaces, indent=2, sort_keys=True))
    print(f"saved_epoch_checkpoint={epoch_path}", flush=True)
    print(f"saved_epoch_schema={schema_path}", flush=True)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    task_path, instances, failures = load_instances(args)
    train, dev = split_instances(instances, args.dev_fraction, args.seed)
    print(f"task_path={task_path}", flush=True)
    print(f"loaded={len(instances)} train={len(train)} dev={len(dev)} failures={len(failures)} device={args.device}", flush=True)
    if failures[:5]:
        print(f"first_failures={failures[:5]}", flush=True)
    warmup_epochs = args.epochs if args.warmup_epochs is None else args.warmup_epochs
    print(f"warmup_epochs={warmup_epochs} constraint_epochs={args.constraint_epochs} lr={args.lr}", flush=True)

    # Build the graph namespace from both splits. The DomiKnowS graph must
    # declare every object/symbol/relation that can appear in executable logic;
    # otherwise dev examples with unseen object IDs fail at compile time.
    graph_instances = list(train) + list(dev)
    if args.schema_limit is not None and args.schema_limit != args.limit:
        schema_args = argparse.Namespace(**vars(args))
        schema_args.limit = args.schema_limit
        _schema_task_path, schema_instances, schema_failures = load_instances(schema_args)
        graph_instances = schema_instances
        print(
            f"schema_loaded={len(schema_instances)} schema_failures={len(schema_failures)} "
            f"schema_limit={args.schema_limit}",
            flush=True,
        )
    ctx, spaces = build_graphqa_context(graph_instances, args)
    # CLEVR-style executable training requires compile_executable before the
    # InferenceProgram is constructed. InferenceModel snapshots graph logical
    # constraints at construction time, so compiling afterward leaves the
    # datanode with no active executable constraint labels.
    train_data = compile_program_train_dataset(train, ctx, spaces, device=args.device)
    dev_data = None
    if dev:
        try:
            dev_data = compile_program_train_dataset(dev, ctx, spaces, device=args.device)
        except ValueError as exc:
            print(f"dev_compile_skipped={exc}", flush=True)
    program = create_graphqa_program(ctx, args)
    if args.checkpoint:
        program.load(args.checkpoint, map_location=args.device)
        print(f"loaded_checkpoint={args.checkpoint}", flush=True)
    if args.eval_only:
        if not args.checkpoint:
            program.load(args.output, map_location=args.device)
            print(f"loaded_checkpoint={args.output}", flush=True)
    else:
        Optim = functools.partial(torch.optim.AdamW, lr=args.lr)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.save_every_epoch:
            total_warmup = int(warmup_epochs or 0)
            total_constraint = int(args.constraint_epochs or 0)
            for epoch in range(total_warmup):
                print(f"epoch_stage=warmup epoch={epoch + 1}/{total_warmup}", flush=True)
                program.train(
                    train_data,
                    valid_set=dev_data,
                    Optim=Optim,
                    warmup_epochs=1,
                    constraint_epochs=0,
                    device=args.device,
                    c_lr=args.lr,
                )
                _save_epoch_checkpoint(program, spaces, args.output, "warmup", epoch + 1)
            for epoch in range(total_constraint):
                print(f"epoch_stage=constraint epoch={epoch + 1}/{total_constraint}", flush=True)
                program.train(
                    train_data,
                    valid_set=dev_data,
                    Optim=Optim,
                    warmup_epochs=0,
                    constraint_epochs=1,
                    device=args.device,
                    c_lr=args.lr,
                )
                _save_epoch_checkpoint(program, spaces, args.output, "constraint", epoch + 1)
        else:
            program.train(
                train_data,
                valid_set=dev_data,
                Optim=Optim,
                warmup_epochs=warmup_epochs,
                constraint_epochs=args.constraint_epochs,
                device=args.device,
                c_lr=args.lr,
            )
        program.save(args.output)
        schema_path = args.output.with_suffix(args.output.suffix + ".schema.json")
        schema_path.write_text(json.dumps(spaces, indent=2, sort_keys=True))
        print(f"saved_schema={schema_path}", flush=True)
        print(f"saved={args.output}", flush=True)
    if dev_data:
        print(f"dev_family={evaluate_family_accuracy(dev_data, ctx, program)}", flush=True)
        if not args.skip_condition_eval:
            print(f"dev_condition={program.evaluate_condition(dev_data, device=args.device, return_dict=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
