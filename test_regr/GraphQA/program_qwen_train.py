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
import random
from pathlib import Path

import torch

from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.metric import MacroAverageTracker
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import EdgeSensor, ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.sensors import FunctionalReaderSensor, FunctionalSensor

from .dataset import DEFAULT_VQAR_ROOT, discover_vqar_dataset, load_kb_facts, load_vqar_tasks, vqar_task_to_graphqa_instance
from .execution import create_candidate_membership_instance, create_executable_instance, materialize_bounded_facts
from .oracle import answer_object, answer_objects
from .graph import OBJECT_SYMBOL_RELATIONS, alias_values, canonical_relation, collect_kb_relations, collect_object_relations, create_graphqa_graph, safe_name
from .modules import GraphQAPredicateClassifier, label_spaces
from .train_predicate_classifier import choose_default_task_path, filter_kb_facts_for_instance

DEFAULT_MODEL = "/localscratch/premsrit/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DEFAULT_OUTPUT = Path("/egr/research-hlr2/premsrit/GraphQA/models/qwen3_8b_graphqa_domiknows_program.pt")


class GraphQASolverModel(SolverModel):
    """SolverModel with supervised CE warmup for DomiKnowS program.train."""

    def __init__(self, *args, **kwargs):
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
        return self.shared.forward_examples([{"kind": self.kind, "prompt": prompt} for prompt in prompts])


def build_graphqa_program(instances, args):
    ctx = create_graphqa_graph(instances)
    spaces = label_spaces(instances)
    spaces["answer_object"] = [str(value) for value in ctx.object_values]
    spaces["_require_oracle_clean"] = not args.allow_oracle_inconsistent_executables
    spaces["_enable_set_answer_execution"] = args.enable_set_answer_execution
    spaces["_max_set_answer_candidates"] = args.max_set_answer_candidates
    spaces["_max_set_answer_negatives"] = args.max_set_answer_negatives
    attach_program_train_sensors(ctx, spaces, args)
    poi = [
        ctx.scene,
        ctx.obj,
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
    )
    return None, ctx, program, spaces


def attach_program_train_sensors(ctx, spaces, args):
    device = args.device
    ctx.scene["index"] = FunctionalReaderSensor(keyword="scene_indices", forward=lambda data: _tensor(data, device=device))
    ctx.obj["index"] = FunctionalReaderSensor(keyword="object_indices", forward=lambda data: _tensor(data, device=device))
    ctx.symbol["index"] = FunctionalReaderSensor(keyword="symbol_indices", forward=lambda data: _tensor(data, device=device))
    ctx.obj["ids"] = FunctionalReaderSensor(keyword="object_ids", forward=_safe_list)
    ctx.symbol["ids"] = FunctionalReaderSensor(keyword="symbol_ids", forward=_safe_list)
    ctx.scene["feature"] = FunctionalSensor(ctx.scene["index"], forward=lambda idx: idx.float().unsqueeze(-1))
    ctx.obj["feature"] = FunctionalSensor(ctx.obj["index"], forward=lambda idx: idx.float().unsqueeze(-1))
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

    ctx.object_symbol_pair["object_link"] = FunctionalReaderSensor(
        keyword="object_symbol_object_link",
        forward=lambda data, _device=device: torch.as_tensor(data, dtype=torch.long, device=_device),
    )
    ctx.object_symbol_pair["symbol_link"] = FunctionalReaderSensor(
        keyword="object_symbol_symbol_link",
        forward=lambda data, _device=device: torch.as_tensor(data, dtype=torch.long, device=_device),
    )
    ctx.symbol_pair["src_link"] = FunctionalReaderSensor(
        keyword="symbol_pair_src_link",
        forward=lambda data, _device=device: torch.as_tensor(data, dtype=torch.long, device=_device),
    )
    ctx.symbol_pair["dst_link"] = FunctionalReaderSensor(
        keyword="symbol_pair_dst_link",
        forward=lambda data, _device=device: torch.as_tensor(data, dtype=torch.long, device=_device),
    )
    ctx.object_pair["src_link"] = FunctionalReaderSensor(
        keyword="object_pair_src_link",
        forward=lambda data, _device=device: torch.as_tensor(data, dtype=torch.long, device=_device),
    )
    ctx.object_pair["dst_link"] = FunctionalReaderSensor(
        keyword="object_pair_dst_link",
        forward=lambda data, _device=device: torch.as_tensor(data, dtype=torch.long, device=_device),
    )

    ctx.object_symbol_pair[ctx.object_symbol_object] = EdgeSensor(
        ctx.object_symbol_pair["index"], ctx.obj["index"], ctx.object_symbol_pair["object_link"],
        relation=ctx.object_symbol_object, forward=_edge_from_matrix,
    )
    ctx.object_symbol_pair[ctx.object_symbol_symbol] = EdgeSensor(
        ctx.object_symbol_pair["index"], ctx.symbol["index"], ctx.object_symbol_pair["symbol_link"],
        relation=ctx.object_symbol_symbol, forward=_edge_from_matrix,
    )
    ctx.symbol_pair[ctx.symbol_pair_src] = EdgeSensor(
        ctx.symbol_pair["index"], ctx.symbol["index"], ctx.symbol_pair["src_link"],
        relation=ctx.symbol_pair_src, forward=_edge_from_matrix,
    )
    ctx.symbol_pair[ctx.symbol_pair_dst] = EdgeSensor(
        ctx.symbol_pair["index"], ctx.symbol["index"], ctx.symbol_pair["dst_link"],
        relation=ctx.symbol_pair_dst, forward=_edge_from_matrix,
    )
    ctx.object_pair[ctx.object_pair_src] = EdgeSensor(
        ctx.object_pair["index"], ctx.obj["index"], ctx.object_pair["src_link"],
        relation=ctx.object_pair_src, forward=_edge_from_matrix,
    )
    ctx.object_pair[ctx.object_pair_dst] = EdgeSensor(
        ctx.object_pair["index"], ctx.obj["index"], ctx.object_pair["dst_link"],
        relation=ctx.object_pair_dst, forward=_edge_from_matrix,
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

    _attach_family(ctx.object_symbol_pair, ctx.object_symbol_relation, ctx.object_symbol_relations, spaces["object_symbol"], "object_symbol", shared, device)
    _attach_family(ctx.symbol_pair, ctx.symbol_pair_relation, ctx.symbol_relations, spaces["symbol_pair"], "symbol_pair", shared, device)
    _attach_family(ctx.object_pair, ctx.object_pair_relation, ctx.object_relations, spaces["object_pair"], "object_pair", shared, device)


def _attach_family(base_concept, family_concept, child_concepts, labels, kind, shared, device):
    if not labels:
        return
    base_concept[f"{kind}_prompts"] = FunctionalReaderSensor(keyword=f"{kind}_prompts", forward=lambda data: list(data))
    base_concept[f"{kind}_label"] = FunctionalReaderSensor(
        keyword=f"{kind}_label",
        forward=lambda data, _device=device: torch.as_tensor(data, dtype=torch.long, device=_device),
    )
    # Supervise the actual multiclass family parent concept used by direct
    # evaluation, while keeping the string-key labels available to child labels.
    base_concept[family_concept] = FunctionalReaderSensor(
        keyword=f"{kind}_label",
        label=True,
        forward=lambda data, _device=device: torch.as_tensor(data, dtype=torch.long, device=_device),
    )
    base_concept[family_concept] = ModuleLearner(
        f"{kind}_prompts",
        module=GraphQAFamilyLearner(shared, kind),
        device=device,
    )
    label_to_index = {label: index for index, label in enumerate(labels)}
    for label, concept in child_concepts.items():
        if label not in label_to_index:
            continue
        index = label_to_index[label]
        base_concept[concept] = FunctionalSensor(
            base_concept[family_concept],
            forward=lambda logits, idx=index: _binary_logits_from_family(logits, idx),
        )
        # CLEVR-style local supervision is attached to the concrete child
        # predicate, not only to the open-vocabulary family parent.
        base_concept[concept] = FunctionalSensor(
            base_concept[f"{kind}_label"],
            forward=lambda labels, idx=index: (labels.view(-1) == idx).long(),
            label=True,
        )


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
    for instance in instances:
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
        answer = instance.get("expected_answer") or expected_answers[0]
        if require_clean and oracle_answers != [str(answer)]:
            return []
        instance_for_execution = dict(instance)
        instance_for_execution["expected_answer"] = str(answer)
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
    symbols = [str(symbol) for symbol in instance.get("symbols", [])]
    object_index = {obj: i for i, obj in enumerate(objects)}
    symbol_index = {symbol: i for i, symbol in enumerate(symbols)}

    object_symbol_rows = []
    symbol_pair_rows = []
    object_pair_rows = []
    seen = {"object_symbol": set(), "symbol_pair": set(), "object_pair": set()}

    for pred, left, right in facts:
        pred = canonical_relation(pred)
        if pred in OBJECT_SYMBOL_RELATIONS and left in object_index and right in symbol_index and pred in spaces["object_symbol"]:
            key = (object_index[left], symbol_index[right], pred)
            if key not in seen["object_symbol"]:
                seen["object_symbol"].add(key)
                object_symbol_rows.append((object_index[left], symbol_index[right], pred, _object_symbol_prompt(instance, left, right)))
        elif pred in spaces["symbol_pair"] and left in symbol_index and right in symbol_index:
            key = (symbol_index[left], symbol_index[right], pred)
            if key not in seen["symbol_pair"]:
                seen["symbol_pair"].add(key)
                symbol_pair_rows.append((symbol_index[left], symbol_index[right], pred, _symbol_pair_prompt(instance, left, right)))
        elif pred in spaces["object_pair"] and left in object_index and right in object_index:
            key = (object_index[left], object_index[right], pred)
            if key not in seen["object_pair"]:
                seen["object_pair"].add(key)
                object_pair_rows.append((object_index[left], object_index[right], pred, _object_pair_prompt(instance, left, right)))

    if not object_symbol_rows and objects and symbols and spaces["object_symbol"]:
        object_symbol_rows.append((0, 0, spaces["object_symbol"][0], _dummy_prompt("object-symbol")))
    if not symbol_pair_rows and symbols and spaces["symbol_pair"]:
        symbol_pair_rows.append((0, 0, spaces["symbol_pair"][0], _dummy_prompt("symbol-pair")))
    if not object_pair_rows and objects and spaces["object_pair"]:
        object_pair_rows.append((0, 0, spaces["object_pair"][0], _dummy_prompt("object-pair")))

    converted.update({
        "scene_indices": torch.tensor([0], dtype=torch.long, device=device),
        "object_indices": torch.arange(len(objects), dtype=torch.long, device=device),
        "symbol_indices": torch.arange(len(symbols), dtype=torch.long, device=device),
        "object_ids": objects,
        "symbol_ids": symbols,
        "object_symbol_pair_indices": torch.arange(len(object_symbol_rows), dtype=torch.long, device=device),
        "symbol_pair_indices": torch.arange(len(symbol_pair_rows), dtype=torch.long, device=device),
        "object_pair_indices": torch.arange(len(object_pair_rows), dtype=torch.long, device=device),
        "object_symbol_object_link": _link_matrix(len(object_symbol_rows), len(objects), [(i, row[0]) for i, row in enumerate(object_symbol_rows)], device),
        "object_symbol_symbol_link": _link_matrix(len(object_symbol_rows), len(symbols), [(i, row[1]) for i, row in enumerate(object_symbol_rows)], device),
        "symbol_pair_src_link": _link_matrix(len(symbol_pair_rows), len(symbols), [(i, row[0]) for i, row in enumerate(symbol_pair_rows)], device),
        "symbol_pair_dst_link": _link_matrix(len(symbol_pair_rows), len(symbols), [(i, row[1]) for i, row in enumerate(symbol_pair_rows)], device),
        "object_pair_src_link": _link_matrix(len(object_pair_rows), len(objects), [(i, row[0]) for i, row in enumerate(object_pair_rows)], device),
        "object_pair_dst_link": _link_matrix(len(object_pair_rows), len(objects), [(i, row[1]) for i, row in enumerate(object_pair_rows)], device),
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
            instance["symbols"] = sorted(set(instance.get("symbols", [])) | _symbols_needed_by_query(instance))
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
    for pred, _obj, symbol in instance.get("visual_facts", []):
        if canonical_relation(pred) in {"Name", "Attribute"} and symbol is not None:
            needed.add(symbol)
    query = instance.get("query", {})
    if query.get("target_type") not in (None, "__any_object__"):
        needed.add(query["target_type"])
    for conditions in [query.get("conditions", [])] + list(query.get("alternatives", [])):
        for pred, _left, right in conditions:
            pred = canonical_relation(pred)
            if pred in {"Name", "Attribute", "ObjectType", "ObjectCategory", "SemanticClass"}:
                if right is not None:
                    needed.add(right)
            elif pred == "KG":
                _rel, dst = right
                needed.add(dst)

    filtered = []
    frontier = set(needed)
    seen = set()
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

    extra_kg = 0
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

def _symbols_needed_by_query(instance):
    symbols = set()
    query = instance.get("query", {})
    target = query.get("target_type")
    if target and target != "__any_object__":
        symbols.add(target)
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
                    symbols.add(dst)
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
    reserved = set(ctx.object_symbol_relations) | set(ctx.symbol_relations) | set(ctx.object_relations) | {
        "scene", "obj", "symbol", "object_symbol_pair", "symbol_pair", "object_pair",
        "object_symbol_relation", "symbol_pair_relation", "object_pair_relation",
        "scene_contains_obj", "scene_contains_symbol", "object_symbol_object", "object_symbol_symbol",
        "symbol_pair_src", "symbol_pair_dst", "object_pair_src", "object_pair_dst",
    }
    for name, concept in ctx.concepts.items():
        if name in reserved:
            continue
        # The graph stores safe object and symbol names in the same namespace.
        # The dataset values are already safe-normalized for symbols; objects are ids.
        if prefix == "object_" and concept.sup is ctx.obj:
            yield name, concept
        elif prefix == "symbol_" and concept.sup is ctx.symbol:
            yield name, concept


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


def _object_symbol_prompt(instance, obj, symbol):
    return "\n".join([
        "Classify the GraphQA object-symbol predicate.",
        f"Objects: {', '.join(map(str, instance.get('objects', [])))}",
        f"Object: {obj}",
        f"Symbol: {symbol}",
        f"Query: {instance.get('query', {})}",
        "Labels: Name, ObjectType, ObjectCategory, Attribute.",
    ])


def _symbol_pair_prompt(instance, src, dst):
    return "\n".join([
        "Classify the GraphQA symbol-pair KG predicate.",
        f"Source symbol: {src}",
        f"Destination symbol: {dst}",
        f"Query: {instance.get('query', {})}",
    ])


def _object_pair_prompt(instance, src, dst):
    return "\n".join([
        "Classify the GraphQA object-object relation predicate.",
        f"Source object: {src}",
        f"Destination object: {dst}",
        f"Query: {instance.get('query', {})}",
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-condition-eval", action="store_true")
    parser.add_argument("--enable-set-answer-execution", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument(
        "--allow-oracle-inconsistent-executables",
        action="store_true",
        help="Keep examples even when perfect bounded predicates do not select exactly the gold object/set.",
    )
    return parser.parse_args()


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
    train_data, ctx, program, spaces = build_graphqa_program(graph_instances, args)
    train_data = compile_program_train_dataset(train, ctx, spaces, device=args.device)
    dev_data = compile_program_train_dataset(dev, ctx, spaces, device=args.device) if dev else None
    if args.checkpoint:
        program.load(args.checkpoint, map_location=args.device)
        print(f"loaded_checkpoint={args.checkpoint}", flush=True)
    if args.eval_only:
        if not args.checkpoint:
            program.load(args.output, map_location=args.device)
            print(f"loaded_checkpoint={args.output}", flush=True)
    else:
        Optim = functools.partial(torch.optim.AdamW, lr=args.lr)
        program.train(
            train_data,
            valid_set=dev_data,
            Optim=Optim,
            warmup_epochs=warmup_epochs,
            constraint_epochs=args.constraint_epochs,
            device=args.device,
            c_lr=args.lr,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        program.save(args.output)
        print(f"saved={args.output}", flush=True)
    if dev_data:
        print(f"dev_family={evaluate_family_accuracy(dev_data, ctx, program)}", flush=True)
        if not args.skip_condition_eval:
            print(f"dev_condition={program.evaluate_condition(dev_data, device=args.device, return_dict=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
