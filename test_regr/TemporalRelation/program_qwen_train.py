"""Train TemporalRelation through DomiKnowS InferenceProgram.

This is the CLEVR-style path: dataset -> graph.compile_executable ->
InferenceProgram(..., SolverModel) -> program.train(...).
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

import torch
from torch.nn import functional as F

from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.metric import MacroAverageTracker
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import EdgeSensor, ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateReaderSensor
from domiknows.sensor.pytorch.sensors import FunctionalReaderSensor, FunctionalSensor

from .config import TEMPORAL_CONFIG
from .dataset import DEFAULT_TEMPORAL_DATA_ROOT, load_temporal_instances
from .execution import create_executable_instance, mark_text_for_pair
from .graph import TEMPORAL_LABELS, create_temporal_graph, unpack_pair
from .program import BinaryOracleLearner, _tensor

DEFAULT_MODEL = TEMPORAL_CONFIG.training_model
DEFAULT_OUTPUT = TEMPORAL_CONFIG.output_path("qwen3_8b_temporal_domiknows_program.pt")
LOCAL_IGNORE_LABEL = -100
_TEMPORAL_CLASS_WEIGHTS = None


class WeightedTemporalCrossEntropyLoss(torch.nn.Module):
    """Cross-entropy with optional temporal class weights and ignored rows."""

    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            self.register_buffer("weights", None, persistent=False)
        else:
            self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float32), persistent=False)

    def forward(self, input, target, *args, **kwargs):
        input = input.view(-1, input.shape[-1])
        target = target.view(-1).to(dtype=torch.long, device=input.device)
        weight = self.weights.to(input.device) if self.weights is not None else None
        return F.cross_entropy(input, target, weight=weight, ignore_index=LOCAL_IGNORE_LABEL)


def _make_temporal_ce_loss(weights=None):
    if weights is None:
        return NBCrossEntropyLoss()
    return WeightedTemporalCrossEntropyLoss(weights)


def _parse_temporal_class_weights(args):
    if args.label_weights:
        values = [float(value.strip()) for value in args.label_weights.split(",") if value.strip()]
        if len(values) != len(TEMPORAL_LABELS):
            raise ValueError(
                f"--label-weights must provide {len(TEMPORAL_LABELS)} comma-separated values "
                f"in {TEMPORAL_LABELS} order"
            )
    else:
        values = [1.0] * len(TEMPORAL_LABELS)
    values[TEMPORAL_LABELS.index("Vague")] *= float(args.vague_weight)
    values[TEMPORAL_LABELS.index("Equal")] *= float(args.equal_weight)
    if all(abs(value - 1.0) < 1e-12 for value in values):
        return None
    return values


class TemporalSolverModel(SolverModel):
    """SolverModel with supervised CE loss for DomiKnowS program.train warmup."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("loss", MacroAverageTracker(_make_temporal_ce_loss(_TEMPORAL_CLASS_WEIGHTS)))
        super().__init__(*args, **kwargs)


class QwenTemporalRelationLearner(torch.nn.Module):
    """Qwen causal-LM verbalizer for EventPair temporal-relation classes.

    The predicate value should mean "the LLM would answer Before/After/Equal/Vague"
    for the marked event pair. We therefore score the full answer string for each
    class instead of mean-pooling hidden states and training an unrelated head.
    """

    def __init__(
        self,
        model_path=DEFAULT_MODEL,
        device="cuda",
        freeze_backbone=True,
        lora_r=0,
        lora_alpha=8,
        lora_dropout=0.05,
        lora_target_modules="q_proj,v_proj",
        max_length=128,
        encode_batch_size=1,
    ):
        super().__init__()
        self.device_name = device
        self.max_length = int(max_length)
        self.encode_batch_size = max(1, int(encode_batch_size or 1))

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if str(device).startswith("cuda"):
            model_kwargs["dtype"] = torch.float16
        self.backbone = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        if int(lora_r) > 0:
            from peft import LoraConfig, TaskType, get_peft_model

            targets = [m.strip() for m in str(lora_target_modules).split(",") if m.strip()]
            config = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=targets,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.backbone = get_peft_model(self.backbone, config)
            if hasattr(self.backbone, "gradient_checkpointing_enable"):
                self.backbone.gradient_checkpointing_enable()
            if hasattr(self.backbone, "enable_input_require_grads"):
                self.backbone.enable_input_require_grads()
            if hasattr(self.backbone, "config"):
                self.backbone.config.use_cache = False
        elif freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone.to(device)
        self.label_token_ids = self._build_label_token_ids()

    def backbone_has_trainable_parameters(self):
        return any(param.requires_grad for param in self.backbone.parameters())

    def _build_label_token_ids(self):
        token_ids = []
        for label in TEMPORAL_LABELS:
            encoded = self.tokenizer.encode(f" {label}", add_special_tokens=False)
            if not encoded:
                encoded = self.tokenizer.encode(label, add_special_tokens=False)
            if not encoded:
                raise ValueError(f"Could not tokenize temporal label {label!r}")
            token_ids.append(encoded)
        return token_ids

    @staticmethod
    def _format_prompt(marked_text):
        return (
            "Classify the temporal relation from event E1 to event E2. "
            "Choose exactly one label from: Before, After, Equal, Vague.\n"
            f"Text: {marked_text}\n"
            "Answer:"
        )

    def forward(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = list(prompts)
        if not prompts:
            return torch.empty((0, len(TEMPORAL_LABELS)), dtype=torch.float32, device=self.device_name)

        chunks = []
        grad_enabled = self.training and self.backbone_has_trainable_parameters()
        for start in range(0, len(prompts), self.encode_batch_size):
            batch = prompts[start : start + self.encode_batch_size]
            chunks.append(self._score_label_sequences(batch, grad_enabled))
        return torch.cat(chunks, dim=0)

    def _score_label_sequences(self, prompts, grad_enabled):
        rows = []
        masks = []
        label_masks = []
        pad_id = self.tokenizer.pad_token_id
        for prompt in prompts:
            prompt_ids = self.tokenizer.encode(self._format_prompt(prompt), add_special_tokens=False)
            for label_ids in self.label_token_ids:
                keep_prompt = max(1, self.max_length - len(label_ids))
                ids = prompt_ids[-keep_prompt:] + label_ids
                rows.append(ids)
                masks.append([1] * len(ids))
                label_masks.append([0] * (len(ids) - len(label_ids)) + [1] * len(label_ids))

        max_len = max(len(row) for row in rows)
        input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long, device=self.device_name)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=self.device_name)
        label_mask = torch.zeros((len(rows), max_len), dtype=torch.bool, device=self.device_name)
        for row_idx, (ids, mask, lm) in enumerate(zip(rows, masks, label_masks)):
            n = len(ids)
            input_ids[row_idx, :n] = torch.tensor(ids, dtype=torch.long, device=self.device_name)
            attention_mask[row_idx, :n] = torch.tensor(mask, dtype=torch.long, device=self.device_name)
            label_mask[row_idx, :n] = torch.tensor(lm, dtype=torch.bool, device=self.device_name)

        with torch.set_grad_enabled(grad_enabled):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            log_probs = outputs.logits[:, :-1, :].log_softmax(dim=-1)
            target_ids = input_ids[:, 1:]
            target_label_mask = label_mask[:, 1:]
            token_scores = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            label_scores = (token_scores * target_label_mask.float()).sum(dim=-1)
            label_lengths = target_label_mask.float().sum(dim=-1).clamp_min(1.0)
            label_scores = label_scores / label_lengths
        return label_scores.view(len(prompts), len(TEMPORAL_LABELS)).float()

def build_temporal_program(instances, args):
    ctx = create_temporal_graph(
        instances,
        include_global_constraints=not args.no_global_consistency,
        include_exactly_one=getattr(args, "exactly_one_label", True),
        include_transitivity=getattr(args, "transitivity", True),
    )
    attach_program_train_sensors(ctx, args)
    dataset = compile_program_train_dataset(
        instances,
        ctx,
        device=args.device,
        max_events_per_instance=args.max_events_per_instance,
        pair_selection=args.pair_selection,
        max_pairs_per_instance=args.max_pairs_per_instance,
    )

    poi = [
        ctx.document,
        ctx.sentence,
        ctx.token,
        ctx.event,
        ctx.query_event1,
        ctx.query_event2,
        ctx.event_pair,
        ctx.temporal_relation,
        *ctx.label_concepts.values(),
        ctx.graph.constraint,
    ]
    program = InferenceProgram(
        ctx.graph,
        TemporalSolverModel,
        poi=poi,
        device=args.device,
        inferTypes=[item.strip() for item in args.infer_types.split(",") if item.strip()],
        beta=args.beta,
        training_style=args.training_style,
        use_gumbel=args.use_gumbel,
        initial_temp=args.gumbel_temp_start,
        final_temp=args.gumbel_temp_end,
        anneal_start_epoch=args.gumbel_anneal_start_epoch,
        anneal_epochs=args.gumbel_anneal_epochs,
        hard_gumbel=args.hard_gumbel,
    )
    # This local DomiKnowS branch forwards InferenceProgram kwargs into the
    # main SolverModel. Set the constraint-model tnorm after construction.
    if hasattr(program, "cmodel"):
        program.cmodel.tnorm = args.tnorm
        program.cmodel.counting_tnorm = getattr(program.cmodel, "counting_tnorm", None) or args.tnorm
        if hasattr(program.cmodel, "pos_weight"):
            program.cmodel.pos_weight = float(args.executable_pos_weight)
    return dataset, ctx, program


def attach_program_train_sensors(ctx, args):
    device = args.device
    ctx.document["index"] = FunctionalReaderSensor(keyword="document_indices", forward=lambda data: _tensor(data, device=device))
    ctx.sentence["index"] = FunctionalReaderSensor(keyword="sentence_indices", forward=lambda data: _tensor(data, device=device))
    ctx.token["index"] = FunctionalReaderSensor(keyword="event_indices", forward=lambda data: _tensor(data, device=device))

    ctx.sentence[ctx.document_contains_sentence] = EdgeSensor(
        ctx.sentence["index"], ctx.document["index"], relation=ctx.document_contains_sentence,
        forward=lambda sentence, _document: torch.ones_like(sentence).unsqueeze(-1),
    )
    ctx.token[ctx.sentence_contains_token] = EdgeSensor(
        ctx.token["index"], ctx.sentence["index"], relation=ctx.sentence_contains_token,
        forward=lambda token, _sentence: torch.ones_like(token).unsqueeze(-1),
    )

    # Query/event detection stays oracle for this baseline; temporal relation is learned by Qwen.
    for name, concept, keyword in [
        ("event", ctx.event, "is_event"),
        ("query_event1", ctx.query_event1, "is_query_event1"),
        ("query_event2", ctx.query_event2, "is_query_event2"),
    ]:
        ctx.token[f"{name}_label"] = FunctionalReaderSensor(keyword=keyword, forward=lambda data, _device=device: _tensor(data, device=_device))
        ctx.token[concept] = ModuleLearner(f"{name}_label", module=BinaryOracleLearner(), device=device)

    ctx.event_pair[ctx.pair_event1.reversed, ctx.pair_event2.reversed] = CompositionCandidateReaderSensor(
        ctx.token["index"],
        relations=(ctx.pair_event1.reversed, ctx.pair_event2.reversed),
        keyword="event_pair_candidates",
        forward=_candidate_event_pair_from_allowed,
    )
    ctx.event_pair["pair_prompts"] = FunctionalReaderSensor(keyword="pair_prompts", forward=lambda data: data)
    ctx.event_pair["temporal_relation_label"] = FunctionalReaderSensor(
        keyword="temporal_relation_label",
        forward=lambda data, _device=device: torch.as_tensor(data, dtype=torch.long, device=_device),
    )
    if args.supervise_local_predicates:
        # Optional diagnostic/warmup mode. This appends a label sensor to the
        # same concept property; the ModuleLearner below remains the learnable
        # predicate, matching the DomiKnowS/CLEVR pattern.
        ctx.event_pair[ctx.temporal_relation] = FunctionalReaderSensor(
            keyword="temporal_relation_label",
            label=True,
            forward=lambda data, _device=device: torch.as_tensor(data, dtype=torch.long, device=_device),
        )
    ctx.event_pair[ctx.temporal_relation] = ModuleLearner(
        "pair_prompts",
        module=QwenTemporalRelationLearner(
            model_path=args.model_path,
            device=device,
            freeze_backbone=args.freeze_backbone,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
            max_length=args.max_length,
            encode_batch_size=args.encode_batch_size,
        ),
        loss=_make_temporal_ce_loss(_TEMPORAL_CLASS_WEIGHTS),
        device=device,
    )
    # queryL reads the child concepts (Before/After/Equal/Vague). Expose each
    # class logit as a binary child predicate derived from the shared parent
    # multiclass learner, so Qwen is loaded only once.
    for label_index, label in enumerate(TEMPORAL_LABELS):
        concept = ctx.label_concepts[label]
        ctx.event_pair[concept] = FunctionalSensor(
            ctx.event_pair[ctx.temporal_relation],
            forward=lambda logits, idx=label_index: _binary_logits_from_multiclass(logits, idx),
        )


def _binary_logits_from_multiclass(logits, index):
    positive = logits[:, index]
    if logits.shape[1] == 1:
        negative = -positive
    else:
        mask = torch.ones(logits.shape[1], dtype=torch.bool, device=logits.device)
        mask[index] = False
        negative = torch.logsumexp(logits[:, mask], dim=-1)
    return torch.stack([negative, positive], dim=-1)


def _candidate_event_pair_from_allowed(index, data, arg1=None, arg2=None, **_kwargs):
    if arg1 is None or arg2 is None:
        return False
    left = int(arg1.getAttribute("index").detach().cpu().view(-1)[0].item())
    right = int(arg2.getAttribute("index").detach().cpu().view(-1)[0].item())
    return (left, right) in set(tuple(pair) for pair in data)


_BOOLEAN_EXECUTABLE_ASSERTION = False

def set_boolean_executable_assertion(enabled):
    global _BOOLEAN_EXECUTABLE_ASSERTION
    _BOOLEAN_EXECUTABLE_ASSERTION = bool(enabled)

def args_boolean_executable_assertion():
    return _BOOLEAN_EXECUTABLE_ASSERTION

def compile_program_train_dataset(
    instances,
    ctx,
    device="cpu",
    max_events_per_instance=None,
    pair_selection="all",
    max_pairs_per_instance=None,
):
    data = [
        _to_program_train_data(
            instance,
            device=device,
            max_events_per_instance=max_events_per_instance,
            pair_selection=pair_selection,
            max_pairs_per_instance=max_pairs_per_instance,
        )
        for instance in instances
    ]
    return ctx.graph.compile_executable(
        data,
        logic_keyword="logic_str",
        logic_label_keyword="logic_label",
        extra_namespace_values=ctx.namespace,
    )




def create_boolean_label_query_logic(instance):
    """Boolean executable constraint used by InferenceModel BCE loss.

    The public adapter still exposes queryL/iotaL in execution.py. For DomiKnowS
    constraint training, InferenceModel currently expects executable labels in
    [0, 1], so we convert the gold multiclass answer into a boolean assertion:
    the selected EventPair has the gold temporal label.
    """
    converted = create_executable_instance(instance)
    query_pair = converted.get("query_pair") or instance.get("query_pair") or instance.get("event_pairs", [None])[0]
    _e1, _e2, label = unpack_pair(query_pair)
    if label is None:
        label = TEMPORAL_LABELS[int(converted["logic_label"])]
    return f"""andL(
        EventPair("p"),
        event("p1", path=("p", pair_event1)),
        event("p2", path=("p", pair_event2)),
        query_event1("p1"),
        query_event2("p2"),
        {label}("p")
    )"""

def _to_program_train_data(instance, device="cpu", max_events_per_instance=None, pair_selection="all", max_pairs_per_instance=None):
    converted = create_executable_instance(instance)
    all_events = list(instance.get("events", []))
    query_pair = converted.get("query_pair") or instance.get("query_pair") or instance.get("event_pairs", [None])[0]
    query_e1, query_e2, _ = unpack_pair(query_pair)
    events = _select_events_for_query(instance, all_events, query_e1, query_e2, max_events_per_instance)
    event_ids = [_event_id(event) for event in events]

    pair_prompts = []
    pair_labels = []
    labels_by_pair = {}
    for pair in instance.get("event_pairs", []):
        e1, e2, label = unpack_pair(pair)
        labels_by_pair[(e1, e2)] = label
    candidate_pairs = _select_candidate_pairs(
        instance,
        event_ids,
        query_e1,
        query_e2,
        pair_selection=pair_selection,
        max_pairs_per_instance=max_pairs_per_instance,
    )
    event_index = {event_id: idx for idx, event_id in enumerate(event_ids)}
    event_pair_candidates = []
    for left, right in candidate_pairs:
        pair_prompts.append(mark_text_for_pair(instance, left, right))
        event_pair_candidates.append((event_index[left], event_index[right]))
        label = labels_by_pair.get((left, right))
        # Warmup/local supervision must only use genuinely annotated pair
        # directions. Inverse and related-but-unlabeled pairs stay in the graph
        # for execution/global constraints but are ignored by CE loss.
        pair_labels.append(TEMPORAL_LABELS.index(label) if label in TEMPORAL_LABELS else LOCAL_IGNORE_LABEL)

    converted.update({
        "document_indices": torch.tensor([0], dtype=torch.long, device=device),
        "sentence_indices": torch.tensor([0], dtype=torch.long, device=device),
        "event_indices": torch.arange(len(events), dtype=torch.long, device=device),
        "is_event": torch.ones(len(events), dtype=torch.long, device=device),
        "is_query_event1": torch.tensor([1 if event_id == query_e1 else 0 for event_id in event_ids], dtype=torch.long, device=device),
        "is_query_event2": torch.tensor([1 if event_id == query_e2 else 0 for event_id in event_ids], dtype=torch.long, device=device),
        "event_pair_candidates": event_pair_candidates,
        "pair_prompts": pair_prompts,
        "temporal_relation_label": torch.tensor(pair_labels, dtype=torch.long, device=device),
    })
    if converted.get("logic_label") is not None:
        if args_boolean_executable_assertion():
            converted["logic_str"] = create_boolean_label_query_logic(instance)
            converted["logic_label"] = torch.FloatTensor([1.0]).to(device)
        else:
            # Match CLEVR query training: keep queryL(...) and supervise the
            # final answer class with a multiclass LongTensor logic_label.
            converted["logic_label"] = torch.LongTensor([int(converted["logic_label"])]).to(device)
    return converted




def evaluate_temporal_relation_accuracy(dataset, ctx, program, device="cpu"):
    """Direct dev accuracy and prediction distribution for the multiclass head."""
    from collections import Counter

    correct = 0
    total = 0
    pred_counts = Counter()
    gold_counts = Counter()
    was_training = program.model.training
    program.model.eval()
    with torch.no_grad():
        for row in dataset:
            program.model(row)
            logits = ctx.event_pair[ctx.temporal_relation](row)
            labels = row.get("temporal_relation_label")
            if labels is None or logits is None or logits.numel() == 0:
                continue
            labels = labels.to(logits.device).view(-1)
            preds = logits.argmax(dim=-1).view(-1)
            n = min(preds.numel(), labels.numel())
            if n == 0:
                continue
            labels = labels[:n]
            preds = preds[:n]
            valid = labels != LOCAL_IGNORE_LABEL
            if not bool(valid.any()):
                continue
            labels = labels[valid]
            preds = preds[valid]
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
            pred_counts.update(TEMPORAL_LABELS[int(i)] for i in preds.detach().cpu().tolist())
            gold_counts.update(TEMPORAL_LABELS[int(i)] for i in labels.detach().cpu().tolist())
    if was_training:
        program.model.train()
    return {
        "temporal_relation_correct": correct,
        "temporal_relation_total": total,
        "temporal_relation_acc": correct / total if total else 0.0,
        "pred_counts": dict(pred_counts),
        "gold_counts": dict(gold_counts),
    }

def split_instances(instances, dev_fraction=0.2, seed=13):
    import random
    instances = list(instances)
    random.Random(seed).shuffle(instances)
    if len(instances) <= 1 or dev_fraction <= 0:
        return instances, []
    dev_size = min(max(1, int(round(len(instances) * dev_fraction))), len(instances) - 1)
    return instances[dev_size:], instances[:dev_size]


def _event_id(event):
    return event.get("id") if isinstance(event, dict) else event



def expand_document_query_instances(documents):
    """Create one executable query per annotated pair while preserving document context."""
    instances = []
    for document in documents:
        pairs = list(document.get("event_pairs", []))
        for pair in pairs:
            e1, e2, _label = unpack_pair(pair)
            instance = dict(document)
            instance["events"] = list(document.get("events", []))
            instance["event_pairs"] = pairs
            instance["query_pair"] = {"e1": e1, "e2": e2, "label": _label}
            instances.append(instance)
    return instances




def _select_candidate_pairs(
    instance,
    event_ids,
    query_e1,
    query_e2,
    pair_selection="all",
    max_pairs_per_instance=None,
):
    event_set = set(event_ids)
    mode = pair_selection or "all"
    if mode not in {"all", "related", "target"}:
        raise ValueError(f"Unsupported pair_selection={pair_selection!r}")

    def capped(pairs):
        if max_pairs_per_instance is None:
            return pairs
        return pairs[: max(1, int(max_pairs_per_instance))]

    def add_unique(pairs, seen, left, right):
        if left == right or left not in event_set or right not in event_set:
            return
        pair = (left, right)
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    if mode == "all":
        pairs = [(left, right) for left in event_ids for right in event_ids if left != right]
        return capped(pairs)

    selected = []
    seen = set()
    add_unique(selected, seen, query_e1, query_e2)
    add_unique(selected, seen, query_e2, query_e1)
    if mode == "target":
        return capped(selected)

    labeled_pairs = []
    for pair in instance.get("event_pairs", []):
        e1, e2, _label = unpack_pair(pair)
        if e1 in event_set and e2 in event_set:
            labeled_pairs.append((e1, e2))

    related_events = {query_e1, query_e2}
    # Priority 1: all annotated pairs directly touching the queried pair.
    for e1, e2 in labeled_pairs:
        if e1 in related_events or e2 in related_events:
            add_unique(selected, seen, e1, e2)
            add_unique(selected, seen, e2, e1)

    # Priority 2: grow through the labeled temporal graph for transitivity/equality checks.
    changed = True
    while changed:
        changed = False
        for e1, e2 in labeled_pairs:
            if e1 in related_events or e2 in related_events:
                before = len(related_events)
                related_events.update([e1, e2])
                add_unique(selected, seen, e1, e2)
                add_unique(selected, seen, e2, e1)
                changed = changed or len(related_events) > before
                if max_pairs_per_instance is not None and len(selected) >= int(max_pairs_per_instance):
                    return capped(selected)

    return capped(selected)

def _select_events_for_query(instance, events, query_e1, query_e2, max_events_per_instance=None):
    if max_events_per_instance is None:
        return list(events)
    max_events = max(2, int(max_events_per_instance))
    event_by_id = {_event_id(event): event for event in events}
    selected = []

    def add(event_id):
        if event_id in event_by_id and event_id not in selected and len(selected) < max_events:
            selected.append(event_id)

    # Always preserve the executable target pair.
    add(query_e1)
    add(query_e2)

    pairs = list(instance.get("event_pairs", []))
    # First keep the inverse and all one-hop neighbors touching the target events.
    for pair in pairs:
        e1, e2, _label = unpack_pair(pair)
        if e1 in {query_e1, query_e2} or e2 in {query_e1, query_e2}:
            add(e1)
            add(e2)
            if len(selected) >= max_events:
                break

    # Then add supervised-pair endpoints so global consistency has more graph structure.
    for pair in pairs:
        e1, e2, _label = unpack_pair(pair)
        add(e1)
        add(e2)
        if len(selected) >= max_events:
            break

    # Finally add any remaining document events deterministically.
    for event in events:
        add(_event_id(event))
        if len(selected) >= max_events:
            break

    return [event_by_id[event_id] for event_id in selected]


def _pair_count_stats(instances, max_events_per_instance=None, pair_selection="all", max_pairs_per_instance=None):
    if not instances:
        return {"instances": 0, "max_events": 0, "max_pairs": 0, "avg_pairs": 0.0}
    counts = []
    for instance in instances:
        events = list(instance.get("events", []))
        query_pair = instance.get("query_pair") or instance.get("event_pairs", [None])[0]
        query_e1, query_e2, _ = unpack_pair(query_pair)
        selected = _select_events_for_query(instance, events, query_e1, query_e2, max_events_per_instance)
        event_ids = [_event_id(event) for event in selected]
        candidate_pairs = _select_candidate_pairs(
            instance,
            event_ids,
            query_e1,
            query_e2,
            pair_selection=pair_selection,
            max_pairs_per_instance=max_pairs_per_instance,
        )
        labeled = set()
        for pair in instance.get("event_pairs", []):
            e1, e2, label = unpack_pair(pair)
            if label in TEMPORAL_LABELS:
                labeled.add((e1, e2))
        n = len(selected)
        counts.append((n, len(candidate_pairs), sum(1 for pair in candidate_pairs if pair in labeled)))
    return {
        "instances": len(instances),
        "max_events": max(n for n, _pairs, _labels in counts),
        "max_pairs": max(pairs for _n, pairs, _labels in counts),
        "avg_pairs": sum(pairs for _n, pairs, _labels in counts) / len(counts),
        "max_labeled_pairs": max(labels for _n, _pairs, labels in counts),
        "avg_labeled_pairs": sum(labels for _n, _pairs, labels in counts) / len(counts),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train TemporalRelation with DomiKnowS program.train and Qwen learner.")
    parser.add_argument("--path", type=Path, default=DEFAULT_TEMPORAL_DATA_ROOT / "MATRES" / "timebank.txt")
    parser.add_argument(
        "--train-paths",
        default=None,
        help=(
            "Comma-separated dataset files to concatenate for training. "
            "When unset, --path is used. Eval-only still scores --path."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--row-level", action="store_true", help="Use the old row-level setup with only the annotated event pair. Off by default; document-level query expansion is the full experiment.")
    parser.add_argument("--max-events-per-instance", type=int, default=None, help="Optional document event budget. Query endpoints are always retained.")
    parser.add_argument("--pair-selection", choices=["all", "related", "target"], default="all", help="Which event pairs become DomiKnowS EventPair nodes: all ordered pairs, only query-related labeled pairs, or only the target pair plus inverse.")
    parser.add_argument("--max-pairs-per-instance", type=int, default=None, help="Upper bound on selected EventPair nodes/prompts per query instance. Target pair is prioritized.")
    parser.add_argument("--dev-fraction", type=float, default=0.2)
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
    parser.add_argument("--epochs", type=int, default=1, help="Backward-compatible alias for --warmup-epochs when that flag is not set.")
    parser.add_argument("--warmup-epochs", type=int, default=None, help="Number of supervised DomiKnowS warmup epochs before constraint training. Use 1 or 2 for Qwen smoke/full runs.")
    parser.add_argument("--constraint-epochs", type=int, default=0, help="Number of executable queryL/iotaL constraint epochs after optional warmup.")
    parser.add_argument("--allow-unstable-constraint-training", action="store_true", help="Deprecated compatibility flag; constraint training is controlled by --constraint-epochs.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--tnorm",
        default="P",
        choices=["P", "G", "L", "SP"],
        help=(
            "T-norm for the constraint loss. Default changed G->P: every consistency "
            "rule in graph.py is an implication, and under Godel an implication gives "
            "its ANTECEDENT exactly zero gradient. Symmetry is two-sided (an "
            "inconsistent pair should both raise the consequent and lower the "
            "antecedent), so Godel discards half of each correction. Pass --tnorm G "
            "to reproduce the previous behaviour."
        ),
    )
    parser.add_argument(
        "--no-exactly-one-label",
        dest="exactly_one_label",
        action="store_false",
        help=(
            "Drop the exactL(...limit=1) label constraint. It is already guaranteed "
            "at decode time by the shared multiclass head, so under a t-norm it acts "
            "as a sharpening penalty rather than logical enforcement."
        ),
    )
    parser.set_defaults(exactly_one_label=True)
    parser.add_argument(
        "--no-transitivity",
        dest="transitivity",
        action="store_false",
        help=(
            "Drop the before-transitivity constraint. It needs three chained pairs, "
            "so with --pair-selection target and --max-pairs-per-instance < 3 it has "
            "zero groundings and can never fire."
        ),
    )
    parser.set_defaults(transitivity=True)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--infer-types",
        default="local/softmax,local/argmax",
        help="Comma-separated DomiKnowS inference outputs, e.g. local/softmax,local/argmax,ILP.",
    )
    parser.add_argument(
        "--training-style",
        choices=["simple", "primal_dual"],
        default="simple",
        help="DomiKnowS InferenceProgram training style. Use primal_dual for stronger constraint/global training.",
    )
    parser.add_argument(
        "--constraint-only",
        action="store_true",
        help="In primal_dual mode, make constraint epochs optimize mostly executable/global loss instead of mloss + beta*closs.",
    )
    parser.add_argument(
        "--constraint-loss-scale",
        type=float,
        default=1.0,
        help="Scale executable/global loss in primal_dual constraint_only mode.",
    )
    parser.add_argument("--c-warmup-iters", type=int, default=10)
    parser.add_argument("--c-freq", type=int, default=10)
    parser.add_argument("--c-freq-increase", type=int, default=5)
    parser.add_argument("--c-freq-increase-freq", type=int, default=1)
    parser.add_argument("--c-lr-decay", type=float, default=4.0)
    parser.add_argument("--c-lr-decay-param", type=float, default=1.0)
    parser.add_argument(
        "--executable-pos-weight",
        type=float,
        default=1.0,
        help="Positive-label weight for executable BCE constraints. Useful for queryL/boolean executable training.",
    )
    parser.add_argument("--use-gumbel", action="store_true")
    parser.add_argument("--hard-gumbel", action="store_true")
    parser.add_argument("--gumbel-temp-start", type=float, default=1.0)
    parser.add_argument("--gumbel-temp-end", type=float, default=0.1)
    parser.add_argument("--gumbel-anneal-start-epoch", type=int, default=0)
    parser.add_argument("--gumbel-anneal-epochs", type=int, default=None)
    parser.add_argument(
        "--label-weights",
        default=None,
        help=f"Optional comma-separated CE weights in {TEMPORAL_LABELS} order.",
    )
    parser.add_argument("--vague-weight", type=float, default=1.0, help="Multiplier for the Vague class CE weight.")
    parser.add_argument("--equal-weight", type=float, default=1.0, help="Multiplier for the Equal class CE weight.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-only", action="store_true", help="Load --checkpoint/--output and only run dev evaluation.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint to load for --eval-only; defaults to --output.")
    parser.add_argument("--skip-condition-eval", action="store_true", help="Skip slow executable queryL/iotaL condition evaluation.")
    parser.add_argument(
        "--supervise-local-predicates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train temporal_relation ModuleLearner from genuinely annotated local pair labels during warmup. Use --no-supervise-local-predicates for execution-only constraint experiments.",
    )
    parser.add_argument(
        "--no-global-consistency",
        action="store_true",
        help="Disable graph-level temporal consistency rules and train only from per-sample executable query constraints.",
    )
    parser.add_argument(
        "--boolean-executable-assertion",
        action="store_true",
        help="Use the older boolean assertion wrapper instead of CLEVR-style queryL multiclass final labels.",
    )
    return parser.parse_args()


def _parse_data_paths(value):
    if not value:
        return []
    paths = [Path(item.strip()) for item in str(value).split(",") if item.strip()]
    if not paths:
        raise ValueError("--train-paths was provided but no dataset paths were parsed")
    return paths


def main():
    global _TEMPORAL_CLASS_WEIGHTS
    args = parse_args()
    _TEMPORAL_CLASS_WEIGHTS = _parse_temporal_class_weights(args)
    torch.manual_seed(args.seed)
    data_paths = _parse_data_paths(args.train_paths) if (args.train_paths and not args.eval_only) else [args.path]
    if args.row_level:
        instances = []
        for data_path in data_paths:
            instances.extend(load_temporal_instances(data_path, limit=None, group_by_document=False))
        if args.limit is not None:
            instances = instances[: args.limit]
        documents = None
    else:
        documents = []
        for data_path in data_paths:
            documents.extend(load_temporal_instances(data_path, limit=None, group_by_document=True))
        instances = expand_document_query_instances(documents)
        if args.limit is not None:
            instances = instances[: args.limit]
    train, dev = split_instances(instances, args.dev_fraction, args.seed)
    if args.eval_only:
        # Test/eval-only mode should score the full requested file, e.g.
        # MATRES/platinum.txt, rather than a random dev fraction.
        train = list(instances)
        dev = list(instances)
    print(f"dataset={args.path}", flush=True)
    if len(data_paths) > 1:
        print(f"train_paths={[str(path) for path in data_paths]}", flush=True)
    if documents is not None:
        print(f"documents={len(documents)} query_instances={len(instances)}", flush=True)
    print(f"loaded={len(instances)} train={len(train)} dev={len(dev)} device={args.device}", flush=True)
    print(
        f"pair_stats={_pair_count_stats(instances, args.max_events_per_instance, args.pair_selection, args.max_pairs_per_instance)} "
        f"max_events_per_instance={args.max_events_per_instance} "
        f"pair_selection={args.pair_selection} max_pairs_per_instance={args.max_pairs_per_instance}",
        flush=True,
    )
    warmup_epochs = args.epochs if args.warmup_epochs is None else args.warmup_epochs
    constraint_epochs = args.constraint_epochs
    set_boolean_executable_assertion(args.boolean_executable_assertion)

    if args.tnorm == "G" and not args.no_global_consistency:
        print(
            "WARNING: --tnorm G (Godel) with the consistency constraints enabled. Every "
            "rule in graph.py is an implication, and Godel gives an implication's "
            "ANTECEDENT exactly zero gradient — so e.g. before(p)->after(p_rev) can only "
            "raise after(p_rev), never lower an inconsistent before(p). Half of each "
            "symmetry correction is discarded. Use --tnorm P (the default) unless you are "
            "deliberately reproducing the old behaviour.",
            flush=True,
        )

    if (args.transitivity and not args.no_global_consistency
            and args.max_pairs_per_instance is not None
            and int(args.max_pairs_per_instance) < 3):
        print(
            f"WARNING: transitivity is enabled but --max-pairs-per-instance="
            f"{args.max_pairs_per_instance} < 3, so it cannot ground (it needs three "
            "chained pairs x->y, y->z, x->z). It will be compiled and evaluated every "
            "step without ever firing. Use --no-transitivity, or raise the cap.",
            flush=True,
        )

    print(
        f"warmup_epochs={warmup_epochs} constraint_epochs={constraint_epochs} "
        f"lr={args.lr} supervise_local_predicates={args.supervise_local_predicates} "
        f"global_consistency={not args.no_global_consistency} "
        f"tnorm={args.tnorm} exactly_one_label={args.exactly_one_label} "
        f"transitivity={args.transitivity} "
        f"boolean_executable_assertion={args.boolean_executable_assertion} "
        f"class_weights={_TEMPORAL_CLASS_WEIGHTS} "
        f"training_style={args.training_style} constraint_only={args.constraint_only} "
        f"constraint_loss_scale={args.constraint_loss_scale} beta={args.beta} "
        f"use_gumbel={args.use_gumbel} executable_pos_weight={args.executable_pos_weight}",
        flush=True,
    )
    train_data, _ctx, program = build_temporal_program(train, args)
    dev_data = compile_program_train_dataset(
        dev,
        _ctx,
        device=args.device,
        max_events_per_instance=args.max_events_per_instance,
        pair_selection=args.pair_selection,
        max_pairs_per_instance=args.max_pairs_per_instance,
    ) if dev else None
    if args.checkpoint:
        program.load(args.checkpoint, map_location=args.device)
        print(f"loaded_checkpoint={args.checkpoint}", flush=True)
    if args.eval_only:
        if not args.checkpoint:
            program.load(args.output, map_location=args.device)
            print(f"loaded_checkpoint={args.output}", flush=True)
    else:
        Optim = functools.partial(torch.optim.AdamW, lr=args.lr)
        # DomiKnowS phased training executes warmup/constraint epochs directly,
        # so initialize the main optimizer here like the examples do externally.
        program.opt = Optim(program.model.parameters())
        program.train(
            train_data,
            valid_set=dev_data,
            warmup_epochs=warmup_epochs,
            constraint_epochs=constraint_epochs,
            device=args.device,
            c_lr=args.lr,
            c_warmup_iters=args.c_warmup_iters,
            c_freq=args.c_freq,
            c_freq_increase=args.c_freq_increase,
            c_freq_increase_freq=args.c_freq_increase_freq,
            c_lr_decay=args.c_lr_decay,
            c_lr_decay_param=args.c_lr_decay_param,
            constraint_only=args.constraint_only,
            constraint_loss_scale=args.constraint_loss_scale,
            num_epochs=warmup_epochs + constraint_epochs,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        program.save(args.output)
        print(f"saved={args.output}", flush=True)
    if dev_data:
        relation_metrics = evaluate_temporal_relation_accuracy(dev_data, _ctx, program, device=args.device)
        print(f"dev_temporal_relation={relation_metrics}", flush=True)
        if not args.skip_condition_eval:
            condition = program.evaluate_condition(dev_data, device=args.device, return_dict=True)
            print(f"dev_condition={condition}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
