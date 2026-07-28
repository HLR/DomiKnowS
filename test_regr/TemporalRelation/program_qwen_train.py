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
from domiknows.sensor.pytorch.sensors import FunctionalReaderSensor, FunctionalSensor, ReaderSensor

from .config import TEMPORAL_CONFIG
from .dataset import DEFAULT_TEMPORAL_DATA_ROOT, load_temporal_instances
from .execution import create_executable_instance, mark_text_for_pair
from .graph import (
    EXTENDED_LABELS,
    MATRES_LABELS,
    TEMPORAL_LABELS,
    create_temporal_graph,
    unpack_pair,
)
from .program import BinaryOracleLearner, _tensor

DEFAULT_MODEL = TEMPORAL_CONFIG.training_model
DEFAULT_OUTPUT = TEMPORAL_CONFIG.output_path("qwen3_8b_temporal_domiknows_program.pt")
LOCAL_IGNORE_LABEL = -100
_TEMPORAL_CLASS_WEIGHTS = None
_TEMPORAL_DATASET_MASK = None


#: Relations each corpus can actually express. Under a union label space the
#: joint head has one output per relation across *all* corpora, but a MATRES row
#: can never be ``Includes`` and a TB-Dense row never uses MATRES's start-point
#: ``Equal``. Training the head to rank a label its source corpus could not have
#: produced teaches it that the label is simply rare, rather than inapplicable.
DATASET_LEGAL_LABELS = {
    "matres": ("Before", "After", "Equal", "Vague"),
    "tbdense": ("Before", "After", "Includes", "IsIncluded", "Simultaneous", "Vague"),
}


def legal_label_mask(dataset_name, labels):
    """Boolean ``[K]`` mask of relations ``dataset_name`` can express."""
    legal = DATASET_LEGAL_LABELS.get(str(dataset_name).lower())
    if legal is None:
        return torch.ones(len(labels), dtype=torch.bool)
    return torch.tensor([label in legal for label in labels], dtype=torch.bool)


def apply_dataset_mask(logits, dataset_mask):
    """Apply a per-row corpus legality mask to temporal-relation logits."""
    if dataset_mask is None:
        return logits
    mask = torch.as_tensor(dataset_mask, dtype=torch.bool, device=logits.device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.shape[0] == 1 and logits.shape[0] != 1:
        mask = mask.expand(logits.shape[0], -1)
    if mask.shape != logits.shape:
        raise ValueError(
            f"dataset_mask shape {tuple(mask.shape)} does not match "
            f"temporal logits {tuple(logits.shape)}")
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("dataset_mask must permit at least one label per row")
    return logits.masked_fill(~mask, float("-inf"))


class WeightedTemporalCrossEntropyLoss(torch.nn.Module):
    """Cross-entropy with optional temporal class weights and ignored rows.

    ``dataset_mask`` (a ``[K]`` bool) restricts the head to the relations the
    current corpus can express by driving the others to ``-inf`` before the
    softmax, so they receive no gradient and can never be predicted.
    """

    def __init__(self, weights=None, dataset_mask=None):
        super().__init__()
        if weights is None:
            self.register_buffer("weights", None, persistent=False)
        else:
            self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float32), persistent=False)
        if dataset_mask is None:
            self.register_buffer("dataset_mask", None, persistent=False)
        else:
            self.register_buffer("dataset_mask", torch.as_tensor(dataset_mask, dtype=torch.bool), persistent=False)

    def forward(self, input, target, *args, **kwargs):
        input = input.view(-1, input.shape[-1])
        target = target.view(-1).to(dtype=torch.long, device=input.device)
        if self.dataset_mask is not None and self.dataset_mask.numel() == input.shape[-1]:
            illegal = ~self.dataset_mask.to(input.device)
            if illegal.any():
                input = input.masked_fill(illegal.unsqueeze(0), float("-inf"))
        weight = self.weights.to(input.device) if self.weights is not None else None
        return F.cross_entropy(input, target, weight=weight, ignore_index=LOCAL_IGNORE_LABEL)


def _make_temporal_ce_loss(weights=None, dataset_mask=None):
    if weights is None and dataset_mask is None:
        return NBCrossEntropyLoss()
    return WeightedTemporalCrossEntropyLoss(weights, dataset_mask=dataset_mask)


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
        kwargs.setdefault(
            "loss",
            MacroAverageTracker(_make_temporal_ce_loss(
                _TEMPORAL_CLASS_WEIGHTS, _TEMPORAL_DATASET_MASK)),
        )
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
        choices = ", ".join(TEMPORAL_LABELS)
        return (
            "Classify the temporal relation from event E1 to event E2. "
            f"Choose exactly one label from: {choices}.\n"
            f"Text: {marked_text}\n"
            "Answer:"
        )

    def forward(self, prompts, dataset_mask=None):
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
        scores = torch.cat(chunks, dim=0)
        return apply_dataset_mask(scores, dataset_mask)

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
    global _TEMPORAL_DATASET_MASK
    labels, dataset_names = _activate_labels_for_instances(
        instances, getattr(args, "_active_dataset_names", None))
    _TEMPORAL_DATASET_MASK = (
        legal_label_mask(next(iter(dataset_names)), labels)
        if len(dataset_names) == 1 else None
    )
    ctx = create_temporal_graph(
        instances,
        include_global_constraints=not args.no_global_consistency,
        include_exactly_one=getattr(args, "exactly_one_label", True),
        include_transitivity=getattr(args, "transitivity", True),
        labels=labels,
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
        include_global_constraint_loss=args.global_constraint_loss,
        global_constraint_loss_weight=args.global_constraint_loss_weight,
        executable_constraint_loss_weight=args.executable_constraint_loss_weight,
        query_loss=functools.partial(
            _make_temporal_ce_loss,
            _TEMPORAL_CLASS_WEIGHTS,
            _TEMPORAL_DATASET_MASK,
        ),
    )
    # Configure the remaining constraint-model execution options after
    # construction; query_loss is routed there without leaking to SolverModel.
    if hasattr(program, "cmodel"):
        program.cmodel.tnorm = args.tnorm
        program.cmodel.counting_tnorm = getattr(program.cmodel, "counting_tnorm", None) or args.tnorm
        if hasattr(program.cmodel, "pos_weight"):
            program.cmodel.pos_weight = float(args.executable_pos_weight)

    report_constraint_wiring(ctx, program, dataset, args)
    if getattr(args, "constraint_gradient_check", True):
        verify_constraint_gradient_flow(program, dataset, args)
    return dataset, ctx, program


def refresh_constraint_poi(program, ctx):
    """Add constraint properties compiled *after* the program was constructed.

    ``PoiModel`` snapshots ``poi`` into a list at construction, expanding each
    Concept into the properties it had *at that moment*. ``compile_executable``
    adds one property per executable constraint, so any dataset compiled after
    the program — typically the dev/test split — contributes ``ELC`` properties
    that are absent from that snapshot and therefore never evaluated.

    The visible symptom is an evaluation whose counters stay at zero while the
    progress bar completes: the constraint datanode exists and carries
    ``label/label``, but not the ``ELC<n>/label`` entry the active constraint is
    looked up by. Training on the split the graph was built with looks fine,
    which is what makes it easy to miss.

    Returns the number of properties added.
    """
    poi = getattr(getattr(program, "model", None), "poi", None)
    if poi is None:
        return 0
    known = {id(prop) for prop in poi}
    added = [prop for prop in ctx.graph.constraint.values() if id(prop) not in known]
    poi.extend(added)
    return len(added)


def _report_constraint_groundings(ctx, program, dataset):
    """Per-constraint grounding report on one item, plus the EventPair count.

    Wiring can be correct while every constraint still has *zero groundings* —
    that is what happened when the candidate sensor produced no EventPair
    datanodes: the rules compiled, evaluated, and returned ``None`` or a
    constant. These two numbers localise that immediately.
    """
    if not dataset:
        return {}
    try:
        datanode = next(iter(program.populate([dataset[0]], device=program.device)))
        pairs = len(datanode.findDatanodes(select=ctx.event_pair))
        datanode.inferLocal(keys=("softmax",))
        losses = datanode.calculateLcLoss(tnorm=getattr(program.cmodel, "tnorm", "P"))
    except Exception as exc:  # noqa: BLE001 - diagnostics must never block training
        print(f"[constraints] grounding probe skipped ({type(exc).__name__}: {exc})", flush=True)
        return {}

    name_of = {lc.lcName: lc.name for _k, lc in ctx.graph.logicalConstrainsRecursive
               if getattr(lc, "headLC", False)}
    parts = []
    counts = {}
    for key, result in losses.items():
        tensor = result.get("lossTensor") if isinstance(result, dict) else None
        label = name_of.get(key, key)
        counts[label] = 0 if tensor is None else tensor.numel()
        parts.append(f"{label}={'NO-GROUNDING' if tensor is None else tensor.numel()}")
    print(f"[constraints] EventPair groundings={pairs} | per-rule: {', '.join(parts)}",
          flush=True)
    if pairs == 0:
        print("[constraints] WARNING: zero EventPair groundings — every rule "
              "quantified over EventPair is vacuous.", flush=True)
    return counts


def verify_constraint_gradient_flow(program, dataset, args):
    """Assert the constraint loss actually reaches the model's parameters.

    ``report_constraint_wiring`` checks that the constraint *pathways* are on.
    That is not enough: the loss can be non-zero and differentiable while its
    autograd graph never touches a learnable parameter — which is exactly what
    happened here (``closs=3.3863, requires_grad=True``, yet 0 of 2 parameters
    received a gradient), because the rules had no groundings to bind to.

    One forward/backward on a single item, before training starts, converts that
    silent no-op into an immediate failure.
    """
    if not dataset or args.constraint_epochs <= 0:
        return None

    was_training = program.model.training
    program.model.train()
    program.cmodel.train()
    program.model.zero_grad(set_to_none=True)
    try:
        _mloss, _metric, *output = program.model(dataset[0])
        closs, *_ = program.cmodel(output[1])
        if not (torch.is_tensor(closs) and closs.requires_grad):
            raise RuntimeError(
                f"constraint loss is not differentiable (closs={closs!r}). "
                "Training cannot be influenced by any constraint.")
        closs.backward()
        params = list(program.model.parameters())
        reached = [p for p in params if p.grad is not None and p.grad.abs().sum() > 0]
        print(f"[constraints] gradient flow: closs={float(closs.detach()):.4f} -> "
              f"{len(reached)}/{len(params)} model parameter tensors receive a "
              f"non-zero gradient", flush=True)
        if not reached:
            raise RuntimeError(
                f"constraint loss is non-zero (closs={float(closs.detach()):.4f}) but "
                "reaches NO model parameter — training would be identical with and "
                "without constraints. Usual cause: the rules have zero groundings "
                "(check the EventPair grounding count above). Use "
                "--no-constraint-gradient-check to bypass this assertion."
            )
        return len(reached)
    finally:
        program.model.zero_grad(set_to_none=True)
        program.cmodel.zero_grad(set_to_none=True)
        if not was_training:
            program.model.eval()


def report_constraint_wiring(ctx, program, dataset, args):
    """Print, and sanity-check, which constraint pathways are actually live.

    Two independent switches have to be on for a constraint to affect training,
    and each has silently defaulted to off at some point:

    * the **executable** queryL/iotaL loss needs a label sensor on the graph's
      constraint concept, otherwise no constraint datanode is built and every
      item is skipped (seen as ``query_total=0``);
    * the **global** consistency loss needs ``include_global_constraint_loss``,
      which ``InferenceModel`` defaults to False — with it off, the rules are
      compiled and evaluated but their loss is discarded.

    With both off, a "constraint" run is byte-identical to a supervised one.
    That happened, and it cost a full grid of GPU-hours to notice, so it is now
    reported up front and raises rather than training on a silent no-op.
    """
    cmodel = getattr(program, "cmodel", None)
    constraint_props = list(ctx.graph.get_constraint_concept().keys())
    has_label_sensor = any(str(p) == "label" for p in constraint_props)
    n_exec = len(getattr(ctx.graph, "executableLCs", {}) or {})
    global_heads = [lc.name for _k, lc in ctx.graph.logicalConstrainsRecursive
                    if getattr(lc, "headLC", False)]
    global_on = bool(getattr(cmodel, "include_global_constraint_loss", False))

    print(
        "[constraints] executable: "
        f"{'LIVE' if (has_label_sensor and n_exec) else 'INERT'} "
        f"(label_sensor={has_label_sensor}, compiled_lcs={n_exec}, "
        f"weight={getattr(cmodel, 'executable_constraint_loss_weight', None)}) | "
        "global: "
        f"{'LIVE' if (global_on and global_heads) else 'INERT'} "
        f"(enabled={global_on}, head_lcs={len(global_heads)}, "
        f"weight={getattr(cmodel, 'global_constraint_loss_weight', None)}) | "
        f"tnorm={getattr(cmodel, 'tnorm', None)}",
        flush=True,
    )
    if global_heads:
        print(f"[constraints] global rules: {', '.join(global_heads)}", flush=True)

    program.constraint_grounding_counts = _report_constraint_groundings(
        ctx, program, dataset)

    executable_live = has_label_sensor and n_exec > 0
    global_live = global_on and bool(global_heads)
    if args.constraint_epochs > 0 and not (executable_live or global_live):
        raise RuntimeError(
            "constraint_epochs > 0 but NO constraint pathway is live: "
            f"executable(label_sensor={has_label_sensor}, compiled_lcs={n_exec}), "
            f"global(enabled={global_on}, head_lcs={len(global_heads)}). "
            "Training would be identical to a supervised run. Enable "
            "--global-constraint-loss, attach the constraint label sensor, or set "
            "--constraint-epochs 0 to run supervised on purpose."
        )


def attach_program_train_sensors(ctx, args):
    device = args.device

    # The executable queryL/iotaL objective is driven by a label on the graph's
    # constraint concept. Without this sensor no constraint datanode is ever
    # built, so ``getExecutableConstraintLabels()`` returns {} and
    # ``InferenceProgram.evaluate_condition`` skips every item (visible as
    # boolean_total=0 / query_total=0 while the progress bar still completes) —
    # and the executable constraint loss silently contributes nothing during
    # training. Same one-line wiring as the CLEVR reference path
    # (``test_regr/Clever/main.py``).
    ctx.graph.constraint["label"] = ReaderSensor(keyword="logic_label", label=True)

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
    ctx.event_pair["dataset_mask"] = FunctionalReaderSensor(
        keyword="dataset_mask",
        forward=lambda data, _device=device: torch.as_tensor(
            data, dtype=torch.bool, device=_device),
    )
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
        "dataset_mask",
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
        loss=_make_temporal_ce_loss(
            _TEMPORAL_CLASS_WEIGHTS, _TEMPORAL_DATASET_MASK),
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
    """Whether this ordered token pair is an allowed EventPair candidate.

    ``CompositionCandidateSensor.forward_wrap`` passes the candidate datanodes
    as keywords named after the relation's ``has_a`` arguments — here ``e1`` and
    ``e2`` (see ``event_pair.has_a(e1=token, e2=token)`` in graph.py), *not*
    ``arg1``/``arg2``. Keying only on ``arg1``/``arg2`` therefore left them None
    and returned False for every combination, so **no EventPair datanode was
    ever built** and every constraint quantified over EventPair silently had
    zero groundings. Read the datanodes positionally from the keywords, keeping
    the explicit names as a fallback.
    """
    nodes = [arg for arg in (arg1, arg2) if arg is not None]
    if len(nodes) < 2:
        nodes = [value for value in _kwargs.values() if hasattr(value, "getAttribute")]
    if len(nodes) < 2:
        return False

    def position(node):
        return int(node.getAttribute("index").detach().cpu().view(-1)[0].item())

    left, right = position(nodes[0]), position(nodes[1])
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
        "dataset": _instance_dataset(instance),
        "dataset_mask": legal_label_mask(
            _instance_dataset(instance), TEMPORAL_LABELS
        ).to(device).unsqueeze(0).expand(len(candidate_pairs), -1).clone(),
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


def _instance_dataset(instance):
    dataset_name = str(instance.get("dataset") or "matres").lower()
    if dataset_name not in DATASET_LEGAL_LABELS:
        raise ValueError(
            f"Unsupported or missing temporal dataset identity {dataset_name!r}; "
            f"expected one of {sorted(DATASET_LEGAL_LABELS)}")
    return dataset_name


def _activate_labels_for_instances(instances, dataset_names=None):
    dataset_names = (
        set(dataset_names)
        if dataset_names is not None
        else {_instance_dataset(instance) for instance in instances}
    )
    if not dataset_names:
        dataset_names = {"matres"}
    labels = EXTENDED_LABELS if "tbdense" in dataset_names else MATRES_LABELS
    TEMPORAL_LABELS.set(labels)
    return labels, dataset_names




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
        "--dataset",
        choices=["auto", "matres", "tbdense"],
        default="auto",
        help="Corpus identity for --path. Auto uses row metadata or the path name.",
    )
    parser.add_argument(
        "--train-paths",
        default=None,
        help=(
            "Comma-separated dataset files to concatenate for training. "
            "When unset, --path is used. Eval-only still scores --path."
        ),
    )
    parser.add_argument(
        "--train-datasets",
        default=None,
        help=(
            "Comma-separated corpus identities aligned with --train-paths "
            "(auto, matres, or tbdense). Defaults to auto for every path."
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
    parser.add_argument(
        "--global-constraint-loss",
        dest="global_constraint_loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include the graph-global consistency constraints (symmetry, inverse, "
            "exactly-one, transitivity) in the training loss. InferenceModel "
            "defaults this to False, so before this flag existed those constraints "
            "were compiled and evaluated but their loss was DISCARDED — training "
            "was identical with and without them. Use --no-global-constraint-loss "
            "for a constraint-free control."
        ),
    )
    parser.add_argument("--global-constraint-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--constraint-gradient-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before training, assert the constraint loss actually reaches a model "
            "parameter. Catches the case where closs is non-zero and differentiable "
            "but its graph never touches a learnable weight, so constraint training "
            "is silently identical to supervised training."
        ),
    )
    parser.add_argument(
        "--executable-constraint-loss-weight", type=float, default=1.0,
        help="Weight of the executable queryL/iotaL constraint loss.",
    )
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
        "--grounding-only",
        action="store_true",
        help=(
            "Build one program, print constraint groundings, require positive "
            "before-transitivity groundings when enabled, and exit before training."
        ),
    )
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


def _parse_dataset_names(value, expected):
    if not value:
        return ["auto"] * expected
    names = [item.strip().lower() for item in str(value).split(",") if item.strip()]
    invalid = [name for name in names if name not in {"auto", "matres", "tbdense"}]
    if invalid:
        raise ValueError(f"Unsupported --train-datasets value(s): {invalid}")
    if len(names) != expected:
        raise ValueError(
            f"--train-datasets supplied {len(names)} value(s) for {expected} "
            "--train-paths entries")
    return names


def main():
    global _TEMPORAL_CLASS_WEIGHTS
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.train_datasets and (not args.train_paths or args.eval_only):
        raise ValueError("--train-datasets requires active --train-paths training")
    data_paths = (
        _parse_data_paths(args.train_paths)
        if (args.train_paths and not args.eval_only)
        else [args.path]
    )
    dataset_names = (
        _parse_dataset_names(args.train_datasets, len(data_paths))
        if (args.train_paths and not args.eval_only)
        else [args.dataset]
    )
    if args.row_level:
        instances = []
        for data_path, dataset_name in zip(data_paths, dataset_names):
            instances.extend(load_temporal_instances(
                data_path, limit=None, group_by_document=False,
                dataset_name=dataset_name))
        if args.limit is not None:
            instances = instances[: args.limit]
        documents = None
    else:
        documents = []
        for data_path, dataset_name in zip(data_paths, dataset_names):
            documents.extend(load_temporal_instances(
                data_path, limit=None, group_by_document=True,
                dataset_name=dataset_name))
        instances = expand_document_query_instances(documents)
        if args.limit is not None:
            instances = instances[: args.limit]
    _labels, active_dataset_names = _activate_labels_for_instances(instances)
    args._active_dataset_names = active_dataset_names
    _TEMPORAL_CLASS_WEIGHTS = _parse_temporal_class_weights(args)
    train, dev = split_instances(instances, args.dev_fraction, args.seed)
    if args.eval_only:
        # Test/eval-only mode should score the full requested file, e.g.
        # MATRES/platinum.txt, rather than a random dev fraction.
        train = list(instances)
        dev = list(instances)
    print(f"dataset={args.path}", flush=True)
    if len(data_paths) > 1:
        print(f"train_paths={[str(path) for path in data_paths]}", flush=True)
    print(f"dataset_identities={sorted({_instance_dataset(item) for item in instances})}",
          flush=True)
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
    if args.grounding_only:
        transitive = getattr(program, "constraint_grounding_counts", {}).get(
            "temporal_before_transitive", 0)
        if args.transitivity and not args.no_global_consistency and transitive <= 0:
            raise RuntimeError(
                "temporal_before_transitive has zero groundings; mixed-corpus "
                "training is blocked")
        print(
            f"[constraints] grounding-only validation passed: "
            f"temporal_before_transitive={transitive}",
            flush=True,
        )
        return 0
    dev_data = compile_program_train_dataset(
        dev,
        _ctx,
        device=args.device,
        max_events_per_instance=args.max_events_per_instance,
        pair_selection=args.pair_selection,
        max_pairs_per_instance=args.max_pairs_per_instance,
    ) if dev else None
    if dev_data is not None:
        # The dev split's executable constraints were compiled after the program
        # snapshotted its POI, so their properties must be added or every dev
        # constraint metric silently reports zero.
        added = refresh_constraint_poi(program, _ctx)
        print(f"[constraints] dev split added {added} executable constraint "
              f"propert{'y' if added == 1 else 'ies'} to the POI", flush=True)
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
