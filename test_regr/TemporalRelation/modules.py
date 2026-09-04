"""CLEVR-style predicate classifiers for TemporalRelation.

Unlike ``llm_inference.py`` (a zero-shot text-generation baseline), this module
exposes fixed-size logits that line up with DomiKnowS concepts:

* event/query_event1/query_event2: binary logits ``[negative, positive]``
* temporal_relation: 4-way logits over ``Before/After/Equal/Vague``

This mirrors the CLEVR example where each visual predicate is attached to a
ModuleLearner returning concept-aligned tensors rather than generated text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .execution import create_candidate_event_pairs, mark_text_for_pair
from .graph import TEMPORAL_LABELS, unpack_pair


@dataclass
class TemporalPredicateBatch:
    event_ids: list[str]
    pair_ids: list[tuple[str, str]]
    event_logits: torch.Tensor
    query_event1_logits: torch.Tensor
    query_event2_logits: torch.Tensor
    temporal_relation_logits: torch.Tensor


def positive_negative_logits(is_positive: Iterable[bool], confidence=6.0, device="cpu"):
    """Return binary logits shaped like CLEVR predicate tensors: [no, yes]."""
    rows = []
    for value in is_positive:
        if value:
            rows.append([-float(confidence), float(confidence)])
        else:
            rows.append([float(confidence), -float(confidence)])
    return torch.tensor(rows, dtype=torch.float32, device=device)


def one_hot_logits(indices, class_count, confidence=6.0, device="cpu"):
    """Return multiclass logits with one high-confidence class per row."""
    logits = torch.full((len(indices), class_count), -float(confidence), dtype=torch.float32, device=device)
    for row, index in enumerate(indices):
        if index is not None:
            logits[row, int(index)] = float(confidence)
    return logits


class OracleTemporalPredicateClassifier(torch.nn.Module):
    """Perfect predicate module for tests and oracle execution.

    This is the TemporalRelation analogue of CLEVR oracle mode: it returns
    concept-aligned tensors, not strings. ``event`` is treated as observed true
    for every listed MATRES event.
    """

    def __init__(self, confidence=6.0, device="cpu"):
        super().__init__()
        self.confidence = float(confidence)
        self.device_name = device

    def forward(self, instance):
        events = list(instance.get("events", []))
        event_ids = [_event_id(event) for event in events]
        query_pair = instance.get("query_pair") or (instance.get("event_pairs") or [{}])[0]
        query_e1, query_e2, _ = unpack_pair(query_pair)

        pairs = create_candidate_event_pairs(instance)
        pair_ids = []
        label_indices = []
        for pair in pairs:
            e1, e2, label = unpack_pair(pair)
            pair_ids.append((e1, e2))
            label_indices.append(TEMPORAL_LABELS.index(label) if label in TEMPORAL_LABELS else None)

        return TemporalPredicateBatch(
            event_ids=event_ids,
            pair_ids=pair_ids,
            event_logits=positive_negative_logits([True] * len(event_ids), self.confidence, self.device_name),
            query_event1_logits=positive_negative_logits(
                [event_id == query_e1 for event_id in event_ids], self.confidence, self.device_name
            ),
            query_event2_logits=positive_negative_logits(
                [event_id == query_e2 for event_id in event_ids], self.confidence, self.device_name
            ),
            temporal_relation_logits=one_hot_logits(
                label_indices, len(TEMPORAL_LABELS), self.confidence, self.device_name
            ),
        )


class TemporalPredicateClassifier(torch.nn.Module):
    """Qwen/causal-LM backbone with DomiKnowS-aligned predicate heads.

    The backbone produces hidden states for bounded event/pair prompts. Linear
    heads map those hidden states directly to graph concept logits. No text is
    generated and therefore no ``max_new_tokens`` is needed.
    """

    def __init__(
        self,
        model_path="Qwen/Qwen3-8B",
        device="cpu",
        freeze_backbone=True,
        lora_r=0,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=None,
        max_length=512,
        encode_batch_size=None,
        supervised_pairs_only=False,
        max_pairs_per_instance=None,
    ):
        super().__init__()
        self.device_name = device
        self.max_length = int(max_length)
        self.encode_batch_size = encode_batch_size
        self.supervised_pairs_only = bool(supervised_pairs_only)
        self.max_pairs_per_instance = max_pairs_per_instance
        self.lora_enabled = int(lora_r) > 0
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if str(device).startswith("cuda"):
            model_kwargs["dtype"] = torch.float16
        self.backbone = AutoModel.from_pretrained(model_path, **model_kwargs)
        hidden_size = int(getattr(self.backbone.config, "hidden_size"))

        if self.lora_enabled:
            from peft import LoraConfig, TaskType, get_peft_model

            target_modules = lora_target_modules or [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            lora_config = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=target_modules,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
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
        self.event_head = torch.nn.Linear(hidden_size, 2).to(device)
        self.query_event1_head = torch.nn.Linear(hidden_size, 2).to(device)
        self.query_event2_head = torch.nn.Linear(hidden_size, 2).to(device)
        self.temporal_relation_head = torch.nn.Linear(hidden_size, len(TEMPORAL_LABELS)).to(device)
        if freeze_backbone and not self.backbone_has_trainable_parameters():
            self.backbone.eval()

    def backbone_has_trainable_parameters(self):
        return any(param.requires_grad for param in self.backbone.parameters())

    def forward(self, instance):
        events = list(instance.get("events", []))
        event_ids = [_event_id(event) for event in events]
        event_features = self._encode_texts([event_prompt(instance, event) for event in events])

        pairs = self._select_pairs(instance)
        pair_ids = []
        pair_prompts = []
        for pair in pairs:
            e1, e2, _ = unpack_pair(pair)
            pair_ids.append((e1, e2))
            pair_prompts.append(pair_prompt(instance, e1, e2))
        pair_features = self._encode_texts(pair_prompts)

        return TemporalPredicateBatch(
            event_ids=event_ids,
            pair_ids=pair_ids,
            event_logits=self.event_head(event_features),
            query_event1_logits=self.query_event1_head(event_features),
            query_event2_logits=self.query_event2_head(event_features),
            temporal_relation_logits=self.temporal_relation_head(pair_features),
        )

    def _select_pairs(self, instance):
        pairs = list(instance.get("event_pairs", [])) if self.supervised_pairs_only else create_candidate_event_pairs(instance)
        if self.max_pairs_per_instance is not None and len(pairs) > int(self.max_pairs_per_instance):
            pairs = pairs[: int(self.max_pairs_per_instance)]
        return pairs

    def _encode_texts(self, texts):
        if not texts:
            hidden_size = int(getattr(self.backbone.config, "hidden_size"))
            return torch.empty((0, hidden_size), dtype=torch.float32, device=self.device_name)
        batch_size = self.encode_batch_size
        if batch_size is None:
            batch_size = 1 if self.backbone_has_trainable_parameters() else len(texts)
        batch_size = max(1, int(batch_size))
        pooled_chunks = []
        grad_enabled = self.backbone_has_trainable_parameters()
        for start in range(0, len(texts), batch_size):
            chunk = list(texts)[start : start + batch_size]
            inputs = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device_name)
            with torch.set_grad_enabled(grad_enabled):
                outputs = self.backbone(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled_chunks.append(pooled.float())
        return torch.cat(pooled_chunks, dim=0)


def event_prompt(instance, event):
    event_id = _event_id(event)
    event_text = event.get("text") if isinstance(event, dict) else event_id
    query_pair = instance.get("query_pair") or (instance.get("event_pairs") or [{}])[0]
    query_e1, query_e2, _ = unpack_pair(query_pair)
    return "\n".join(
        [
            "Classify predicates for one candidate event mention.",
            f"Text: {instance.get('text') or instance.get('doc_id') or ''}",
            f"Candidate event: {event_id}: {event_text}",
            f"TemporalRelation query pair: first={query_e1}, second={query_e2}",
        ]
    )


def pair_prompt(instance, e1, e2):
    return "\n".join(
        [
            "Classify the temporal_relation concept for this event pair.",
            "Labels: Before, After, Equal, Vague.",
            f"Text: {mark_text_for_pair(instance, e1, e2)}",
        ]
    )


def predictions_from_logits(batch: TemporalPredicateBatch):
    """Convert concept logits to readable predictions for evaluation/debugging."""
    relation_indices = torch.argmax(batch.temporal_relation_logits.detach(), dim=-1).cpu().tolist()
    q1 = torch.argmax(batch.query_event1_logits.detach(), dim=-1).cpu().tolist()
    q2 = torch.argmax(batch.query_event2_logits.detach(), dim=-1).cpu().tolist()
    return {
        "query_event_groundings": [
            {"event_id": event_id, "query_event1": bool(q1_i), "query_event2": bool(q2_i)}
            for event_id, q1_i, q2_i in zip(batch.event_ids, q1, q2)
        ],
        "event_pair_predictions": [
            {"e1": e1, "e2": e2, "label": TEMPORAL_LABELS[index]}
            for (e1, e2), index in zip(batch.pair_ids, relation_indices)
        ],
    }


def _event_id(event):
    return event.get("id") if isinstance(event, dict) else event
