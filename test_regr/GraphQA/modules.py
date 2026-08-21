"""Qwen-backed predicate classifiers for GraphQA.

The learner follows the CLEVR pattern: a language backbone encodes bounded
object/symbol/pair prompts, and small heads emit DomiKnowS-aligned predicate
classes. It does not generate text.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .graph import OBJECT_SYMBOL_RELATIONS, canonical_relation, collect_kb_relations, collect_object_relations

NO_RELATION_LABEL = "NoRelation"


@dataclass
class GraphQAPredicateBatch:
    object_symbol_logits: torch.Tensor
    symbol_pair_logits: torch.Tensor
    object_pair_logits: torch.Tensor
    object_symbol_labels: list[str]
    symbol_pair_labels: list[str]
    object_pair_labels: list[str]


class GraphQAPredicateClassifier(torch.nn.Module):
    def __init__(
        self,
        model_path="Qwen/Qwen3-8B",
        object_symbol_labels=None,
        symbol_pair_labels=None,
        object_pair_labels=None,
        device="cpu",
        freeze_backbone=True,
        lora_r=0,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=None,
        max_length=128,
        encode_batch_size=None,
    ):
        super().__init__()
        self.device_name = device
        self.max_length = int(max_length)
        self.encode_batch_size = encode_batch_size
        self.object_symbol_labels = list(object_symbol_labels or sorted(OBJECT_SYMBOL_RELATIONS))
        self.symbol_pair_labels = list(symbol_pair_labels or ["TypeOf"])
        self.object_pair_labels = list(object_pair_labels or [])
        self.lora_enabled = int(lora_r) > 0

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if str(device).startswith("cuda"):
            model_kwargs["dtype"] = torch.float16
        self.backbone = AutoModel.from_pretrained(model_path, **model_kwargs)

        if self.lora_enabled:
            from peft import LoraConfig, TaskType, get_peft_model

            target_modules = lora_target_modules or ["q_proj", "v_proj"]
            config = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=target_modules,
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
        hidden_size = int(getattr(self.backbone.config, "hidden_size"))
        self.object_symbol_head = torch.nn.Linear(hidden_size, len(self.object_symbol_labels)).to(device)
        self.symbol_pair_head = torch.nn.Linear(hidden_size, len(self.symbol_pair_labels)).to(device)
        self.object_pair_head = torch.nn.Linear(hidden_size, max(1, len(self.object_pair_labels))).to(device)

        if freeze_backbone and not self.backbone_has_trainable_parameters():
            self.backbone.eval()

    def backbone_has_trainable_parameters(self):
        return any(param.requires_grad for param in self.backbone.parameters())

    def encode(self, prompts):
        prompts = list(prompts)
        if not prompts:
            hidden_size = int(getattr(self.backbone.config, "hidden_size"))
            return torch.empty((0, hidden_size), dtype=torch.float32, device=self.device_name)
        batch_size = self.encode_batch_size
        if batch_size is None:
            batch_size = 1 if self.backbone_has_trainable_parameters() else len(prompts)
        batch_size = max(1, int(batch_size))
        features = []
        grad_enabled = self.training and self.backbone_has_trainable_parameters()
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
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
            features.append(((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)).float())
        return torch.cat(features, dim=0)

    def forward_examples(self, examples):
        prompts = [example["prompt"] for example in examples]
        features = self.encode(prompts)
        kind = examples[0]["kind"] if examples else "object_symbol"
        if kind == "object_symbol":
            return _mask_logits_to_allowed_labels(self.object_symbol_head(features), prompts, self.object_symbol_labels)
        if kind == "symbol_pair":
            return _mask_logits_to_allowed_labels(self.symbol_pair_head(features), prompts, self.symbol_pair_labels)
        if kind == "object_pair":
            return _mask_logits_to_allowed_labels(self.object_pair_head(features), prompts, self.object_pair_labels)
        raise ValueError(f"Unknown GraphQA example kind: {kind!r}")


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


def label_spaces(instances):
    return {
        "object_symbol": sorted(OBJECT_SYMBOL_RELATIONS) + [NO_RELATION_LABEL],
        "symbol_pair": collect_kb_relations(instances) + [NO_RELATION_LABEL],
        "object_pair": collect_object_relations(instances) + [NO_RELATION_LABEL],
    }


def create_predicate_examples(instance):
    examples = []
    facts = list(instance.get("facts", [])) or list(instance.get("visual_facts", [])) + list(instance.get("kb_facts", []))
    query = instance.get("query", {})
    for pred, left, right in facts:
        pred = canonical_relation(pred)
        if pred in OBJECT_SYMBOL_RELATIONS:
            examples.append({
                "kind": "object_symbol",
                "label": pred,
                "prompt": _object_symbol_prompt(instance, left, right, query),
            })
        elif pred in collect_kb_relations([instance]):
            examples.append({
                "kind": "symbol_pair",
                "label": pred,
                "prompt": _symbol_pair_prompt(instance, left, right, query),
            })
        elif pred in collect_object_relations([instance]):
            examples.append({
                "kind": "object_pair",
                "label": pred,
                "prompt": _object_pair_prompt(instance, left, right, query),
            })
    return examples


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


def _format_relevant_facts(instance, *, left=None, right=None, max_facts=16, max_scan=4096):
    left = None if left is None else str(left)
    right = None if right is None else str(right)
    exact = []
    touching = []
    seen = set()
    scanned = 0
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
            pred, fact_left, fact_right = item
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
    lines = [f"{pred}({left}, {right})" for pred, left, right in selected]
    if scanned >= int(max_scan):
        lines.append("... evidence scan capped")
    return "\n".join(lines)


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


def _object_symbol_feature_prompt(instance, obj, symbol, query=None, labels=None):
    labels = labels or sorted(OBJECT_SYMBOL_RELATIONS) + [NO_RELATION_LABEL]
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


def _object_pair_feature_prompt(instance, src, dst, query=None, labels=None):
    labels = labels or collect_object_relations([instance]) + [NO_RELATION_LABEL]
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


def _object_symbol_prompt(instance, obj, symbol, query):
    return "\n".join([
        "Classify the GraphQA object-symbol predicate.",
        "Use the provided bounded scene/KG facts as evidence; do not infer from object id alone.",
        f"Object: {obj}",
        f"Symbol: {symbol}",
        "Relevant facts:",
        _format_relevant_facts(instance, left=obj, right=symbol),
        f"Query: {query}",
        f"Labels: Name, ObjectType, ObjectCategory, Attribute, {NO_RELATION_LABEL}.",
    ])


def _symbol_pair_prompt(instance, src, dst, query):
    return "\n".join([
        "Classify the GraphQA symbol-pair KG predicate.",
        "Use the provided bounded scene/KG facts as evidence; do not infer from symbol names alone.",
        f"Source symbol: {src}",
        f"Destination symbol: {dst}",
        f"Query: {query}",
        "Relevant facts:",
        _format_relevant_facts(instance, left=src, right=dst),
        f"Labels include: {NO_RELATION_LABEL}.",
    ])


def _object_pair_prompt(instance, src, dst, query):
    return "\n".join([
        "Classify the GraphQA object-object relation predicate.",
        "Use the provided bounded scene/KG facts as evidence; do not infer from object id alone.",
        f"Source object: {src}",
        f"Destination object: {dst}",
        f"Query: {query}",
        "Relevant facts:",
        _format_relevant_facts(instance, left=src, right=dst),
        f"Labels include: {NO_RELATION_LABEL}.",
    ])
