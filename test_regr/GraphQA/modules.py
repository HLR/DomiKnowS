"""Qwen-backed predicate classifiers for GraphQA.

The learner follows the CLEVR pattern: a language backbone encodes bounded
object/symbol/pair prompts, and small heads emit DomiKnowS-aligned predicate
classes. It does not generate text.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .graph import OBJECT_SYMBOL_RELATIONS, canonical_relation, collect_kb_relations, collect_object_relations


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
            return self.object_symbol_head(features)
        if kind == "symbol_pair":
            return self.symbol_pair_head(features)
        if kind == "object_pair":
            return self.object_pair_head(features)
        raise ValueError(f"Unknown GraphQA example kind: {kind!r}")


def label_spaces(instances):
    return {
        "object_symbol": sorted(OBJECT_SYMBOL_RELATIONS),
        "symbol_pair": collect_kb_relations(instances),
        "object_pair": collect_object_relations(instances),
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


def _object_symbol_prompt(instance, obj, symbol, query):
    return "\n".join([
        "Classify the GraphQA object-symbol predicate.",
        f"Objects: {', '.join(map(str, instance.get('objects', [])))}",
        f"Object: {obj}",
        f"Symbol: {symbol}",
        f"Query: {query}",
        "Labels: Name, ObjectType, ObjectCategory, Attribute.",
    ])


def _symbol_pair_prompt(instance, src, dst, query):
    return "\n".join([
        "Classify the GraphQA symbol-pair KG predicate.",
        f"Source symbol: {src}",
        f"Destination symbol: {dst}",
        f"Query: {query}",
    ])


def _object_pair_prompt(instance, src, dst, query):
    return "\n".join([
        "Classify the GraphQA object-object relation predicate.",
        f"Source object: {src}",
        f"Destination object: {dst}",
        f"Query: {query}",
    ])
