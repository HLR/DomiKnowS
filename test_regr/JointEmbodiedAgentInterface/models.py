"""One vision-language backbone with graph-specific EAI and VLABench heads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from test_regr.VLABenchAgentInterface.graph import labels_to_plan, plan_to_tokens
from test_regr.VLABenchAgentInterface.models import planner_prompt, resolve_vision_language_loader


DOMAINS = ("eai", "vlabench")
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


class JointQwenVLPlanner(nn.Module):
    """A single Qwen2.5-VL/LoRA policy with two compact graph label heads.

    The backbone and LoRA adapter are registered exactly once.  Domain views
    route the existing standalone program APIs into this module without
    copying or re-registering the shared parameters.
    """

    supports_batched_prefixes = False

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        *,
        eai_vocabulary: Any,
        vlabench_vocabulary: Any,
        hidden_size: int | None = None,
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.vocabularies = {
            "eai": eai_vocabulary,
            "vlabench": vlabench_vocabulary,
        }
        config = getattr(model, "config", None)
        hidden_size = hidden_size or getattr(config, "hidden_size", None) or getattr(config, "d_model", None)
        if hidden_size is None:
            raise ValueError("planner hidden size is required when the backbone config does not declare it")
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        self.label_heads = nn.ModuleDict({
            domain: nn.Linear(int(hidden_size), int(vocabulary.label_count)).to(device)
            for domain, vocabulary in self.vocabularies.items()
        })

    @classmethod
    def from_pretrained(
        cls,
        *,
        eai_vocabulary: Any,
        vlabench_vocabulary: Any,
        model_id: str = DEFAULT_MODEL_ID,
        use_lora: bool = True,
        adapter_path: str | None = None,
        load_in_4bit: bool = True,
        gradient_checkpointing: bool = True,
        local_files_only: bool = False,
    ) -> "JointQwenVLPlanner":
        model_class, processor_class = resolve_vision_language_loader()

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        kwargs: dict[str, Any] = {"torch_dtype": dtype, "local_files_only": local_files_only}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )
        model = model_class.from_pretrained(model_id, **kwargs)
        processor = processor_class.from_pretrained(model_id, local_files_only=local_files_only)
        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
        if adapter_path:
            from peft import PeftModel, prepare_model_for_kbit_training
            if load_in_4bit:
                model = prepare_model_for_kbit_training(model)
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        elif use_lora:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            if load_in_4bit:
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                bias="none",
                task_type="CAUSAL_LM",
            ))
        planner = cls(
            model,
            processor,
            eai_vocabulary=eai_vocabulary,
            vlabench_vocabulary=vlabench_vocabulary,
        )
        if adapter_path:
            head_path = Path(adapter_path) / "joint_label_heads.pt"
            if head_path.exists():
                planner.label_heads.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
        return planner

    @property
    def device(self) -> torch.device:
        return next(self.label_heads.parameters()).device

    def vocabulary(self, domain: str):
        if domain not in DOMAINS:
            raise ValueError(f"unknown planner domain {domain!r}")
        return self.vocabularies[domain]

    def for_domain(self, domain: str) -> "JointPlannerDomainView":
        return JointPlannerDomainView(self, domain)

    def _prompt(self, domain: str, context: Mapping[str, Any], prefix_labels: Sequence[int]) -> str:
        vocabulary = self.vocabulary(domain)
        prefix = " ".join(
            vocabulary.token_for_label(int(label))
            for label in prefix_labels
            if int(label) != int(vocabulary.eos_label)
        )
        if domain == "eai":
            instruction = context.get("instruction") or context.get("text") or context.get("causal_prompt_text") or ""
            goal = context.get("goal", "")
            allowed = ", ".join(vocabulary.tokens)
            prompt = (
                "Generate the exact embodied action/entity token sequence. "
                f"Use only: {allowed}.\nInstruction: {instruction}\nGoal: {goal}\nPlan:"
            )
        else:
            prompt = planner_prompt(
                context.get("instruction", ""),
                context.get("entity_table", ()),
                vocabulary,
            )
        return f"{prompt} {prefix}" if prefix else prompt

    def _chat(self, prompt: str, image_count: int) -> str:
        if not hasattr(self.processor, "apply_chat_template"):
            return prompt
        content = [{"type": "image"} for _ in range(image_count)]
        content.append({"type": "text", "text": prompt})
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _inputs(self, domain: str, context: Mapping[str, Any], prefix_labels: Sequence[int]):
        prompt = self._prompt(domain, context, prefix_labels)
        images = list(context.get("images", ())) if domain == "vlabench" else []
        opened = []
        try:
            from PIL import Image
            resolved = []
            for image in images:
                if isinstance(image, (str, Path)):
                    image = Image.open(image).convert("RGB")
                    opened.append(image)
                resolved.append(image)
            batch = self.processor(
                text=[self._chat(prompt, len(resolved))],
                images=resolved or None,
                padding=True,
                return_tensors="pt",
            )
        finally:
            for image in opened:
                image.close()
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = self.device
        return {
            key: value.to(model_device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }

    def next_label_logits(
        self,
        domain: str,
        context: Mapping[str, Any],
        prefix_labels: Sequence[int],
    ) -> torch.Tensor:
        inputs = self._inputs(domain, context, prefix_labels)
        output = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden_states = getattr(output, "hidden_states", None) or getattr(output, "decoder_hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("vision-language backbone did not return hidden states")
        hidden = hidden_states[-1][:, -1, :].float().to(self.label_heads[domain].weight.device)
        return self.label_heads[domain](hidden)[0]

    def shift_right(self, domain: str, labels: torch.Tensor) -> torch.Tensor:
        labels = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        eos = int(self.vocabulary(domain).eos_label)
        start = torch.full((labels.shape[0], 1), eos, dtype=torch.long, device=labels.device)
        return torch.cat((start, labels[:, :-1]), dim=1)

    def sequence_logits(
        self,
        domain: str,
        context: Mapping[str, Any],
        prefix_labels: torch.Tensor,
    ) -> torch.Tensor:
        prefixes = torch.as_tensor(prefix_labels, dtype=torch.long, device=self.device)
        if prefixes.ndim == 1:
            prefixes = prefixes.unsqueeze(0)
        rows = []
        for row in prefixes:
            values = row.detach().cpu().tolist()
            rows.append(torch.stack([
                self.next_label_logits(domain, context, values[: index + 1])
                for index in range(len(values))
            ]))
        return torch.stack(rows)

    def forward(self, domain: str, context: Mapping[str, Any], target_labels: torch.Tensor) -> torch.Tensor:
        logits = self.sequence_logits(domain, context, self.shift_right(domain, target_labels))
        return logits.squeeze(0) if logits.shape[0] == 1 else logits

    def supervised_loss(
        self,
        domain: str,
        *,
        context: Mapping[str, Any],
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        labels = torch.as_tensor(target_labels, dtype=torch.long, device=self.device)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        logits = self.sequence_logits(domain, context, self.shift_right(domain, labels))
        eos = int(self.vocabulary(domain).eos_label)
        keep = ((labels == eos).cumsum(dim=-1) <= 1)
        return F.cross_entropy(logits[keep], labels[keep])

    def sample_labels(
        self,
        domain: str,
        context: Mapping[str, Any],
        dfa: Any,
        *,
        max_steps: int,
        deterministic: bool = False,
    ) -> tuple[list[int], torch.Tensor]:
        vocabulary = self.vocabulary(domain)
        state = dfa.start_state
        labels: list[int] = []
        logprob = torch.zeros((), device=self.device)
        for step in range(int(max_steps)):
            logits = self.next_label_logits(domain, context, [vocabulary.eos_label, *labels])
            allowed = dfa.allowed_tokens(state, remaining_steps=max_steps - step - 1)
            if not allowed:
                raise RuntimeError(f"{domain} planner DFA has no productive transition")
            masked = torch.full_like(logits, float("-inf"))
            indices = torch.tensor(sorted(int(label) for label in allowed), device=logits.device)
            masked[indices] = logits[indices]
            distribution = torch.distributions.Categorical(logits=masked)
            label = torch.argmax(masked) if deterministic else distribution.sample()
            logprob = logprob + distribution.log_prob(label)
            labels.append(int(label))
            state = dfa.step(state, int(label))
            if state is None:
                raise RuntimeError(f"{domain} planner emitted a rejected DFA transition")
            if int(label) == int(vocabulary.eos_label) and dfa.is_accepting(state):
                break
        if not dfa.is_accepting(state):
            raise RuntimeError(f"{domain} planner did not terminate in an accepting DFA state")
        return labels, logprob

    def save_pretrained(self, path: str | Path) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(target))
        if hasattr(self.processor, "save_pretrained"):
            self.processor.save_pretrained(str(target))
        torch.save(self.label_heads.state_dict(), target / "joint_label_heads.pt")


class JointPlannerDomainView(nn.Module):
    """Non-owning adapter exposing a standalone planner API for one domain."""

    supports_batched_prefixes = False

    def __init__(self, joint: JointQwenVLPlanner, domain: str):
        super().__init__()
        if domain not in DOMAINS:
            raise ValueError(domain)
        object.__setattr__(self, "_joint", joint)
        self.domain = domain
        self.vocabulary = joint.vocabulary(domain)
        self.label_count = self.vocabulary.label_count
        self.eos_label = self.vocabulary.eos_label

    @property
    def joint(self) -> JointQwenVLPlanner:
        return object.__getattribute__(self, "_joint")

    @property
    def device(self):
        return self.joint.device

    def parameters(self, recurse: bool = True):
        return self.joint.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        return self.joint.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)

    def train(self, mode: bool = True):
        self.joint.train(mode)
        return super().train(mode)

    def forward(self, _contains, context, target_labels):
        return self.joint(self.domain, context, target_labels)

    def sequence_logits(self, context, prefix_labels):
        return self.joint.sequence_logits(self.domain, context, prefix_labels)

    def sample_labels(self, context, dfa, *, max_steps, deterministic=False):
        return self.joint.sample_labels(
            self.domain, context, dfa, max_steps=max_steps, deterministic=deterministic,
        )

    def supervised_loss(self, **kwargs):
        if self.domain == "eai":
            context = kwargs.get("context") or {
                "instruction": kwargs.get("instruction", ""),
                "goal": kwargs.get("goal", ""),
            }
            labels = kwargs.get("target_labels")
            if labels is None:
                raise ValueError("EAI supervised_loss requires target_labels")
        else:
            context = kwargs.get("context") or {
                "instruction": kwargs.get("instruction", ""),
                "images": kwargs.get("images", ()),
                "entity_table": kwargs.get("entity_table", ()),
            }
            labels = kwargs.get("target_labels")
            if labels is None:
                tokens = plan_to_tokens(
                    kwargs["target_plan"], context["entity_table"], world=kwargs.get("world"),
                )
                labels = torch.tensor(
                    [self.vocabulary.label_for_token(token) for token in tokens],
                    device=self.device,
                )
        return self.joint.supervised_loss(self.domain, context=context, target_labels=labels)

    def sample_with_logprob(self, **kwargs):
        context = kwargs.get("context") or {
            "instruction": kwargs.get("instruction", ""),
            "goal": kwargs.get("goal", ""),
            "images": kwargs.get("images", ()),
            "entity_table": kwargs.get("entity_table", ()),
        }
        labels, logprob = self.sample_labels(
            context,
            kwargs["dfa"],
            max_steps=kwargs.get("max_steps", 60),
        )
        if self.domain == "vlabench":
            return labels_to_plan(labels, self.vocabulary, world=kwargs.get("world")), logprob
        return labels, logprob

    @torch.no_grad()
    def generate_plan(self, **kwargs):
        context = kwargs.get("context") or {
            "instruction": kwargs.get("instruction", ""),
            "goal": kwargs.get("goal", ""),
            "images": kwargs.get("images", ()),
            "entity_table": kwargs.get("entity_table", ()),
        }
        labels, _ = self.sample_labels(
            context,
            kwargs["dfa"],
            max_steps=kwargs.get("max_steps", 60),
            deterministic=True,
        )
        if self.domain == "vlabench":
            return labels_to_plan(labels, self.vocabulary, world=kwargs.get("world"))
        return labels
