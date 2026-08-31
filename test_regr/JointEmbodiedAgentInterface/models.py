"""One vision-language backbone with graph-specific EAI and VLABench heads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from test_regr.VLABenchAgentInterface.graph import labels_to_plan, plan_to_tokens
from test_regr.VLABenchAgentInterface.models import (
    planner_prompt,
    prepare_kbit_model,
    resolve_vision_language_loader,
    vision_language_hidden_size,
)


DOMAINS = ("eai", "vlabench")
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


class JointQwenVLPlanner(nn.Module):
    """A single Qwen2.5-VL/LoRA policy with two compact graph label heads.

    The backbone and LoRA adapter are registered exactly once.  Domain views
    route the existing standalone program APIs into this module without
    copying or re-registering the shared parameters.
    """

    supports_batched_prefixes = True
    graph_decoder_version = 1

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        *,
        eai_vocabulary: Any,
        vlabench_vocabulary: Any,
        hidden_size: int | None = None,
        decoder_hidden_size: int = 512,
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.vocabularies = {
            "eai": eai_vocabulary,
            "vlabench": vlabench_vocabulary,
        }
        hidden_size = hidden_size or vision_language_hidden_size(model)
        if hidden_size is None:
            raise ValueError("planner hidden size is required when the backbone config does not declare it")
        if int(decoder_hidden_size) <= 0:
            raise ValueError("planner decoder hidden size must be positive")
        self.backbone_hidden_size = int(hidden_size)
        self.decoder_hidden_size = min(int(decoder_hidden_size), self.backbone_hidden_size)
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        self.context_projections = nn.ModuleDict({
            domain: nn.Linear(self.backbone_hidden_size, self.decoder_hidden_size).to(device)
            for domain in self.vocabularies
        })
        self.token_embeddings = nn.ModuleDict({
            domain: nn.Embedding(int(vocabulary.label_count), self.decoder_hidden_size).to(device)
            for domain, vocabulary in self.vocabularies.items()
        })
        self.graph_decoders = nn.ModuleDict({
            domain: nn.GRU(
                self.decoder_hidden_size,
                self.decoder_hidden_size,
                batch_first=True,
            ).to(device)
            for domain in self.vocabularies
        })
        self.label_heads = nn.ModuleDict({
            domain: nn.Linear(self.decoder_hidden_size, int(vocabulary.label_count)).to(device)
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
        decoder_hidden_size: int = 512,
    ) -> "JointQwenVLPlanner":
        model_class, processor_class = resolve_vision_language_loader()

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        kwargs: dict[str, Any] = {"dtype": dtype, "local_files_only": local_files_only}
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
        hidden_size = vision_language_hidden_size(model)
        processor = processor_class.from_pretrained(model_id, local_files_only=local_files_only)
        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
        if adapter_path:
            from peft import PeftModel
            if load_in_4bit:
                model = prepare_kbit_model(
                    model,
                    gradient_checkpointing=gradient_checkpointing,
                )
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        elif use_lora:
            from peft import LoraConfig, get_peft_model
            if load_in_4bit:
                model = prepare_kbit_model(
                    model,
                    gradient_checkpointing=gradient_checkpointing,
                )
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
            hidden_size=hidden_size,
            decoder_hidden_size=decoder_hidden_size,
        )
        if adapter_path:
            decoder_path = Path(adapter_path) / "joint_graph_decoder.pt"
            if decoder_path.exists():
                decoder_state = torch.load(decoder_path, map_location="cpu", weights_only=True)
                if int(decoder_state.get("graph_decoder_version", -1)) != planner.graph_decoder_version:
                    raise ValueError("adapter graph decoder version is incompatible")
                if int(decoder_state.get("decoder_hidden_size", -1)) != planner.decoder_hidden_size:
                    raise ValueError("adapter graph decoder hidden size is incompatible")
                planner.context_projections.load_state_dict(decoder_state["context_projections"])
                planner.token_embeddings.load_state_dict(decoder_state["token_embeddings"])
                planner.graph_decoders.load_state_dict(decoder_state["graph_decoders"])
                planner.label_heads.load_state_dict(decoder_state["label_heads"])
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

    def _prepare_inputs(self, domain: str, context: Mapping[str, Any], prefix_labels: Sequence[int]):
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
        # Keep replayable processor outputs on CPU. Stage-2 simulator
        # collection may retain many planner decisions before one optimizer
        # update; retaining CUDA inputs (or complete Qwen autograd graphs)
        # makes memory grow with episode length.
        return {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def _inputs_to_model_device(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = self.device
        return {
            key: value.to(model_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _inputs(self, domain: str, context: Mapping[str, Any], prefix_labels: Sequence[int]):
        return self._inputs_to_model_device(self._prepare_inputs(domain, context, prefix_labels))

    def _model_context(self, inputs: Mapping[str, Any]) -> torch.Tensor:
        output = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden_states = getattr(output, "hidden_states", None) or getattr(output, "decoder_hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("vision-language backbone did not return hidden states")
        return hidden_states[-1][:, -1, :].float().to(self.device)

    def encode_context(self, domain: str, context: Mapping[str, Any]) -> torch.Tensor:
        """Run Qwen once; its configured layer checkpointing bounds memory."""

        inputs = self._inputs(domain, context, ())
        return self._model_context(inputs)

    def prepare_replay_context(self, domain: str, context: Mapping[str, Any]) -> dict[str, Any]:
        """Preprocess an observation into CPU tensors for bounded-memory RL replay."""

        return self._prepare_inputs(domain, context, ())

    def encode_replay_context(
        self,
        domain: str,
        prepared_context: Mapping[str, Any],
    ) -> torch.Tensor:
        """Encode prepared inputs, moving only this decision to the model device."""

        return self._model_context(self._inputs_to_model_device(prepared_context))

    def _initial_decoder_state(self, domain: str, context_vector: torch.Tensor, batch: int) -> torch.Tensor:
        projected = torch.tanh(self.context_projections[domain](context_vector))
        if projected.shape[0] == 1 and batch != 1:
            projected = projected.expand(batch, -1)
        if projected.shape[0] != batch:
            raise ValueError("planner context batch does not match graph-token prefix batch")
        return projected.unsqueeze(0).contiguous()

    def _decode_prefixes(
        self,
        domain: str,
        context_vector: torch.Tensor,
        prefix_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefixes = torch.as_tensor(prefix_labels, dtype=torch.long, device=self.device)
        if prefixes.ndim == 1:
            prefixes = prefixes.unsqueeze(0)
        if prefixes.shape[1] == 0:
            eos = int(self.vocabulary(domain).eos_label)
            prefixes = torch.full((prefixes.shape[0], 1), eos, dtype=torch.long, device=self.device)
        hidden = self._initial_decoder_state(domain, context_vector, prefixes.shape[0])
        embedded = self.token_embeddings[domain](prefixes)
        decoded, hidden = self.graph_decoders[domain](embedded, hidden)
        return self.label_heads[domain](decoded), hidden

    def next_label_logits(
        self,
        domain: str,
        context: Mapping[str, Any],
        prefix_labels: Sequence[int],
    ) -> torch.Tensor:
        context_vector = self.encode_context(domain, context)
        logits, _hidden = self._decode_prefixes(
            domain,
            context_vector,
            torch.as_tensor(prefix_labels, dtype=torch.long, device=self.device),
        )
        return logits[0, -1]

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
        context_vector = self.encode_context(domain, context)
        logits, _hidden = self._decode_prefixes(domain, context_vector, prefix_labels)
        return logits

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
        context_vector = self.encode_context(domain, context)
        return self.sample_labels_from_context(
            domain,
            context_vector,
            dfa,
            max_steps=max_steps,
            deterministic=deterministic,
        )

    def sample_labels_from_context(
        self,
        domain: str,
        context_vector: torch.Tensor,
        dfa: Any,
        *,
        max_steps: int,
        deterministic: bool = False,
    ) -> tuple[list[int], torch.Tensor]:
        """Sample a trajectory while reusing an already encoded observation."""

        vocabulary = self.vocabulary(domain)
        state = dfa.start_state
        labels: list[int] = []
        logprob = torch.zeros((), device=self.device)
        hidden = self._initial_decoder_state(domain, context_vector, 1)
        previous = torch.tensor(
            [[int(vocabulary.eos_label)]],
            dtype=torch.long,
            device=self.device,
        )
        for step in range(int(max_steps)):
            embedded = self.token_embeddings[domain](previous)
            decoded, hidden = self.graph_decoders[domain](embedded, hidden)
            logits = self.label_heads[domain](decoded)[0, -1]
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
            previous = label.reshape(1, 1)
            state = dfa.step(state, int(label))
            if state is None:
                raise RuntimeError(f"{domain} planner emitted a rejected DFA transition")
            if int(label) == int(vocabulary.eos_label) and dfa.is_accepting(state):
                break
        if not dfa.is_accepting(state):
            raise RuntimeError(f"{domain} planner did not terminate in an accepting DFA state")
        return labels, logprob

    def labels_logprob_from_context(
        self,
        domain: str,
        context_vector: torch.Tensor,
        labels: Sequence[int],
        dfa: Any,
        *,
        max_steps: int,
    ) -> torch.Tensor:
        """Re-evaluate a sampled DFA trajectory with differentiable masked logits."""

        vocabulary = self.vocabulary(domain)
        state = dfa.start_state
        logprob = torch.zeros((), device=self.device)
        hidden = self._initial_decoder_state(domain, context_vector, 1)
        previous = torch.tensor(
            [[int(vocabulary.eos_label)]],
            dtype=torch.long,
            device=self.device,
        )
        for step, raw_label in enumerate(labels):
            if step >= int(max_steps):
                raise RuntimeError(f"{domain} replay trajectory exceeds its maximum length")
            embedded = self.token_embeddings[domain](previous)
            decoded, hidden = self.graph_decoders[domain](embedded, hidden)
            logits = self.label_heads[domain](decoded)[0, -1]
            allowed = dfa.allowed_tokens(state, remaining_steps=max_steps - step - 1)
            label_value = int(raw_label)
            if label_value not in allowed:
                raise RuntimeError(f"{domain} replay trajectory contains a rejected DFA transition")
            masked = torch.full_like(logits, float("-inf"))
            indices = torch.tensor(sorted(int(label) for label in allowed), device=logits.device)
            masked[indices] = logits[indices]
            distribution = torch.distributions.Categorical(logits=masked)
            label = torch.tensor(label_value, dtype=torch.long, device=logits.device)
            logprob = logprob + distribution.log_prob(label)
            previous = label.reshape(1, 1)
            state = dfa.step(state, label_value)
            if state is None:
                raise RuntimeError(f"{domain} replay trajectory contains a rejected DFA transition")
        if not dfa.is_accepting(state):
            raise RuntimeError(f"{domain} replay trajectory is not accepting")
        return logprob

    def replay_labels_logprob(
        self,
        domain: str,
        prepared_context: Mapping[str, Any],
        labels: Sequence[int],
        dfa: Any,
        *,
        max_steps: int,
    ) -> torch.Tensor:
        context_vector = self.encode_replay_context(domain, prepared_context)
        return self.labels_logprob_from_context(
            domain,
            context_vector,
            labels,
            dfa,
            max_steps=max_steps,
        )

    def save_pretrained(self, path: str | Path) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(target))
        if hasattr(self.processor, "save_pretrained"):
            self.processor.save_pretrained(str(target))
        torch.save(
            {
                "graph_decoder_version": self.graph_decoder_version,
                "decoder_hidden_size": self.decoder_hidden_size,
                "context_projections": self.context_projections.state_dict(),
                "token_embeddings": self.token_embeddings.state_dict(),
                "graph_decoders": self.graph_decoders.state_dict(),
                "label_heads": self.label_heads.state_dict(),
            },
            target / "joint_graph_decoder.pt",
        )


class JointPlannerDomainView(nn.Module):
    """Non-owning adapter exposing a standalone planner API for one domain."""

    supports_batched_prefixes = True

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
            self.domain,
            context,
            dfa,
            max_steps=max_steps,
            deterministic=deterministic,
        )

    def encode_context(self, context):
        return self.joint.encode_context(self.domain, context)

    def prepare_replay_context(self, context):
        return self.joint.prepare_replay_context(self.domain, context)

    def encode_replay_context(self, prepared_context):
        return self.joint.encode_replay_context(self.domain, prepared_context)

    def replay_labels_logprob(self, prepared_context, labels, dfa, *, max_steps):
        return self.joint.replay_labels_logprob(
            self.domain,
            prepared_context,
            labels,
            dfa,
            max_steps=max_steps,
        )

    def sample_labels_from_context(self, context_vector, dfa, *, max_steps, deterministic=False):
        return self.joint.sample_labels_from_context(
            self.domain,
            context_vector,
            dfa,
            max_steps=max_steps,
            deterministic=deterministic,
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
        encoded_context = kwargs.get("encoded_context")
        if encoded_context is None:
            labels, logprob = self.sample_labels(
                context,
                kwargs["dfa"],
                max_steps=kwargs.get("max_steps", 60),
            )
        else:
            labels, logprob = self.sample_labels_from_context(
                encoded_context,
                kwargs["dfa"],
                max_steps=kwargs.get("max_steps", 60),
            )
        output = (
            labels_to_plan(labels, self.vocabulary, world=kwargs.get("world"))
            if self.domain == "vlabench"
            else labels
        )
        if kwargs.get("return_labels", False):
            return output, logprob, labels
        return output, logprob

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
