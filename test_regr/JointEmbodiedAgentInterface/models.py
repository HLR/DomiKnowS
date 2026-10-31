"""One vision-language causal backbone with shared Causal Transformer decoding and domain label heads."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from domiknows.generation.dfa.predicate_monitor import require_bound_predicate_dfa
from domiknows.generation.dfa.stop_policy import (
    StopPolicy, make_progress_tracker, remaining_steps_for, should_stop_decoding,
    should_stop_on_token, stop_policy_from_legacy,
)
from test_regr.common_backbone import COMMON_VLM_MODEL_ID
from test_regr.VLABenchAgentInterface.plan_codec import labels_to_plan, plan_to_tokens
from test_regr.VLABenchAgentInterface.models import (
    planner_prompt,
    prepare_kbit_model,
    resolve_vision_language_loader,
    vision_language_hidden_size,
)

from .utils.planner_context import planner_context


DOMAINS = ("eai", "vlabench")
DEFAULT_MODEL_ID = COMMON_VLM_MODEL_ID


class FreeDecoding:
    """No grammar and no constraint: every label is allowed at every step until the end label or the step budget.

    The baseline "without any DFA" of the knowledge ablation; any sequence is accepted, and a plan is read from the
    labels as emitted (``labels_to_plan`` ignores arguments that do not follow a skill).
    """

    def __init__(self, label_count: int, eos_label: int, min_steps: int = 0):
        self.start_state = 0
        self._labels = frozenset(range(int(label_count)))
        self._eos = int(eos_label)
        self._min_steps = int(min_steps)  # the end label is withheld for this many steps

    def allowed_tokens(self, state, remaining_steps=None):
        if state is None:
            return frozenset()
        return self._labels - {self._eos} if state < self._min_steps else self._labels

    def step(self, state, label):
        return state + 1

    def is_accepting(self, state):
        return True


class EncodedContext(torch.Tensor):
    """A tensor representation of prompt context carrying original metadata."""

    @staticmethod
    def __new__(cls, tensor: torch.Tensor, context: Any = None, domain: str = ""):
        obj = torch.Tensor._make_subclass(cls, tensor)
        obj.context = context
        obj.domain = domain
        return obj


class JointQwenVLPlanner(nn.Module):
    """A single Qwen-VL/LoRA causal transformer policy with domain-specific label heads.

    The backbone and LoRA adapter are registered exactly once and act as the
    common Causal Transformer decoder for both VLABench and EAI. Domain views
    route the existing standalone program APIs into this module without
    copying or re-registering the shared parameters.
    """

    supports_batched_prefixes = True
    graph_decoder_version = 2
    causal_decoder_version = 2

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        *,
        eai_vocabulary: Any,
        vlabench_vocabulary: Any,
        hidden_size: int | None = None,
        decoder_hidden_size: int | None = None,
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
        self.backbone_hidden_size = int(hidden_size)
        self.decoder_hidden_size = int(decoder_hidden_size or self.backbone_hidden_size)
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        self.label_heads = nn.ModuleDict({
            domain: nn.Linear(self.backbone_hidden_size, int(vocabulary.label_count)).to(device)
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
        decoder_hidden_size: int | None = None,
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
            causal_head_path = Path(adapter_path) / "joint_causal_decoder.pt"
            if causal_head_path.exists():
                state = torch.load(causal_head_path, map_location="cpu", weights_only=True)
                planner.label_heads.load_state_dict(state["label_heads"])
            else:
                legacy_path = Path(adapter_path) / "joint_graph_decoder.pt"
                if legacy_path.exists():
                    state = torch.load(legacy_path, map_location="cpu", weights_only=True)
                    if "label_heads" in state:
                        planner.label_heads.load_state_dict(state["label_heads"])
                else:
                    planner.initialize_semantic_embeddings("eai")
                    planner.initialize_semantic_embeddings("vlabench")
        else:
            planner.initialize_semantic_embeddings("eai")
            planner.initialize_semantic_embeddings("vlabench")
        if adapter_path:
            from test_regr.VLABenchAgentInterface.research.checkpoint import load_adapter_metadata
            load_adapter_metadata(adapter_path,planner)
        return planner

    def initialize_semantic_embeddings(self, domain: str) -> None:
        """Semantically initialize label heads from backbone token representations."""
        if domain not in DOMAINS:
            raise ValueError(f"unknown planner domain {domain!r}")
        vocabulary = self.vocabularies.get(domain)
        if vocabulary is None:
            return

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None and callable(getattr(self.processor, "encode", None)):
            tokenizer = self.processor

        input_embeddings = None
        if hasattr(self.model, "get_input_embeddings"):
            input_embeddings = self.model.get_input_embeddings()
        elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "get_input_embeddings"):
            input_embeddings = self.model.base_model.get_input_embeddings()

        if input_embeddings is None or tokenizer is None:
            return

        embed_weight = getattr(input_embeddings, "weight", None)
        if embed_weight is None:
            return

        device = self.label_heads[domain].weight.device
        dtype = self.label_heads[domain].weight.dtype

        label_count = int(vocabulary.label_count)
        with torch.no_grad():
            new_embeddings = torch.zeros(label_count, self.backbone_hidden_size, device=device, dtype=dtype)
            for label in range(label_count):
                if hasattr(vocabulary, "token_for_label"):
                    token_str = vocabulary.token_for_label(label)
                elif hasattr(vocabulary, "tokens") and label < len(vocabulary.tokens):
                    token_str = vocabulary.tokens[label]
                else:
                    token_str = str(label)

                clean_str = token_str.replace("_", " ").strip()
                if not clean_str:
                    clean_str = token_str

                try:
                    token_ids = tokenizer.encode(clean_str, add_special_tokens=False)
                except Exception:
                    token_ids = []

                if token_ids:
                    token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=embed_weight.device)
                    if hasattr(input_embeddings, "forward"):
                        tok_vecs = input_embeddings(token_ids_tensor).to(device=device, dtype=dtype)
                    else:
                        tok_vecs = embed_weight[token_ids_tensor].to(device=device, dtype=dtype)
                    new_embeddings[label] = tok_vecs.mean(dim=0)
                else:
                    new_embeddings[label] = torch.randn(self.backbone_hidden_size, device=device, dtype=dtype) * 0.02

            std = new_embeddings.std()
            if std > 0:
                new_embeddings = new_embeddings / (std * (self.backbone_hidden_size ** 0.5)) * 0.1

            self.label_heads[domain].weight.data.copy_(new_embeddings)
            if self.label_heads[domain].bias is not None:
                nn.init.zeros_(self.label_heads[domain].bias)

    @property
    def device(self) -> torch.device:
        return next(self.label_heads.parameters()).device

    def vocabulary(self, domain: str):
        if domain not in DOMAINS:
            raise ValueError(f"unknown planner domain {domain!r}")
        return self.vocabularies[domain]

    def shared_parameters(self, recurse: bool = True):
        return self.model.parameters(recurse=recurse)

    def domain_parameters(self, domain: str, recurse: bool = True):
        if domain not in DOMAINS:
            raise ValueError(f"unknown planner domain {domain!r}")
        return self.label_heads[domain].parameters(recurse=recurse)

    def for_domain(self, domain: str) -> "JointPlannerDomainView":
        return JointPlannerDomainView(self, domain)

    def get_learnable_parameter_names(self) -> set[str]:
        """Return parameter names that are trainable in the baseline unmanaged state."""
        names = set()
        base_states = getattr(self, "_base_parameter_requires_grad", None)
        for name, param in self.named_parameters():
            if base_states is not None:
                if base_states.get(param, False):
                    names.add(name)
            elif param.requires_grad:
                names.add(name)
        names.update(getattr(self, "_checkpoint_parameter_names", ()))
        return names

    def compute_parameter_ownership_checksum(self) -> str:
        """Compute a deterministic checksum of planner parameter structure, shapes, and domain ownership."""
        import hashlib
        shared_pids = {id(p) for p in self.shared_parameters()}
        eai_pids = {id(p) for p in self.domain_parameters("eai")}
        vlabench_pids = {id(p) for p in self.domain_parameters("vlabench")}

        records = []
        for name, param in sorted(self.named_parameters(), key=lambda x: x[0]):
            pid = id(param)
            if pid in shared_pids:
                owner = "shared"
            elif pid in eai_pids:
                owner = "eai"
            elif pid in vlabench_pids:
                owner = "vlabench"
            else:
                owner = "unassigned"
            shape_str = "x".join(str(s) for s in param.shape)
            records.append(f"{name}:{owner}:{shape_str}:{param.dtype}")
        encoded = "\n".join(records).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _prompt_base(self, domain: str, context: Mapping[str, Any]) -> str:
        vocabulary = self.vocabulary(domain)
        if domain == "eai":
            instruction = str(context.get("instruction", "")).strip()
            goal = str(context.get("goal", "")).strip()
            allowed = ", ".join(vocabulary.tokens)
            return (
                "Generate the exact embodied action/entity token sequence. "
                f"Use only: {allowed}.\nInstruction: {instruction}\nGoal: {goal}\nPlan:"
            )
        else:
            from test_regr.VLABenchAgentInterface.research.prompt import observation_prompt
            return observation_prompt(context)+planner_prompt(
                context.get("instruction", ""),
                context.get("entity_table", ()),
                vocabulary,
            )

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

    def _prepare_inputs_and_boundaries(
        self,
        domain: str,
        context: Mapping[str, Any],
        active_tokens: Sequence[str] = (),
    ) -> tuple[dict[str, Any], list[int]]:
        prompt = self._prompt_base(domain, context)
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
            base_chat = self._chat(prompt, len(resolved))
            continuation = "".join(f" {t}" for t in active_tokens)
            full_text = base_chat + continuation
            batch = self.processor(
                text=[full_text],
                images=resolved or None,
                padding=True,
                return_tensors="pt",
            )
        finally:
            for image in opened:
                image.close()

        input_ids = batch.get("input_ids")
        total_len = input_ids.shape[1] if input_ids is not None else (len(active_tokens) + 1)

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None and callable(getattr(self.processor, "encode", None)):
            tokenizer = self.processor

        if not active_tokens:
            boundaries = [total_len - 1]
        else:
            if tokenizer is None or input_ids is None:
                raise ValueError("causal label alignment requires a tokenizer and processor input_ids")

            def encode(text):
                if hasattr(tokenizer, "encode"):
                    return list(tokenizer.encode(text, add_special_tokens=False))
                return list(tokenizer(text, add_special_tokens=False)["input_ids"])

            # The processor expands image placeholders. Anchor the unchanged text
            # suffix at the end of its actual input_ids instead of indexing the
            # unexpanded text stream (which can point into the image patches).
            text_ids = encode(full_text)
            expansion = total_len - len(text_ids)
            prefix_text = base_chat
            boundaries = []
            for chunk in ("", *(f" {token}" for token in active_tokens)):
                prefix_text += chunk
                prefix_ids = encode(prefix_text)
                if not prefix_ids or text_ids[:len(prefix_ids)] != prefix_ids:
                    raise ValueError("tokenization crosses a causal label boundary")
                boundaries.append(expansion + len(prefix_ids) - 1)

            first_text_index = boundaries[0] - expansion
            suffix = text_ids[first_text_index:]
            if (boundaries[0] < 0 or boundaries[-1] != total_len - 1
                    or input_ids[0, boundaries[0]:].tolist() != suffix):
                raise ValueError("processor input_ids do not preserve the causal label suffix")

        cpu_batch = {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        return cpu_batch, boundaries

    def _inputs_to_model_device(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = self.device
        return {
            key: value.to(model_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def sequence_logits(
        self,
        domain: str,
        context: Mapping[str, Any],
        prefix_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute next-token logits for every step across the causal sequence."""
        prefix = torch.as_tensor(prefix_labels, dtype=torch.long, device=self.device)
        if prefix.dim() == 1:
            prefix = prefix.unsqueeze(0)
        vocabulary = self.vocabulary(domain)
        eos_label = int(vocabulary.eos_label)
        label_count = int(vocabulary.label_count)

        rows = []
        for row in prefix:
            eos_positions = (row[1:] == eos_label).nonzero(as_tuple=False)
            padding_start = row.numel() if eos_positions.numel() == 0 else int(eos_positions[0].item()) + 1
            active_labels = row[1:padding_start].tolist()
            active_tokens = [vocabulary.token_for_label(int(l)) for l in active_labels]

            inputs, boundary_positions = self._prepare_inputs_and_boundaries(domain, context, active_tokens)
            model_inputs = self._inputs_to_model_device(inputs)
            output = self.model(**model_inputs, output_hidden_states=True, use_cache=False)
            hidden_states = getattr(output, "hidden_states", None) or getattr(output, "decoder_hidden_states", None)
            if hidden_states is None:
                raise RuntimeError("vision-language backbone did not return hidden states")
            last_hidden = hidden_states[-1][0].float().to(self.device)

            b_indices = torch.tensor(
                boundary_positions[:len(active_tokens) + 1],
                dtype=torch.long,
                device=self.device,
            )
            b_indices = b_indices.clamp(max=last_hidden.shape[0] - 1)
            active_hidden = last_hidden.index_select(0, b_indices)
            active_logits = self.label_heads[domain](active_hidden)

            logits_list = [active_logits[i] for i in range(active_logits.shape[0])]
            if len(logits_list) < row.numel():
                constant_eos = torch.full((label_count,), -8.0, device=self.device, dtype=active_logits.dtype)
                constant_eos[eos_label] = 8.0
                while len(logits_list) < row.numel():
                    logits_list.append(constant_eos)
            rows.append(torch.stack(logits_list[:row.numel()], dim=0))

        return torch.stack(rows, dim=0)

    def shift_right(self, domain: str, labels: torch.Tensor) -> torch.Tensor:
        labels = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        eos = int(self.vocabulary(domain).eos_label)
        start = torch.full((labels.shape[0], 1), eos, dtype=torch.long, device=labels.device)
        return torch.cat((start, labels[:, :-1]), dim=1)

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
        """Compute standard teacher-forced causal cross-entropy loss through the shared backbone."""
        labels = torch.as_tensor(target_labels, dtype=torch.long, device=self.device)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        shifted = self.shift_right(domain, labels)
        logits = self.sequence_logits(domain, context, shifted)
        eos = int(self.vocabulary(domain).eos_label)
        keep = ((labels == eos).cumsum(dim=-1) <= 1)
        return F.cross_entropy(logits[keep], labels[keep])

    def encode_context(self, domain: str, context: Mapping[str, Any]) -> EncodedContext:
        """Encode the initial prompt observation into an EncodedContext tensor."""
        inputs, boundaries = self._prepare_inputs_and_boundaries(domain, context, ())
        model_inputs = self._inputs_to_model_device(inputs)
        output = self.model(**model_inputs, output_hidden_states=True, use_cache=False)
        hidden_states = getattr(output, "hidden_states", None) or getattr(output, "decoder_hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("vision-language backbone did not return hidden states")
        b_idx = min(boundaries[0], hidden_states[-1].shape[1] - 1)
        prompt_hidden = hidden_states[-1][:, b_idx, :].float().to(self.device)
        return EncodedContext(prompt_hidden, context=context, domain=domain)

    def prepare_replay_context(self, domain: str, context: Mapping[str, Any]) -> dict[str, Any]:
        """Preprocess an observation into CPU tensors for bounded-memory RL replay."""
        inputs, _ = self._prepare_inputs_and_boundaries(domain, context, ())
        return inputs

    def encode_replay_context(
        self,
        domain: str,
        prepared_context: Mapping[str, Any],
    ) -> EncodedContext:
        """Encode prepared inputs on demand."""
        model_inputs = self._inputs_to_model_device(prepared_context)
        output = self.model(**model_inputs, output_hidden_states=True, use_cache=False)
        hidden_states = getattr(output, "hidden_states", None) or getattr(output, "decoder_hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("vision-language backbone did not return hidden states")
        prompt_hidden = hidden_states[-1][:, -1, :].float().to(self.device)
        return EncodedContext(prompt_hidden, context=prepared_context, domain=domain)

    def _planning_dfa(self, domain, context, dfa):
        if getattr(self, "free_decoding", False):
            vocabulary = self.vocabulary(domain)
            return FreeDecoding(vocabulary.label_count, vocabulary.eos_label,
                                min_steps=getattr(self, "free_decoding_min_steps", 0))
        if domain == "eai":
            from test_regr.EmbodiedAgentInterface.utils.constraint_compiler import bind_eai_policy
            dfa = bind_eai_policy(dfa, self.vocabulary(domain), context)
        if domain == "vlabench" and isinstance(context, Mapping) and any(
            key in context for key in ("entities", "entity_table")
        ):
            from test_regr.VLABenchAgentInterface.generation_runtime import bind_vlabench_contextual_dfa
            dfa = bind_vlabench_contextual_dfa(dfa, self.vocabulary(domain), context)
        return require_bound_predicate_dfa(dfa)

    def _sampling_forward(self, inputs, *, use_cache):
        # HF disables caches while gradient checkpointing is active, even in a
        # no_grad rollout. Temporarily suspend only checkpointing, preserving
        # module train/eval modes and restoring every flag on failure as well.
        checkpointed = []
        if use_cache:
            for module in self.model.modules():
                if module.__dict__.get("gradient_checkpointing", False):
                    checkpointed.append((module, module.gradient_checkpointing))
                    module.gradient_checkpointing = False
        try:
            return self.model(**inputs, output_hidden_states=True, use_cache=use_cache)
        finally:
            for module, original in checkpointed:
                module.gradient_checkpointing = original

    def sample_labels(
        self,
        domain: str,
        context: Mapping[str, Any],
        dfa: Any,
        *,
        max_steps: int | None = None,
        stop_policy: StopPolicy | None = None,
        deterministic: bool = False,
    ) -> tuple[list[int], torch.Tensor]:
        """Sample a valid plan autoregressively through the causal transformer with DFA constraints."""
        policy = stop_policy_from_legacy(max_new_tokens=max_steps, stop_policy=stop_policy)
        dfa = self._planning_dfa(domain, context, dfa)
        if domain=="vlabench":
            from test_regr.VLABenchAgentInterface.research.runtime import policy_context
            context=policy_context(dfa,context)
        vocabulary = self.vocabulary(domain)
        eos_label = int(vocabulary.eos_label)
        state = dfa.start_state
        labels: list[int] = []
        token_strings: list[str] = []
        logprob = torch.zeros((), device=self.device)
        use_cache = not torch.is_grad_enabled() and getattr(self, "sampling_use_cache", True)
        past = previous_ids = rope_deltas = None

        update_progress = make_progress_tracker()
        while True:
            step = len(labels)
            progress = update_progress(step_index=step, dfa_state=state, prompt_token_count=0,
                                       generated_labels=labels, accepted=dfa.is_accepting(state))
            if should_stop_decoding(policy, progress):
                break
            inputs, boundaries = self._prepare_inputs_and_boundaries(domain, context, token_strings)
            model_inputs = self._inputs_to_model_device(inputs)
            full_ids = model_inputs.get("input_ids")
            prefix_length = 0
            # Verify the actual expanded processor IDs: tokenization can change
            # at a label boundary. A changed prefix requires a fresh prefill.
            if (use_cache and past is not None and previous_ids is not None and full_ids is not None
                    and full_ids.shape[1] > previous_ids.shape[1]
                    and boundaries[-1] >= previous_ids.shape[1]
                    and torch.equal(full_ids[:, :previous_ids.shape[1]], previous_ids)):
                prefix_length = previous_ids.shape[1]
                model_inputs = dict(model_inputs)
                model_inputs["input_ids"] = full_ids[:, prefix_length:]
                for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw",
                            "second_per_grid_ts", "pixel_attention_mask", "image_sizes"):
                    model_inputs.pop(key, None)
                for key in ("position_ids", "token_type_ids", "mm_token_type_ids"):
                    if key in model_inputs:
                        model_inputs[key] = model_inputs[key][..., prefix_length:]
                if "position_ids" not in model_inputs:
                    mask = model_inputs.get("attention_mask")
                    if mask is None:
                        positions = torch.arange(prefix_length, full_ids.shape[1], device=full_ids.device)[None, :]
                    else:
                        positions = mask.long().cumsum(-1) - 1
                        positions = positions.masked_fill(mask == 0, 0)[:, prefix_length:]
                    # Qwen-VL compresses image patch positions in M-RoPE. Use
                    # this trajectory's image delta, never stale image metadata
                    # left by an earlier visual rollout when sampling EAI text.
                    if rope_deltas is not None:
                        positions = positions.unsqueeze(0).expand(3, -1, -1) + rope_deltas.unsqueeze(0)
                    model_inputs["position_ids"] = positions
                model_inputs["past_key_values"] = past
            output = self._sampling_forward(model_inputs, use_cache=use_cache)
            past = getattr(output, "past_key_values", None) if use_cache else None
            if prefix_length == 0:
                rope_deltas = None
                if past is not None and any(model_inputs.get(key) is not None for key in ("image_grid_thw", "video_grid_thw")):
                    for module in self.model.modules():
                        delta = module.__dict__.get("rope_deltas")
                        if torch.is_tensor(delta):
                            rope_deltas = delta.detach().clone()
                            break
            previous_ids = full_ids.detach().clone() if past is not None and full_ids is not None else None
            hidden_states = getattr(output, "hidden_states", None) or getattr(output, "decoder_hidden_states", None)
            if hidden_states is None:
                raise RuntimeError("vision-language backbone did not return hidden states")
            last_idx = min(boundaries[-1] - prefix_length, hidden_states[-1].shape[1] - 1)
            hidden = hidden_states[-1][0, last_idx, :].float().to(self.device)
            logits = self.label_heads[domain](hidden)

            allowed = dfa.allowed_tokens(state, remaining_steps=remaining_steps_for(policy, step))
            if not allowed:
                raise RuntimeError(
                    f"{domain} planner DFA has no productive transition at step {step}/{policy.max_steps}: "
                    f"tokens_so_far={token_strings}, state={state}, dfa_type={type(dfa).__name__}"
                )
            masked = torch.full_like(logits, float("-inf"))
            indices = torch.tensor(sorted(int(label) for label in allowed), device=logits.device)
            masked[indices] = logits[indices]
            if domain=="vlabench":
                from test_regr.VLABenchAgentInterface.research.runtime import joint_log_bias
                for choice,bias in joint_log_bias(dfa,state,allowed).items():masked[choice]+=bias
            distribution = torch.distributions.Categorical(logits=masked)
            label = torch.argmax(masked) if deterministic else distribution.sample()
            observer = getattr(self, "decision_observer", None)
            if observer is not None:
                from domiknows.generation.decision_trace import decision_record
                observer(decision_record(logits, allowed, label, vocabulary, token_strings))
            logprob = logprob + distribution.log_prob(label)
            label_int = int(label)
            labels.append(label_int)

            state = dfa.step(state, label_int)
            if state is None:
                raise RuntimeError(f"{domain} planner emitted a rejected DFA transition")
            if should_stop_on_token(policy, eos_emitted=label_int == eos_label, accepted=dfa.is_accepting(state)):
                break
            token_strings.append(vocabulary.token_for_label(label_int))

        if not dfa.is_accepting(state):
            raise RuntimeError(f"{domain} planner did not terminate in an accepting DFA state")
        return labels, logprob

    def sample_labels_from_context(
        self,
        domain: str,
        context_vector: Any,
        dfa: Any,
        *,
        max_steps: int | None = None,
        stop_policy: StopPolicy | None = None,
        deterministic: bool = False,
    ) -> tuple[list[int], torch.Tensor]:
        """Sample a plan from an EncodedContext or context dictionary."""
        context = getattr(context_vector, "context", None)
        if context is None:
            context = context_vector if isinstance(context_vector, Mapping) else {}
        return self.sample_labels(
            domain,
            context,
            dfa,
            max_steps=max_steps,
            stop_policy=stop_policy,
            deterministic=deterministic,
        )

    def replay_labels_logprob(
        self,
        domain: str,
        prepared_context: Mapping[str, Any],
        labels: Sequence[int],
        dfa: Any,
        *,
        max_steps: int | None = None,
        stop_policy: StopPolicy | None = None,
    ) -> torch.Tensor:
        """Re-evaluate a sampled DFA trajectory with differentiable masked logits in a single forward pass."""
        policy = stop_policy_from_legacy(max_new_tokens=max_steps, stop_policy=stop_policy)
        dfa = self._planning_dfa(domain, prepared_context, dfa)
        label_list = [int(x) for x in labels]
        if not label_list:
            if not dfa.is_accepting(dfa.start_state):
                raise RuntimeError(f"{domain} replay trajectory is not accepting")
            return torch.zeros((), device=self.device, requires_grad=True)
        if policy.max_steps is not None and len(label_list) > policy.max_steps:
            raise RuntimeError(f"{domain} replay trajectory exceeds its maximum length")

        vocabulary = self.vocabulary(domain)
        eos_label = int(vocabulary.eos_label)
        active_tokens = [vocabulary.token_for_label(l) for l in label_list]

        inputs, boundary_positions = self._prepare_inputs_and_boundaries(domain, prepared_context, active_tokens)
        model_inputs = self._inputs_to_model_device(inputs)
        output = self.model(**model_inputs, output_hidden_states=True, use_cache=False)
        hidden_states = getattr(output, "hidden_states", None) or getattr(output, "decoder_hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("vision-language backbone did not return hidden states")
        last_hidden = hidden_states[-1][0].float().to(self.device)

        b_indices = torch.tensor(boundary_positions[:len(label_list)], dtype=torch.long, device=self.device)
        b_indices = b_indices.clamp(max=last_hidden.shape[0] - 1)
        step_hidden = last_hidden.index_select(0, b_indices)
        all_logits = self.label_heads[domain](step_hidden)

        state = dfa.start_state
        logprob = torch.zeros((), device=self.device)
        replay_policy = replace(policy, timeout_seconds=None, external_stop_fn=None)
        update_progress = make_progress_tracker()
        for step, label_value in enumerate(label_list):
            progress = update_progress(step_index=step, dfa_state=state, prompt_token_count=0,
                                       generated_labels=label_list[:step], accepted=dfa.is_accepting(state))
            if should_stop_decoding(replay_policy, progress):
                raise RuntimeError(f"{domain} replay trajectory exceeds its stop-policy budget")
            logits = all_logits[step]
            allowed = dfa.allowed_tokens(state, remaining_steps=remaining_steps_for(policy, step))
            if label_value not in allowed:
                raise RuntimeError(f"{domain} replay trajectory contains a rejected DFA transition")
            masked = torch.full_like(logits, float("-inf"))
            indices = torch.tensor(sorted(int(lbl) for lbl in allowed), device=logits.device)
            masked[indices] = logits[indices]
            dist = torch.distributions.Categorical(logits=masked)
            if domain=="vlabench":
                from test_regr.VLABenchAgentInterface.research.runtime import joint_log_bias
                for choice,bias in joint_log_bias(dfa,state,allowed).items():masked[choice]+=bias
                dist = torch.distributions.Categorical(logits=masked)
            target = torch.tensor(label_value, dtype=torch.long, device=logits.device)
            logprob = logprob + dist.log_prob(target)
            state = dfa.step(state, label_value)
            if state is None:
                raise RuntimeError(f"{domain} replay trajectory contains a rejected DFA transition")
            if step + 1 < len(label_list) and should_stop_on_token(
                policy, eos_emitted=label_value == eos_label, accepted=dfa.is_accepting(state),
            ):
                raise RuntimeError(f"{domain} replay trajectory continues after stop-policy termination")

        if not dfa.is_accepting(state):
            raise RuntimeError(f"{domain} replay trajectory is not accepting")

        return logprob

    def labels_logprob_from_context(
        self,
        domain: str,
        context_vector: Any,
        labels: Sequence[int],
        dfa: Any,
        *,
        max_steps: int | None = None,
        stop_policy: StopPolicy | None = None,
    ) -> torch.Tensor:
        context = getattr(context_vector, "context", None)
        if context is None:
            context = context_vector if isinstance(context_vector, Mapping) else {}
        return self.replay_labels_logprob(
            domain,
            context,
            labels,
            dfa,
            max_steps=max_steps,
            stop_policy=stop_policy,
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
                "causal_decoder_version": self.causal_decoder_version,
                "backbone_hidden_size": self.backbone_hidden_size,
                "label_heads": self.label_heads.state_dict(),
            },
            target / "joint_causal_decoder.pt",
        )
        from test_regr.VLABenchAgentInterface.research.checkpoint import save_adapter_metadata
        save_adapter_metadata(target,self)


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

    def domain_parameters(self, recurse: bool = True):
        return self.joint.domain_parameters(self.domain, recurse=recurse)

    def shared_parameters(self, recurse: bool = True):
        return self.joint.shared_parameters(recurse=recurse)

    def parameters(self, recurse: bool = True):
        from itertools import chain
        return chain(self.shared_parameters(recurse=recurse), self.domain_parameters(recurse=recurse))

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        from itertools import chain
        memo = set()
        for name, param in chain(
            self.joint.model.named_parameters(prefix=f"{prefix}model." if prefix else "model.", recurse=recurse),
            self.joint.label_heads[self.domain].named_parameters(prefix=f"{prefix}label_heads.{self.domain}." if prefix else f"label_heads.{self.domain}.", recurse=recurse),
        ):
            if remove_duplicate and param in memo:
                continue
            memo.add(param)
            yield name, param

    def train(self, mode: bool = True):
        self.joint.train(mode)
        return super().train(mode)

    def forward(self, _contains, context, target_labels):
        logits = self.joint(self.domain, context, target_labels)
        if getattr(self.joint, "capture_logits", False):
            self.joint.last_logits = logits  # the program reuses this forward for the DFA losses
        return logits

    def sequence_logits(self, context, prefix_labels):
        return self.joint.sequence_logits(self.domain, context, prefix_labels)

    def sample_labels(self, context, dfa, *, max_steps=None, deterministic=False, stop_policy=None):
        return self.joint.sample_labels(
            self.domain,
            context,
            dfa,
            max_steps=max_steps,
            stop_policy=stop_policy,
            deterministic=deterministic,
        )

    def encode_context(self, context):
        return self.joint.encode_context(self.domain, context)

    def prepare_replay_context(self, context):
        return self.joint.prepare_replay_context(self.domain, context)

    def encode_replay_context(self, prepared_context):
        return self.joint.encode_replay_context(self.domain, prepared_context)

    def replay_labels_logprob(self, prepared_context, labels, dfa, *, max_steps=None, stop_policy=None):
        return self.joint.replay_labels_logprob(
            self.domain,
            prepared_context,
            labels,
            dfa,
            max_steps=max_steps,
            stop_policy=stop_policy,
        )

    def sample_labels_from_context(self, context_vector, dfa, *, max_steps=None, deterministic=False, stop_policy=None):
        return self.joint.sample_labels_from_context(
            self.domain,
            context_vector,
            dfa,
            max_steps=max_steps,
            stop_policy=stop_policy,
            deterministic=deterministic,
        )

    def supervised_loss(self, **kwargs):
        if self.domain == "eai":
            context = planner_context(self.domain, kwargs, context=kwargs.get("context"))
            labels = kwargs.get("target_labels")
            if labels is None:
                raise ValueError("EAI supervised_loss requires target_labels")
        else:
            context = planner_context(self.domain, kwargs, context=kwargs.get("context"))
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
        context = planner_context(self.domain, kwargs, context=kwargs.get("context"))
        dfa = kwargs.get("dfa")
        if self.domain == "eai" and dfa is not None and not getattr(self.joint, "free_decoding", False):
            from test_regr.EmbodiedAgentInterface.utils.constraint_compiler import bind_eai_policy
            dfa = bind_eai_policy(dfa, self.vocabulary, context)
        if self.domain == "vlabench" and dfa is not None and not getattr(self.joint, "free_decoding", False):
            from test_regr.VLABenchAgentInterface.generation_runtime import bind_vlabench_contextual_dfa
            dfa = bind_vlabench_contextual_dfa(dfa, self.vocabulary, context)
        encoded_context = kwargs.get("encoded_context")
        if encoded_context is None:
            labels, logprob = self.sample_labels(
                context,
                dfa,
                max_steps=kwargs.get("max_steps", None if kwargs.get("stop_policy") is not None else 60),
                stop_policy=kwargs.get("stop_policy"),
            )
        else:
            labels, logprob = self.sample_labels_from_context(
                encoded_context,
                dfa,
                max_steps=kwargs.get("max_steps", None if kwargs.get("stop_policy") is not None else 60),
                stop_policy=kwargs.get("stop_policy"),
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
        context = planner_context(self.domain, kwargs, context=kwargs.get("context"))
        dfa = kwargs.get("dfa")
        if self.domain == "eai" and dfa is not None and not getattr(self.joint, "free_decoding", False):
            from test_regr.EmbodiedAgentInterface.utils.constraint_compiler import bind_eai_policy
            dfa = bind_eai_policy(dfa, self.vocabulary, context)
        if self.domain == "vlabench" and dfa is not None and not getattr(self.joint, "free_decoding", False):
            from test_regr.VLABenchAgentInterface.generation_runtime import bind_vlabench_contextual_dfa
            dfa = bind_vlabench_contextual_dfa(dfa, self.vocabulary, context)
        labels, _ = self.sample_labels(
            context,
            dfa,
            max_steps=kwargs.get("max_steps", None if kwargs.get("stop_policy") is not None else 60),
            stop_policy=kwargs.get("stop_policy"),
            deterministic=True,
        )
        if self.domain == "vlabench":
            return labels_to_plan(labels, self.vocabulary, world=kwargs.get("world"))
        return labels
