"""Compact graph-token planner and stochastic multi-view controller."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .plan_codec import PlanVocabulary, labels_to_plan, plan_to_tokens
    from .generation_runtime import bind_vlabench_contextual_dfa
except ImportError:
    from plan_codec import PlanVocabulary, labels_to_plan, plan_to_tokens
    from generation_runtime import bind_vlabench_contextual_dfa


from domiknows.generation.dfa.predicate_monitor import require_bound_predicate_dfa
from domiknows.generation.dfa.stop_policy import (
    StopPolicy, make_progress_tracker, remaining_steps_for, should_stop_decoding,
    should_stop_on_token, stop_policy_from_legacy,
)
from test_regr.common_backbone import COMMON_VLM_MODEL_ID


PLANNER_MODEL_ID = COMMON_VLM_MODEL_ID
VISION_MODEL_ID = "google/siglip-base-patch16-224"


def sanitize_special_token_ids(config: Any) -> Any:
    """Clear BOS/EOS ids outside each nested config vocabulary."""
    pending = [config]
    seen = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        is_mapping = isinstance(current, Mapping)
        vocab_size = current.get("vocab_size") if is_mapping else getattr(current, "vocab_size", None)
        model_type = current.get("model_type") if is_mapping else getattr(current, "model_type", None)
        # Some SigLIP checkpoints omit the text vocabulary size from the
        # nested raw dictionary. Transformers then supplies SiglipTextConfig's
        # out-of-vocabulary BOS/EOS defaults unless the fields are explicit.
        if is_mapping and model_type in {"siglip", "siglip_text_model"}:
            current.setdefault("bos_token_id", None)
            current.setdefault("eos_token_id", None)
        try:
            vocab_size = int(vocab_size) if vocab_size is not None else None
        except (TypeError, ValueError):
            vocab_size = None
        if vocab_size is not None:
            for name in ("bos_token_id", "eos_token_id"):
                value = current.get(name) if is_mapping else getattr(current, name, None)
                try:
                    invalid = value is None or not 0 <= int(value) < vocab_size
                except (TypeError, ValueError):
                    invalid = True
                if invalid:
                    if is_mapping:
                        current[name] = None
                    else:
                        setattr(current, name, None)
        for name in ("text_config", "vision_config", "audio_config"):
            nested = current.get(name) if is_mapping else getattr(current, name, None)
            if nested is not None:
                pending.append(nested)
    return config


def clear_siglip_special_token_ids(config: Any) -> Any:
    """Restore SigLIP BOS/EOS ids to None after constructor validation."""
    pending = [config]
    seen = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        is_mapping = isinstance(current, Mapping)
        get_value = current.get if is_mapping else lambda name: getattr(current, name, None)
        model_type = get_value("model_type")
        if model_type in {"siglip", "siglip_text_model"}:
            if is_mapping:
                current["bos_token_id"] = None
                current["eos_token_id"] = None
            else:
                setattr(current, "bos_token_id", None)
                setattr(current, "eos_token_id", None)
        for name in ("text_config", "vision_config", "audio_config"):
            nested = get_value(name)
            if nested is not None:
                pending.append(nested)
    return config


def special_token_id_summary(config: Any) -> str:
    """Return a compact, nested BOS/EOS snapshot for model-load diagnostics."""
    pending = [("root", config)]
    seen = set()
    entries = []
    while pending:
        path, current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        is_mapping = isinstance(current, Mapping)
        get_value = current.get if is_mapping else lambda name: getattr(current, name, None)
        vocab_size = get_value("vocab_size")
        bos_token_id = get_value("bos_token_id")
        eos_token_id = get_value("eos_token_id")
        model_type = get_value("model_type")
        if any(value is not None for value in (vocab_size, bos_token_id, eos_token_id, model_type)):
            entries.append(
                f"{path}:class={type(current).__name__} model_type={model_type!r} "
                f"vocab={vocab_size!r} bos={bos_token_id!r} eos={eos_token_id!r}"
            )
        for name in ("text_config", "vision_config", "audio_config"):
            nested = get_value(name)
            if nested is not None:
                pending.append((f"{path}.{name}", nested))
    return "; ".join(entries) or "none"


def load_sanitized_auto_config(transformers: Any, model_id: str, *, local_files_only: bool) -> Any:
    """Build an AutoConfig from raw JSON so invalid nested token ids never validate."""

    auto_config = getattr(transformers, "AutoConfig", None)
    # Transformers exposes this class under both spellings across releases.
    # Prefer the canonical v5 name so the raw-config path remains active on
    # servers that no longer expose the legacy alias.
    pretrained_config = getattr(transformers, "PreTrainedConfig", None)
    if pretrained_config is None:
        pretrained_config = getattr(transformers, "PretrainedConfig", None)
    get_config_dict = getattr(pretrained_config, "get_config_dict", None)
    for_model = getattr(auto_config, "for_model", None)
    if callable(get_config_dict) and callable(for_model):
        raw_result = get_config_dict(model_id, local_files_only=local_files_only)
        config_dict = raw_result[0] if isinstance(raw_result, tuple) else raw_result
        if isinstance(config_dict, Mapping) and config_dict.get("model_type"):
            config_dict = dict(config_dict)
            model_type = config_dict.pop("model_type")
            if model_type == "siglip":
                text_config = config_dict.get("text_config")
                if isinstance(text_config, Mapping):
                    text_config.setdefault("bos_token_id", None)
                    text_config.setdefault("eos_token_id", None)
            sanitize_special_token_ids(config_dict)
            constructor_config_dict = config_dict
            if model_type == "siglip":
                constructor_config_dict = dict(config_dict)
                text_config = constructor_config_dict.get("text_config")
                if isinstance(text_config, Mapping):
                    text_config = dict(text_config)
                    vocab_size = text_config.get("vocab_size")
                    for name in ("bos_token_id", "eos_token_id"):
                        value = text_config.get(name)
                        if (
                            value is None
                            or not isinstance(value, int)
                            or value < 0
                            or (vocab_size is not None and value >= int(vocab_size))
                        ):
                            # Transformers 5.16 treats None as an omitted
                            # dataclass field and reinstates SigLIP defaults.
                            text_config[name] = 0
                    constructor_config_dict["text_config"] = text_config
            # Construct directly from the cleaned raw dictionary. Calling
            # AutoConfig.from_pretrained with the nested dictionary first
            # constructs the original config and only applies nested kwargs
            # afterward, so SiglipTextConfig's out-of-vocabulary defaults
            # still trigger validation warnings before sanitization runs.
            if model_type == "siglip":
                print(
                    f"[vlabench-model] raw SigLIP config before AutoConfig.for_model "
                    f"{special_token_id_summary(config_dict)}",
                    file=sys.stderr,
                    flush=True,
                )
            if model_type == "siglip":
                hf_logging = getattr(transformers, "logging", None)
                get_verbosity = getattr(hf_logging, "get_verbosity", None)
                set_verbosity = getattr(hf_logging, "set_verbosity", None)
                set_verbosity_error = getattr(hf_logging, "set_verbosity_error", None)
                previous_verbosity = get_verbosity() if callable(get_verbosity) else None
                if callable(set_verbosity_error):
                    set_verbosity_error()
                try:
                    config = for_model(model_type, **constructor_config_dict)
                finally:
                    if previous_verbosity is not None and callable(set_verbosity):
                        set_verbosity(previous_verbosity)
            else:
                config = for_model(model_type, **constructor_config_dict)
            if model_type == "siglip":
                print(
                    f"[vlabench-model] SigLIP config after AutoConfig.for_model "
                    f"{special_token_id_summary(config)}",
                    file=sys.stderr,
                    flush=True,
                )
            return clear_siglip_special_token_ids(sanitize_special_token_ids(config))
    if auto_config is None:
        return None
    return clear_siglip_special_token_ids(
        sanitize_special_token_ids(auto_config.from_pretrained(model_id, local_files_only=local_files_only))
    )


def resolve_vision_language_loader():
    """Return a Transformers VLM loader across its renamed auto-model APIs."""

    import transformers

    processor_class = getattr(transformers, "AutoProcessor", None)
    if processor_class is None:
        raise ImportError("the installed transformers package does not export AutoProcessor")
    failures = []
    for class_name in (
        "AutoModelForImageTextToText",
        "Qwen3VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "AutoModelForVision2Seq",
    ):
        try:
            model_class = getattr(transformers, class_name)
        except (AttributeError, ImportError, ModuleNotFoundError, RuntimeError) as error:
            failures.append(f"{class_name}: {error}")
            continue
        return model_class, processor_class
    version = getattr(transformers, "__version__", "unknown")
    details = "; ".join(failures) or "no compatible class is exported"
    raise ImportError(
        "No compatible Qwen vision-language model loader is available in transformers "
        f"{version}. Tried AutoModelForImageTextToText, "
        "Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, "
        "and AutoModelForVision2Seq. "
        f"Details: {details}"
    )


def prepare_kbit_model(model: nn.Module, *, gradient_checkpointing: bool) -> nn.Module:
    """Prepare a quantized PEFT backbone without changing checkpointing mode."""

    from peft import prepare_model_for_kbit_training

    kwargs: dict[str, Any] = {
        "use_gradient_checkpointing": bool(gradient_checkpointing),
    }
    if gradient_checkpointing:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return prepare_model_for_kbit_training(model, **kwargs)


def vision_language_hidden_size(model: nn.Module) -> int | None:
    """Find the language width through multimodal configs and PEFT wrappers."""

    pending = [model]
    seen = set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        config = getattr(candidate, "config", None)
        configs = [config]
        if config is not None:
            configs.extend(
                getattr(config, name, None)
                for name in ("text_config", "language_config", "decoder", "decoder_config")
            )
        for nested in configs:
            size = getattr(nested, "hidden_size", None) or getattr(nested, "d_model", None)
            if size is not None:
                return int(size)
        for name in ("base_model", "model"):
            wrapped = getattr(candidate, name, None)
            if isinstance(wrapped, nn.Module):
                pending.append(wrapped)
        get_base_model = getattr(candidate, "get_base_model", None)
        if callable(get_base_model):
            try:
                pending.append(get_base_model())
            except (AttributeError, TypeError):
                pass
    return None


def planner_prompt(instruction: str, entity_table: Sequence[str], vocabulary: PlanVocabulary) -> str:
    from .research.config import default_perception_options
    if default_perception_options():entity_table=tuple(f"numbered object {i}" for i in range(len(entity_table)))
    entities = "\n".join(f"{index}: {name}" for index, name in enumerate(entity_table)) or "No objects."
    skills = ", ".join(vocabulary.skills)
    directive = ""
    buttons = [
        (idx, str(name))
        for idx, name in enumerate(entity_table)
        if "button" in str(name).lower()
    ]
    if buttons:
        # Entity list order is not an observation of object/button association.
        pairs_str = "; ".join(f"{name} (id: {idx})" for idx, name in buttons)
        directive = (
            f" NOTE: Multiple-choice question with answer buttons ({pairs_str}). "
            "Determine the correct object that satisfies the instruction, then output: "
            "press target_entity_name=<button_id> for the corresponding button."
        )
    return (
        "Predict a compact VLABench plan using only graph labels. "
        "A skill label is followed by its graph-declared argument and object pointer labels."
        f"{directive}\n"
        f"Skills: {skills}\nObjects:\n{entities}\nInstruction: {instruction}\nPlan labels:"
    )


class QwenVLPlanner(nn.Module):
    """Qwen vision-language context encoder plus compact graph-token decoder.

    Qwen processes each observation once. A small recurrent decoder then
    teacher-forces or autoregressively samples the complete graph trajectory;
    target length therefore no longer multiplies expensive VLM passes.
    """

    supports_batched_prefixes = True
    graph_decoder_version = 1

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        vocabulary: PlanVocabulary,
        hidden_size: int | None = None,
        decoder_hidden_size: int = 512,
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.vocabulary = vocabulary
        hidden_size = hidden_size or vision_language_hidden_size(model)
        if hidden_size is None:
            raise ValueError("planner hidden size is required when the model config does not declare it")
        if int(decoder_hidden_size) <= 0:
            raise ValueError("planner decoder hidden size must be positive")
        self.backbone_hidden_size = int(hidden_size)
        self.decoder_hidden_size = min(int(decoder_hidden_size), self.backbone_hidden_size)
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
        self.context_projection = nn.Linear(
            self.backbone_hidden_size, self.decoder_hidden_size
        ).to(model_device)
        self.token_embedding = nn.Embedding(
            vocabulary.label_count, self.decoder_hidden_size
        ).to(model_device)
        self.graph_decoder = nn.GRU(
            self.decoder_hidden_size,
            self.decoder_hidden_size,
            batch_first=True,
        ).to(model_device)
        # Keep the historical public name used by standalone callers/tests.
        self.output = nn.Linear(
            self.decoder_hidden_size, vocabulary.label_count
        ).to(model_device)
        self.label_count = vocabulary.label_count
        self.eos_label = vocabulary.eos_label

    @classmethod
    def from_pretrained(
        cls,
        vocabulary: PlanVocabulary,
        model_id: str = PLANNER_MODEL_ID,
        *,
        use_lora: bool = True,
        adapter_path: str | None = None,
        load_in_4bit: bool = False,
        gradient_checkpointing: bool = True,
        local_files_only: bool = False,
        decoder_hidden_size: int = 512,
    ) -> "QwenVLPlanner":
        model_class, processor_class = resolve_vision_language_loader()
        import transformers

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        kwargs: dict[str, Any] = {"dtype": dtype, "local_files_only": local_files_only}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4",
            )
        auto_config = getattr(transformers, "AutoConfig", None)
        if auto_config is not None:
            config = load_sanitized_auto_config(
                transformers, model_id, local_files_only=local_files_only
            )
            kwargs["config"] = config
            # Prevent from_pretrained from loading a stale generation_config
            # with BOS/EOS ids from an unrelated tokenizer vocabulary.
            generation_config_class = getattr(transformers, "GenerationConfig", None)
            if generation_config_class is not None:
                try:
                    generation_config = generation_config_class.from_model_config(config)
                    generation_config.bos_token_id = getattr(config, "bos_token_id", None)
                    generation_config.eos_token_id = getattr(config, "eos_token_id", None)
                    kwargs["generation_config"] = generation_config
                except (AttributeError, TypeError, ValueError):
                    pass
        model = model_class.from_pretrained(model_id, **kwargs)
        hidden_size = vision_language_hidden_size(model)
        # Some cached Qwen/CLIP tokenizer configs carry BOS/EOS ids from a
        # different vocabulary.  Pass explicit null overrides so the warning
        # is suppressed during tokenizer construction, not after it.
        processor = processor_class.from_pretrained(
            model_id,
            local_files_only=local_files_only,
            bos_token_id=None,
            eos_token_id=None,
        )
        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
        if adapter_path is not None:
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
                r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none", task_type="CAUSAL_LM",
            ))
        planner = cls(
            model,
            processor,
            vocabulary,
            hidden_size=hidden_size,
            decoder_hidden_size=decoder_hidden_size,
        )
        decoder_path = Path(adapter_path) / "graph_decoder.pt" if adapter_path else None
        if decoder_path is not None:
            if not decoder_path.exists():
                raise ValueError(
                    "standalone adapter predates graph-decoder version 1 and cannot be resumed"
                )
            state = torch.load(decoder_path, map_location="cpu", weights_only=True)
            if int(state.get("graph_decoder_version", -1)) != planner.graph_decoder_version:
                raise ValueError("adapter graph decoder version is incompatible")
            if int(state.get("decoder_hidden_size", -1)) != planner.decoder_hidden_size:
                raise ValueError("adapter graph decoder hidden size is incompatible")
            planner.context_projection.load_state_dict(state["context_projection"])
            planner.token_embedding.load_state_dict(state["token_embedding"])
            planner.graph_decoder.load_state_dict(state["graph_decoder"])
            planner.output.load_state_dict(state["output"])
        return planner

    @property
    def device(self) -> torch.device:
        return next(self.output.parameters()).device

    def _chat(self, prompt: str, image_count: int) -> str:
        content = [{"type": "image"} for _ in range(image_count)]
        content.append({"type": "text", "text": prompt})
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
        )

    def _inputs(self, context: Mapping[str, Any], prefix_labels: Sequence[int]):
        prefix = " ".join(
            self.vocabulary.token_for_label(label)
            for label in prefix_labels
            if int(label) != self.eos_label
        )
        prompt = planner_prompt(context["instruction"], context.get("entity_table", ()), self.vocabulary)
        from .research.prompt import observation_prompt
        prompt += observation_prompt(context)
        if prefix:
            prompt += " " + prefix
        images = list(context.get("images", ()))
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
                text=[self._chat(prompt, len(resolved))], images=resolved or None,
                padding=True, return_tensors="pt",
            )
        finally:
            for image in opened:
                image.close()
        model_device = next(self.model.parameters()).device
        return {key: value.to(model_device) if hasattr(value, "to") else value for key, value in batch.items()}

    def _model_context(self, inputs: Mapping[str, Any]) -> torch.Tensor:
        output = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden_states = getattr(output, "hidden_states", None) or getattr(output, "decoder_hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("vision-language backbone did not return hidden states")
        return hidden_states[-1][:, -1, :].float().to(self.device)

    def encode_context(self, context: Mapping[str, Any]) -> torch.Tensor:
        """Run the expensive vision-language backbone exactly once."""

        return self._model_context(self._inputs(context, ()))

    def prepare_replay_context(self, context: Mapping[str, Any]) -> dict[str, Any]:
        """Preprocess a rollout observation into replayable CPU tensors."""

        inputs = self._inputs(context, ())
        return {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }

    def encode_replay_context(self, prepared_context: Mapping[str, Any]) -> torch.Tensor:
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = self.device
        inputs = {
            key: value.to(model_device) if hasattr(value, "to") else value
            for key, value in prepared_context.items()
        }
        return self._model_context(inputs)

    def _initial_decoder_state(self, context_vector: torch.Tensor, batch: int) -> torch.Tensor:
        projected = torch.tanh(self.context_projection(context_vector))
        if projected.shape[0] == 1 and batch != 1:
            projected = projected.expand(batch, -1)
        if projected.shape[0] != batch:
            raise ValueError("planner context batch does not match graph-token prefix batch")
        return projected.unsqueeze(0).contiguous()

    def _decode_prefixes(
        self,
        context_vector: torch.Tensor,
        prefix_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefixes = torch.as_tensor(prefix_labels, dtype=torch.long, device=self.device)
        if prefixes.ndim == 1:
            prefixes = prefixes.unsqueeze(0)
        if prefixes.shape[1] == 0:
            prefixes = torch.full(
                (prefixes.shape[0], 1),
                int(self.eos_label),
                dtype=torch.long,
                device=self.device,
            )
        hidden = self._initial_decoder_state(context_vector, prefixes.shape[0])
        decoded, hidden = self.graph_decoder(self.token_embedding(prefixes), hidden)
        return self.output(decoded), hidden

    def next_label_logits(
        self,
        context: Mapping[str, Any],
        prefix_labels: Sequence[int],
    ) -> torch.Tensor:
        context_vector = self.encode_context(context)
        logits, _hidden = self._decode_prefixes(
            context_vector,
            torch.as_tensor(prefix_labels, dtype=torch.long, device=self.device),
        )
        return logits[0, -1]

    def sequence_logits(self, context: Mapping[str, Any], prefix_labels: torch.Tensor) -> torch.Tensor:
        context_vector = self.encode_context(context)
        logits, _hidden = self._decode_prefixes(context_vector, prefix_labels)
        return logits

    def _shift_right(self, labels: torch.Tensor) -> torch.Tensor:
        labels = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        start = torch.full((labels.shape[0], 1), self.eos_label, dtype=torch.long, device=self.device)
        return torch.cat((start, labels[:, :-1]), dim=1)

    def forward(self, _contains, context: Mapping[str, Any], target_labels: torch.Tensor) -> torch.Tensor:
        logits = self.sequence_logits(context, self._shift_right(target_labels))
        return logits.squeeze(0) if logits.shape[0] == 1 else logits

    def supervised_loss(self, *, instruction, images, entity_table, target_plan, world=None, **_kwargs) -> torch.Tensor:
        tokens = plan_to_tokens(target_plan, entity_table, world=world)
        labels = torch.tensor([self.vocabulary.label_for_token(token) for token in tokens], device=self.device)
        logits = self.sequence_logits(
            {**{k: v for k, v in _kwargs.items() if k in ("planning_facts", "task_target_container", "task_target_entity", "task", "task_name", "task_skill_sequence")}, "instruction": instruction, "images": images, "entity_table": entity_table},
            self._shift_right(labels),
        )[0]
        return F.cross_entropy(logits, labels)

    def sample_labels(self, context: Mapping[str, Any], dfa, *, max_steps: int | None = None, deterministic: bool = False, stop_policy: StopPolicy | None = None):
        from .research.runtime import policy_context
        context=policy_context(dfa,context)
        policy = stop_policy_from_legacy(max_new_tokens=max_steps, stop_policy=stop_policy)
        dfa = bind_vlabench_contextual_dfa(dfa, self.vocabulary, context)
        context_vector = self.encode_context(context)
        return self.sample_labels_from_context(
            context_vector,
            dfa,
            stop_policy=policy,
            deterministic=deterministic,
        )

    def sample_labels_from_context(
        self,
        context_vector: torch.Tensor,
        dfa,
        *,
        max_steps: int | None = None,
        stop_policy: StopPolicy | None = None,
        deterministic: bool = False,
    ):
        policy = stop_policy_from_legacy(max_new_tokens=max_steps, stop_policy=stop_policy)
        require_bound_predicate_dfa(dfa)
        state = dfa.start_state
        labels = []
        logprob = torch.zeros((), device=self.device)
        hidden = self._initial_decoder_state(context_vector, 1)
        previous = torch.tensor(
            [[int(self.eos_label)]], dtype=torch.long, device=self.device
        )
        update_progress = make_progress_tracker()
        while True:
            step = len(labels)
            progress = update_progress(step_index=step, dfa_state=state, prompt_token_count=0,
                                       generated_labels=labels, accepted=dfa.is_accepting(state))
            if should_stop_decoding(policy, progress):
                break
            decoded, hidden = self.graph_decoder(self.token_embedding(previous), hidden)
            logits = self.output(decoded)[0, -1]
            allowed = dfa.allowed_tokens(state, remaining_steps=remaining_steps_for(policy, step))
            mask = torch.full_like(logits, float("-inf"))
            if not allowed:
                raise RuntimeError("planner DFA has no productive transition")
            indices = torch.tensor(sorted(int(label) for label in allowed), device=logits.device)
            mask[indices] = logits[indices]
            from .research.runtime import joint_log_bias
            for choice,bias in joint_log_bias(dfa,state,allowed).items():mask[choice]+=bias
            distribution = torch.distributions.Categorical(logits=mask)
            label = torch.argmax(mask) if deterministic else distribution.sample()
            observer = getattr(self, "decision_observer", None)
            if observer is not None:
                from domiknows.generation.decision_trace import decision_record
                observer(decision_record(logits, allowed, label, self.vocabulary, [self.vocabulary.token_for_label(i) for i in labels]))
            logprob = logprob + distribution.log_prob(label)
            labels.append(int(label))
            previous = label.reshape(1, 1)
            state = dfa.step(state, int(label))
            if state is None:
                raise RuntimeError("planner emitted a rejected DFA transition")
            if should_stop_on_token(policy, eos_emitted=int(label) == self.eos_label, accepted=dfa.is_accepting(state)):
                break
        if not dfa.is_accepting(state):
            raise RuntimeError("planner did not terminate in an accepting DFA state")
        return labels, logprob

    def labels_logprob_from_context(
        self,
        context_vector: torch.Tensor,
        labels: Sequence[int],
        dfa,
        *,
        max_steps: int | None = None,
        stop_policy: StopPolicy | None = None,
    ) -> torch.Tensor:
        policy = stop_policy_from_legacy(max_new_tokens=max_steps, stop_policy=stop_policy)
        require_bound_predicate_dfa(dfa)
        state = dfa.start_state
        logprob = torch.zeros((), device=self.device)
        hidden = self._initial_decoder_state(context_vector, 1)
        previous = torch.tensor(
            [[int(self.eos_label)]], dtype=torch.long, device=self.device
        )
        replay_policy = replace(policy, timeout_seconds=None, external_stop_fn=None)
        update_progress = make_progress_tracker()
        for step, raw_label in enumerate(labels):
            progress = update_progress(step_index=step, dfa_state=state, prompt_token_count=0,
                                       generated_labels=labels[:step], accepted=dfa.is_accepting(state))
            if should_stop_decoding(replay_policy, progress):
                raise RuntimeError("planner replay trajectory exceeds its stop-policy budget")
            decoded, hidden = self.graph_decoder(self.token_embedding(previous), hidden)
            logits = self.output(decoded)[0, -1]
            allowed = dfa.allowed_tokens(state, remaining_steps=remaining_steps_for(policy, step))
            label_value = int(raw_label)
            if label_value not in allowed:
                raise RuntimeError("planner replay trajectory contains a rejected DFA transition")
            masked = torch.full_like(logits, float("-inf"))
            indices = torch.tensor(sorted(int(value) for value in allowed), device=logits.device)
            masked[indices] = logits[indices]
            from .research.runtime import joint_log_bias
            for choice,bias in joint_log_bias(dfa,state,allowed).items():masked[choice]+=bias
            distribution = torch.distributions.Categorical(logits=masked)
            label = torch.tensor(label_value, dtype=torch.long, device=logits.device)
            logprob = logprob + distribution.log_prob(label)
            previous = label.reshape(1, 1)
            state = dfa.step(state, label_value)
            if state is None:
                raise RuntimeError("planner replay trajectory contains a rejected DFA transition")
            if step + 1 < len(labels) and should_stop_on_token(
                policy, eos_emitted=label_value == self.eos_label, accepted=dfa.is_accepting(state),
            ):
                raise RuntimeError("planner replay trajectory continues after stop-policy termination")
        if not dfa.is_accepting(state):
            raise RuntimeError("planner replay trajectory is not accepting")
        return logprob

    def replay_labels_logprob(
        self,
        prepared_context: Mapping[str, Any],
        labels: Sequence[int],
        dfa,
        *,
        max_steps: int | None = None,
        stop_policy: StopPolicy | None = None,
    ) -> torch.Tensor:
        policy = stop_policy_from_legacy(max_new_tokens=max_steps, stop_policy=stop_policy)
        require_bound_predicate_dfa(dfa)
        return self.labels_logprob_from_context(
            self.encode_replay_context(prepared_context),
            labels,
            dfa,
            stop_policy=policy,
        )

    @torch.no_grad()
    def generate_plan(self, *, instruction, images, entity_table, dfa, world=None, max_steps: int | None = None, stop_policy: StopPolicy | None = None, **_kwargs):
        if max_steps is None and stop_policy is None:
            max_steps = 41
        labels, _ = self.sample_labels(
            {**{k: v for k, v in _kwargs.items() if k in ("planning_facts", "task_target_container", "task_target_entity", "task", "task_name", "task_skill_sequence")}, "instruction": instruction, "images": images, "entity_table": entity_table},
            dfa, max_steps=max_steps, stop_policy=stop_policy, deterministic=True,
        )
        return labels_to_plan(labels, self.vocabulary, world=world)

    def sample_with_logprob(self, *, instruction, images, entity_table, dfa, world=None, max_steps: int | None = None, stop_policy: StopPolicy | None = None, **kwargs):
        if max_steps is None and stop_policy is None:
            max_steps = 41
        encoded_context = kwargs.get("encoded_context")
        context = {**{k: v for k, v in kwargs.items() if k in ("planning_facts", "task_target_container", "task_target_entity", "task", "task_name", "task_skill_sequence")}, "instruction": instruction, "images": images, "entity_table": entity_table}
        from .research.runtime import policy_context
        context=policy_context(dfa,context)
        dfa = bind_vlabench_contextual_dfa(dfa, self.vocabulary, context)
        if encoded_context is None:
            labels, logprob = self.sample_labels(context, dfa, max_steps=max_steps, stop_policy=stop_policy)
        else:
            labels, logprob = self.sample_labels_from_context(
                encoded_context, dfa, max_steps=max_steps, stop_policy=stop_policy
            )
        plan = labels_to_plan(labels, self.vocabulary, world=world)
        if kwargs.get("return_labels", False):
            return plan, logprob, labels
        return plan, logprob

    def save_pretrained(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(target))
        self.processor.save_pretrained(str(target))
        torch.save(
            {
                "graph_decoder_version": self.graph_decoder_version,
                "decoder_hidden_size": self.decoder_hidden_size,
                "context_projection": self.context_projection.state_dict(),
                "token_embedding": self.token_embedding.state_dict(),
                "graph_decoder": self.graph_decoder.state_dict(),
                "output": self.output.state_dict(),
            },
            target / "graph_decoder.pt",
        )
        self.vocabulary.save(target / "vocab.json")
        from .research.checkpoint import save_adapter_metadata
        save_adapter_metadata(target,self)

    def compute_parameter_ownership_checksum(self) -> str:
        """Compute a deterministic checksum of standalone planner parameter structure and shapes."""
        import hashlib
        records = []
        for name, param in sorted(self.named_parameters(), key=lambda x: x[0]):
            records.append((name, tuple(param.shape), "vlabench_planner"))
        return hashlib.sha256(repr(records).encode("utf-8")).hexdigest()

