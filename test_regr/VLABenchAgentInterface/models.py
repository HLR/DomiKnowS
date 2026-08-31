"""Compact graph-token planner and stochastic multi-view controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

try:
    from .graph import PlanVocabulary, labels_to_plan, plan_to_tokens
except ImportError:
    from graph import PlanVocabulary, labels_to_plan, plan_to_tokens


PLANNER_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
VISION_MODEL_ID = "google/siglip-base-patch16-224"


def resolve_vision_language_loader():
    """Return a Transformers VLM loader across its renamed auto-model APIs."""

    import transformers

    processor_class = getattr(transformers, "AutoProcessor", None)
    if processor_class is None:
        raise ImportError("the installed transformers package does not export AutoProcessor")
    failures = []
    for class_name in (
        "AutoModelForImageTextToText",
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
        "No compatible Qwen2.5-VL model loader is available in transformers "
        f"{version}. Tried AutoModelForImageTextToText, "
        "Qwen2_5_VLForConditionalGeneration, and AutoModelForVision2Seq. "
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
    entities = "\n".join(f"{index}: {name}" for index, name in enumerate(entity_table)) or "No objects."
    skills = ", ".join(vocabulary.skills)
    return (
        "Predict a compact VLABench plan using only graph labels. "
        "A skill label is followed by its graph-declared argument and object pointer labels.\n"
        f"Skills: {skills}\nObjects:\n{entities}\nInstruction: {instruction}\nPlan labels:"
    )


class TinyImageEncoder(nn.Module):
    def __init__(self, output_dim: int = 64):
        super().__init__()
        self.output_dim = int(output_dim)
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, self.output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images.float())


class FrozenSigLIPEncoder(nn.Module):
    def __init__(self, model_id: str = VISION_MODEL_ID, *, local_files_only: bool = False):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_id, local_files_only=local_files_only)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        config = self.model.config
        self.output_dim = int(
            getattr(config, "projection_dim", 0)
            or getattr(getattr(config, "vision_config", None), "hidden_size", 0)
            or getattr(config, "hidden_size", 768)
        )
        self.register_buffer("image_mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = F.interpolate(images.float(), size=(224, 224), mode="bilinear", align_corners=False)
        images = (images - self.image_mean) / self.image_std
        with torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                value = self.model.get_image_features(pixel_values=images)
            else:
                value = self.model.vision_model(pixel_values=images)
            if not torch.is_tensor(value):
                image_embeds = getattr(value, "image_embeds", None)
                pooler_output = getattr(value, "pooler_output", None)
                last_hidden_state = getattr(value, "last_hidden_state", None)
                if image_embeds is not None:
                    value = image_embeds
                elif pooler_output is not None:
                    value = pooler_output
                elif last_hidden_state is not None:
                    value = last_hidden_state[:, 0]
                else:
                    raise TypeError(
                        "SigLIP image encoder returned neither a tensor nor pooled/hidden-state features"
                    )
        return value.float()


@dataclass(frozen=True)
class ControllerPolicyOutput:
    pose_mean: torch.Tensor
    pose_std: torch.Tensor
    gripper_logits: torch.Tensor
    value: torch.Tensor


class MultiViewController(nn.Module):
    """Behavior-cloning controller and PPO actor-critic over EE chunks."""

    # Version 2 predicts a sequence of bounded local increments and integrates
    # them around the last observed EE pose.  The public action remains an
    # absolute [xyz, roll, pitch, yaw, gripper] target, matching the dataset and
    # simulator adapter, but the actor can no longer point every action in a
    # chunk at an arbitrary distant pose.
    action_representation_version = 3

    def __init__(
        self,
        image_encoder: nn.Module,
        *,
        state_dim: int = 7,
        action_dim: int = 7,
        action_horizon: int = 16,
        hidden_dim: int = 256,
        max_views: int = 4,
        task_count: int = 128,
    ):
        super().__init__()
        if action_dim != 7:
            raise ValueError("VLABench controller actions must be 6D EE pose plus gripper")
        vision_dim = int(getattr(image_encoder, "output_dim"))
        self.image_encoder = image_encoder
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.view_embedding = nn.Embedding(max_views, vision_dim)
        self.state_projection = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.vision_projection = nn.Sequential(nn.Linear(vision_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.task_embedding = nn.Embedding(task_count, hidden_dim)
        self.temporal = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())
        self.policy_head = nn.Linear(hidden_dim, action_horizon * action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.pose_step_scale = (0.02, 0.02, 0.02, 0.10, 0.10, 0.10)
        self.exploration_std = (0.01, 0.01, 0.01, 0.05, 0.05, 0.05)
        self.log_std = nn.Parameter(torch.tensor(self.exploration_std).log())

    def reset_for_action_representation_migration(self) -> None:
        """Reset only the legacy absolute-pose actor parameters.

        Vision, state, temporal, task and value features remain available when
        a Stage 1 checkpoint made by the old absolute actor is migrated.  The
        controller-only warm-up then learns the local action head cleanly.
        """

        self.policy_head.reset_parameters()
        self.task_embedding.reset_parameters()
        with torch.no_grad():
            self.log_std.copy_(
                torch.tensor(
                    self.exploration_std,
                    dtype=self.log_std.dtype,
                    device=self.log_std.device,
                ).log()
            )

    def _local_pose_chunk(
        self,
        raw_pose: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        if state.ndim != 3 or state.shape[0] != raw_pose.shape[0] or state.shape[-1] < 6:
            raise ValueError("controller state must contain a history of 6D EE poses")
        scale = raw_pose.new_tensor(self.pose_step_scale).view(1, 1, 6)
        increments = torch.tanh(raw_pose) * scale
        base = state[:, -1, :6].to(device=raw_pose.device, dtype=raw_pose.dtype).unsqueeze(1)
        pose = base + increments.cumsum(dim=1)
        # Keep Euler coordinates on their principal branch without severing
        # gradients, including chunks that cross the -pi/pi boundary.
        angles = torch.atan2(
            torch.sin(pose[..., 3:6]),
            torch.cos(pose[..., 3:6]),
        )
        pose = torch.cat((pose[..., :3], angles), -1)
        return pose

    def _features(self, images: torch.Tensor, state: torch.Tensor, task_index: torch.Tensor) -> torch.Tensor:
        if images.ndim != 6:
            raise ValueError("images must have shape [batch, history, views, channels, height, width]")
        batch, history, views, channels, height, width = images.shape
        if views > self.view_embedding.num_embeddings:
            raise ValueError(f"controller supports at most {self.view_embedding.num_embeddings} views")
        flat = images.reshape(batch * history * views, channels, height, width)
        vision = self.image_encoder(flat).reshape(batch, history, views, -1)
        view_ids = torch.arange(views, device=images.device)
        vision = vision + self.view_embedding(view_ids).view(1, 1, views, -1)
        vision = self.vision_projection(vision.mean(dim=2))
        state = state[..., : self.state_dim]
        if state.shape[-1] < self.state_dim:
            state = F.pad(state, (0, self.state_dim - state.shape[-1]))
        state_features = self.state_projection(state.float())
        temporal, _ = self.temporal(torch.cat((vision, state_features), dim=-1))
        task = self.task_embedding(task_index.long().reshape(batch))
        return self.fusion(torch.cat((temporal[:, -1], task), dim=-1))

    def policy(self, images: torch.Tensor, state: torch.Tensor, task_index: torch.Tensor) -> ControllerPolicyOutput:
        features = self._features(images, state, task_index)
        raw = self.policy_head(features).reshape(-1, self.action_horizon, self.action_dim)
        pose_mean = self._local_pose_chunk(raw[..., :6], state)
        std = self.log_std.clamp(-5.0, 2.0).exp().view(1, 1, 6).expand_as(raw[..., :6])
        output = ControllerPolicyOutput(
            pose_mean,
            std,
            raw[..., 6],
            self.value_head(features).squeeze(-1),
        )
        for name, value in (
            ("pose mean", output.pose_mean),
            ("pose standard deviation", output.pose_std),
            ("gripper logits", output.gripper_logits),
            ("value", output.value),
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"controller produced non-finite {name}")
        return output

    def forward(self, images: torch.Tensor, state: torch.Tensor, task_index: torch.Tensor) -> torch.Tensor:
        output = self.policy(images, state, task_index)
        return torch.cat((output.pose_mean, output.gripper_logits.unsqueeze(-1)), dim=-1)

    def sample_action_chunk(self, images, state, task_index):
        output = self.policy(images, state, task_index)
        pose_distribution = torch.distributions.Normal(output.pose_mean, output.pose_std)
        grip_distribution = torch.distributions.Bernoulli(logits=output.gripper_logits)
        pose = pose_distribution.rsample()
        grip = grip_distribution.sample()
        actions = torch.cat((pose, grip.unsqueeze(-1)), dim=-1)
        logprob = pose_distribution.log_prob(pose).sum(-1) + grip_distribution.log_prob(grip)
        entropy = pose_distribution.entropy().sum(-1) + grip_distribution.entropy()
        return actions, logprob, entropy, output.value

    def evaluate_action_chunk(self, images, state, task_index, actions):
        output = self.policy(images, state, task_index)
        pose_distribution = torch.distributions.Normal(output.pose_mean, output.pose_std)
        grip_distribution = torch.distributions.Bernoulli(logits=output.gripper_logits)
        grip = actions[..., 6].clamp(0, 1)
        logprob = pose_distribution.log_prob(actions[..., :6]).sum(-1) + grip_distribution.log_prob(grip)
        entropy = pose_distribution.entropy().sum(-1) + grip_distribution.entropy()
        return logprob, entropy, output.value

    @torch.no_grad()
    def predict_action_chunk(self, images, state, task_index) -> torch.Tensor:
        was_training = self.training
        self.eval()
        output = self.policy(images, state, task_index)
        actions = torch.cat((output.pose_mean, (torch.sigmoid(output.gripper_logits) >= 0.5).float().unsqueeze(-1)), -1)
        self.train(was_training)
        return actions


def controller_loss(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    if prediction.shape != target.shape:
        raise ValueError(f"controller prediction {prediction.shape} does not match target {target.shape}")
    pose = F.smooth_l1_loss(prediction[..., :-1], target[..., :-1].float())
    gripper = F.binary_cross_entropy_with_logits(prediction[..., -1], target[..., -1].float())
    loss = pose + gripper
    return loss, {"loss": float(loss.detach()), "pose_loss": float(pose.detach()), "gripper_loss": float(gripper.detach())}


class QwenVLPlanner(nn.Module):
    """Qwen2.5-VL backbone with a compact graph-label projection head."""

    supports_batched_prefixes = False

    def __init__(self, model: nn.Module, processor: Any, vocabulary: PlanVocabulary, hidden_size: int | None = None):
        super().__init__()
        self.model = model
        self.processor = processor
        self.vocabulary = vocabulary
        hidden_size = hidden_size or vision_language_hidden_size(model)
        if hidden_size is None:
            raise ValueError("planner hidden size is required when the model config does not declare it")
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
        self.output = nn.Linear(int(hidden_size), vocabulary.label_count).to(model_device)
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
    ) -> "QwenVLPlanner":
        model_class, processor_class = resolve_vision_language_loader()

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        kwargs: dict[str, Any] = {"dtype": dtype, "local_files_only": local_files_only}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4",
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
        planner = cls(model, processor, vocabulary, hidden_size=hidden_size)
        head_path = Path(adapter_path) / "compact_head.pt" if adapter_path else None
        if head_path is not None and head_path.exists():
            planner.output.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
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

    def _model_logits(self, inputs: Mapping[str, Any]) -> torch.Tensor:
        output = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden_states = getattr(output, "hidden_states", None) or getattr(output, "decoder_hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("vision-language backbone did not return hidden states")
        hidden = hidden_states[-1][:, -1, :].float().to(self.output.weight.device)
        return self.output(hidden)[0]

    def _checkpointed_model_logits(self, inputs: Mapping[str, Any]) -> torch.Tensor:
        if not self.training or not torch.is_grad_enabled():
            return self._model_logits(inputs)
        tensor_keys = tuple(key for key, value in inputs.items() if torch.is_tensor(value))
        if not tensor_keys:
            return self._model_logits(inputs)
        static_inputs = {key: value for key, value in inputs.items() if key not in tensor_keys}

        def forward(*values):
            rebuilt = dict(static_inputs)
            rebuilt.update(zip(tensor_keys, values))
            return self._model_logits(rebuilt)

        return activation_checkpoint(
            forward,
            *(inputs[key] for key in tensor_keys),
            use_reentrant=False,
            preserve_rng_state=True,
        )

    def _next_logits(self, context: Mapping[str, Any], prefix_labels: Sequence[int]) -> torch.Tensor:
        inputs = self._inputs(context, prefix_labels)
        return self._checkpointed_model_logits(inputs)

    def sequence_logits(self, context: Mapping[str, Any], prefix_labels: torch.Tensor) -> torch.Tensor:
        prefixes = torch.as_tensor(prefix_labels, dtype=torch.long, device=self.device)
        if prefixes.ndim == 1:
            prefixes = prefixes.unsqueeze(0)
        rows = []
        for row in prefixes:
            logits = []
            values = row.detach().cpu().tolist()
            for index in range(len(values)):
                logits.append(self._next_logits(context, values[: index + 1]))
            rows.append(torch.stack(logits))
        return torch.stack(rows)

    def _shift_right(self, labels: torch.Tensor) -> torch.Tensor:
        labels = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        start = torch.full((labels.shape[0], 1), self.eos_label, dtype=torch.long, device=self.device)
        return torch.cat((start, labels[:, :-1]), dim=1)

    def forward(self, _contains, context: Mapping[str, Any], target_labels: torch.Tensor) -> torch.Tensor:
        logits = self.sequence_logits(context, self._shift_right(target_labels))
        return logits.squeeze(0) if logits.shape[0] == 1 else logits

    def supervised_loss(self, *, instruction, images, entity_table, target_plan, world=None) -> torch.Tensor:
        tokens = plan_to_tokens(target_plan, entity_table, world=world)
        labels = torch.tensor([self.vocabulary.label_for_token(token) for token in tokens], device=self.device)
        logits = self.sequence_logits(
            {"instruction": instruction, "images": images, "entity_table": entity_table},
            self._shift_right(labels),
        )[0]
        return F.cross_entropy(logits, labels)

    def sample_labels(self, context: Mapping[str, Any], dfa, *, max_steps: int, deterministic: bool = False):
        state = dfa.start_state
        labels = []
        logprob = torch.zeros((), device=self.device)
        for step in range(max_steps):
            logits = self._next_logits(context, [self.eos_label, *labels])
            allowed = dfa.allowed_tokens(state, remaining_steps=max_steps - step - 1)
            mask = torch.full_like(logits, float("-inf"))
            if not allowed:
                raise RuntimeError("planner DFA has no productive transition")
            indices = torch.tensor(sorted(int(label) for label in allowed), device=logits.device)
            mask[indices] = logits[indices]
            distribution = torch.distributions.Categorical(logits=mask)
            label = torch.argmax(mask) if deterministic else distribution.sample()
            logprob = logprob + distribution.log_prob(label)
            labels.append(int(label))
            state = dfa.step(state, int(label))
            if state is None:
                raise RuntimeError("planner emitted a rejected DFA transition")
            if int(label) == self.eos_label and dfa.is_accepting(state):
                break
        if not dfa.is_accepting(state):
            raise RuntimeError("planner did not terminate in an accepting DFA state")
        return labels, logprob

    @torch.no_grad()
    def generate_plan(self, *, instruction, images, entity_table, dfa, world=None, max_steps: int = 41, **_kwargs):
        labels, _ = self.sample_labels(
            {"instruction": instruction, "images": images, "entity_table": entity_table},
            dfa, max_steps=max_steps, deterministic=True,
        )
        return labels_to_plan(labels, self.vocabulary, world=world)

    def sample_with_logprob(self, *, instruction, images, entity_table, dfa, world=None, max_steps: int = 41, **_kwargs):
        labels, logprob = self.sample_labels(
            {"instruction": instruction, "images": images, "entity_table": entity_table},
            dfa, max_steps=max_steps,
        )
        return labels_to_plan(labels, self.vocabulary, world=world), logprob

    def save_pretrained(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(target))
        self.processor.save_pretrained(str(target))
        torch.save(self.output.state_dict(), target / "compact_head.pt")
        self.vocabulary.save(target / "vocab.json")
