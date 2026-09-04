"""Compact graph-token planner and stochastic multi-view controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

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
    latent_pose_mean: torch.Tensor | None = None


class MultiViewController(nn.Module):
    """Behavior-cloning controller and PPO actor-critic over EE chunks."""

    # Version 4 predicts tanh-bounded local increments, conditions them on the
    # active graph operation, and integrates them around the last observed EE
    # pose. The public action remains an absolute
    # [xyz, roll, pitch, yaw, gripper] target matching dataset and simulator.
    action_representation_version = 4
    critic_version = 2
    plan_conditioning_version = 1

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
        controller_skill_count: int = 10,
        controller_entity_count: int = 65,
        controller_operation_count: int = 17,
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
        self.skill_embedding = nn.Embedding(controller_skill_count, hidden_dim, padding_idx=0)
        self.entity_embedding = nn.Embedding(controller_entity_count, hidden_dim, padding_idx=0)
        self.operation_embedding = nn.Embedding(controller_operation_count, hidden_dim, padding_idx=0)
        self.temporal = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())
        self.policy_head = nn.Linear(hidden_dim, action_horizon * action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.pose_step_scale = (0.02, 0.02, 0.02, 0.10, 0.10, 0.10)
        # Standard deviations are in the pre-tanh local-delta space.  A value
        # of 0.35 explores without allowing a sampled target to escape the
        # graph-independent Cartesian safety envelope.
        self.exploration_std = (0.35,) * 6
        self.log_std = nn.Parameter(torch.tensor(self.exploration_std).log())

    def reset_for_action_representation_migration(self) -> None:
        """Reset only the legacy absolute-pose actor parameters.

        Vision, state, temporal, task and value features remain available when
        a Stage 1 checkpoint made by the old absolute actor is migrated.  The
        controller-only warm-up then learns the local action head cleanly.
        """

        self.policy_head.reset_parameters()
        self.task_embedding.reset_parameters()
        self.skill_embedding.reset_parameters()
        self.entity_embedding.reset_parameters()
        self.operation_embedding.reset_parameters()
        self.value_head.reset_parameters()
        with torch.no_grad():
            self.log_std.copy_(
                torch.tensor(
                    self.exploration_std,
                    dtype=self.log_std.dtype,
                    device=self.log_std.device,
                ).log()
            )

    def reset_critic_for_migration(self) -> None:
        """Reset the formerly unbounded critic while preserving the actor."""

        self.value_head.reset_parameters()

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

    def _features(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        task_index: torch.Tensor,
        plan_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        if plan_context is not None:
            context = plan_context.long().reshape(batch, 3)
            skill = context[:, 0].clamp(0, self.skill_embedding.num_embeddings - 1)
            entity = context[:, 1].clamp(0, self.entity_embedding.num_embeddings - 1)
            operation = context[:, 2].clamp(0, self.operation_embedding.num_embeddings - 1)
            task = (
                task
                + self.skill_embedding(skill)
                + self.entity_embedding(entity)
                + self.operation_embedding(operation)
            )
        return self.fusion(torch.cat((temporal[:, -1], task), dim=-1))

    def policy(self, images: torch.Tensor, state: torch.Tensor, task_index: torch.Tensor, plan_context=None) -> ControllerPolicyOutput:
        features = self._features(images, state, task_index, plan_context)
        raw = self.policy_head(features).reshape(-1, self.action_horizon, self.action_dim)
        pose_mean = self._local_pose_chunk(raw[..., :6], state)
        std = self.log_std.clamp(float(torch.tensor(0.05).log()), float(torch.tensor(0.75).log())).exp().view(1, 1, 6).expand_as(raw[..., :6])
        output = ControllerPolicyOutput(
            pose_mean,
            std,
            raw[..., 6],
            # Simulator returns are bounded. A bounded critic prevents one bad
            # value estimate from recursively creating enormous GAE targets.
            # Detaching its input also prevents zero-reward value fitting from
            # silently changing the actor's shared visual/control features.
            torch.tanh(self.value_head(features.detach())).squeeze(-1),
            raw[..., :6],
        )
        for name, value in (
            ("pose mean", output.pose_mean),
            ("pose standard deviation", output.pose_std),
            ("gripper logits", output.gripper_logits),
            ("value", output.value),
            ("latent pose mean", output.latent_pose_mean),
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"controller produced non-finite {name}")
        return output

    def forward(self, images: torch.Tensor, state: torch.Tensor, task_index: torch.Tensor, plan_context=None) -> torch.Tensor:
        output = self.policy(images, state, task_index, plan_context)
        return torch.cat((output.pose_mean, output.gripper_logits.unsqueeze(-1)), dim=-1)

    def _bounded_pose_logprob(self, latent_distribution, latent, bounded_delta):
        scale = bounded_delta.new_tensor(self.pose_step_scale).view(1, 1, 6)
        normalized = (bounded_delta / scale).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        jacobian = torch.log(scale) + torch.log1p(-(normalized * normalized) + 1e-6)
        return latent_distribution.log_prob(latent) - jacobian

    def sample_action_chunk(self, images, state, task_index, plan_context=None):
        output = self.policy(images, state, task_index, plan_context)
        pose_distribution = torch.distributions.Normal(output.latent_pose_mean, output.pose_std)
        grip_distribution = torch.distributions.Bernoulli(logits=output.gripper_logits)
        latent = pose_distribution.rsample()
        scale = latent.new_tensor(self.pose_step_scale).view(1, 1, 6)
        delta = torch.tanh(latent) * scale
        base = state[:, -1, :6].to(device=latent.device, dtype=latent.dtype).unsqueeze(1)
        pose = base + delta.cumsum(dim=1)
        pose = torch.cat((pose[..., :3], torch.atan2(torch.sin(pose[..., 3:]), torch.cos(pose[..., 3:]))), -1)
        grip = grip_distribution.sample()
        actions = torch.cat((pose, grip.unsqueeze(-1)), dim=-1)
        logprob = self._bounded_pose_logprob(pose_distribution, latent, delta).sum(-1) + grip_distribution.log_prob(grip)
        entropy = pose_distribution.entropy().sum(-1) + grip_distribution.entropy()
        return actions, logprob, entropy, output.value

    def evaluate_action_chunk(self, images, state, task_index, actions, plan_context=None):
        output = self.policy(images, state, task_index, plan_context)
        pose_distribution = torch.distributions.Normal(output.latent_pose_mean, output.pose_std)
        grip_distribution = torch.distributions.Bernoulli(logits=output.gripper_logits)
        base = state[:, -1, :6].to(device=actions.device, dtype=actions.dtype).unsqueeze(1)
        previous = torch.cat((base, actions[:, :-1, :6]), dim=1)
        delta = actions[..., :6] - previous
        delta = torch.cat((delta[..., :3], torch.atan2(torch.sin(delta[..., 3:]), torch.cos(delta[..., 3:]))), -1)
        scale = delta.new_tensor(self.pose_step_scale).view(1, 1, 6)
        normalized = (delta / scale).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        latent = torch.atanh(normalized)
        grip = actions[..., 6].clamp(0, 1)
        logprob = self._bounded_pose_logprob(pose_distribution, latent, delta).sum(-1) + grip_distribution.log_prob(grip)
        entropy = pose_distribution.entropy().sum(-1) + grip_distribution.entropy()
        return logprob, entropy, output.value

    @torch.no_grad()
    def predict_action_chunk(self, images, state, task_index, plan_context=None) -> torch.Tensor:
        was_training = self.training
        self.eval()
        output = self.policy(images, state, task_index, plan_context)
        actions = torch.cat((output.pose_mean, (torch.sigmoid(output.gripper_logits) >= 0.5).float().unsqueeze(-1)), -1)
        self.train(was_training)
        return actions


def controller_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    state: torch.Tensor | None = None,
    pose_step_scale: Sequence[float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if prediction.shape != target.shape:
        raise ValueError(f"controller prediction {prediction.shape} does not match target {target.shape}")
    target = target.float()
    if state is None or pose_step_scale is None:
        pose = F.smooth_l1_loss(prediction[..., :-1], target[..., :-1])
    else:
        if state.ndim != 3 or state.shape[0] != prediction.shape[0] or state.shape[-1] < 6:
            raise ValueError("controller state must be [batch, history, >=6] for delta loss")
        scale = prediction.new_tensor(tuple(pose_step_scale)).view(1, 1, 6)
        if scale.numel() != 6 or not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError("controller pose step scale must contain six finite positive values")
        base = state[:, -1, :6].to(device=prediction.device, dtype=prediction.dtype).unsqueeze(1)
        predicted_previous = torch.cat((base, prediction[:, :-1, :6]), dim=1)
        target_previous = torch.cat((base, target[:, :-1, :6]), dim=1)
        predicted_delta = prediction[..., :6] - predicted_previous
        target_delta = target[..., :6] - target_previous
        predicted_delta = torch.cat((
            predicted_delta[..., :3],
            torch.atan2(torch.sin(predicted_delta[..., 3:]), torch.cos(predicted_delta[..., 3:])),
        ), dim=-1)
        target_delta = torch.cat((
            target_delta[..., :3],
            torch.atan2(torch.sin(target_delta[..., 3:]), torch.cos(target_delta[..., 3:])),
        ), dim=-1)
        # The deployed policy can only emit one bounded step per action.  Fit
        # the same normalized delta representation used by sampling instead of
        # allowing large absolute coordinates to dominate behavior cloning.
        pose = F.smooth_l1_loss(
            predicted_delta / scale,
            (target_delta / scale).clamp(-1.0, 1.0),
        )
    gripper = F.binary_cross_entropy_with_logits(prediction[..., -1], target[..., -1].float())
    loss = pose + gripper
    return loss, {"loss": float(loss.detach()), "pose_loss": float(pose.detach()), "gripper_loss": float(gripper.detach())}


class QwenVLPlanner(nn.Module):
    """Qwen2.5-VL context encoder plus compact graph-token decoder.

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

    def supervised_loss(self, *, instruction, images, entity_table, target_plan, world=None) -> torch.Tensor:
        tokens = plan_to_tokens(target_plan, entity_table, world=world)
        labels = torch.tensor([self.vocabulary.label_for_token(token) for token in tokens], device=self.device)
        logits = self.sequence_logits(
            {"instruction": instruction, "images": images, "entity_table": entity_table},
            self._shift_right(labels),
        )[0]
        return F.cross_entropy(logits, labels)

    def sample_labels(self, context: Mapping[str, Any], dfa, *, max_steps: int, deterministic: bool = False):
        context_vector = self.encode_context(context)
        return self.sample_labels_from_context(
            context_vector,
            dfa,
            max_steps=max_steps,
            deterministic=deterministic,
        )

    def sample_labels_from_context(
        self,
        context_vector: torch.Tensor,
        dfa,
        *,
        max_steps: int,
        deterministic: bool = False,
    ):
        state = dfa.start_state
        labels = []
        logprob = torch.zeros((), device=self.device)
        hidden = self._initial_decoder_state(context_vector, 1)
        previous = torch.tensor(
            [[int(self.eos_label)]], dtype=torch.long, device=self.device
        )
        for step in range(max_steps):
            decoded, hidden = self.graph_decoder(self.token_embedding(previous), hidden)
            logits = self.output(decoded)[0, -1]
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
            previous = label.reshape(1, 1)
            state = dfa.step(state, int(label))
            if state is None:
                raise RuntimeError("planner emitted a rejected DFA transition")
            if int(label) == self.eos_label and dfa.is_accepting(state):
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
        max_steps: int,
    ) -> torch.Tensor:
        state = dfa.start_state
        logprob = torch.zeros((), device=self.device)
        hidden = self._initial_decoder_state(context_vector, 1)
        previous = torch.tensor(
            [[int(self.eos_label)]], dtype=torch.long, device=self.device
        )
        for step, raw_label in enumerate(labels):
            if step >= int(max_steps):
                raise RuntimeError("planner replay trajectory exceeds its maximum length")
            decoded, hidden = self.graph_decoder(self.token_embedding(previous), hidden)
            logits = self.output(decoded)[0, -1]
            allowed = dfa.allowed_tokens(state, remaining_steps=max_steps - step - 1)
            label_value = int(raw_label)
            if label_value not in allowed:
                raise RuntimeError("planner replay trajectory contains a rejected DFA transition")
            masked = torch.full_like(logits, float("-inf"))
            indices = torch.tensor(sorted(int(value) for value in allowed), device=logits.device)
            masked[indices] = logits[indices]
            distribution = torch.distributions.Categorical(logits=masked)
            label = torch.tensor(label_value, dtype=torch.long, device=logits.device)
            logprob = logprob + distribution.log_prob(label)
            previous = label.reshape(1, 1)
            state = dfa.step(state, label_value)
            if state is None:
                raise RuntimeError("planner replay trajectory contains a rejected DFA transition")
        if not dfa.is_accepting(state):
            raise RuntimeError("planner replay trajectory is not accepting")
        return logprob

    def replay_labels_logprob(
        self,
        prepared_context: Mapping[str, Any],
        labels: Sequence[int],
        dfa,
        *,
        max_steps: int,
    ) -> torch.Tensor:
        return self.labels_logprob_from_context(
            self.encode_replay_context(prepared_context),
            labels,
            dfa,
            max_steps=max_steps,
        )

    @torch.no_grad()
    def generate_plan(self, *, instruction, images, entity_table, dfa, world=None, max_steps: int = 41, **_kwargs):
        labels, _ = self.sample_labels(
            {"instruction": instruction, "images": images, "entity_table": entity_table},
            dfa, max_steps=max_steps, deterministic=True,
        )
        return labels_to_plan(labels, self.vocabulary, world=world)

    def sample_with_logprob(self, *, instruction, images, entity_table, dfa, world=None, max_steps: int = 41, **kwargs):
        encoded_context = kwargs.get("encoded_context")
        context = {"instruction": instruction, "images": images, "entity_table": entity_table}
        if encoded_context is None:
            labels, logprob = self.sample_labels(context, dfa, max_steps=max_steps)
        else:
            labels, logprob = self.sample_labels_from_context(
                encoded_context, dfa, max_steps=max_steps
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
