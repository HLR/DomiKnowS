"""Controller networks for simulator rollouts: image encoders, the multi-view actor-critic and its loss."""
from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..models import (
    VISION_MODEL_ID, load_sanitized_auto_config, sanitize_special_token_ids, special_token_id_summary,
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
    def __init__(self, model_id: str = VISION_MODEL_ID, *, local_files_only: bool = False, image_size: int | tuple[int, int] | None = None, return_patch_tokens: bool = False):
        super().__init__()
        self.return_patch_tokens = return_patch_tokens
        import transformers

        model_kwargs = {"local_files_only": local_files_only}
        auto_config = getattr(transformers, "AutoConfig", None)
        if auto_config is not None:
            model_kwargs["config"] = load_sanitized_auto_config(
                transformers, model_id, local_files_only=local_files_only
            )
            sanitize_special_token_ids(model_kwargs["config"])
            print(
                f"[vlabench-model] vision config before AutoModel.from_pretrained "
                f"model={model_id} {special_token_id_summary(model_kwargs['config'])}",
                file=sys.stderr,
                flush=True,
            )

        self.model = transformers.AutoModel.from_pretrained(model_id, **model_kwargs)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        config = self.model.config
        self.output_dim = int(
            getattr(config, "projection_dim", 0)
            or getattr(getattr(config, "vision_config", None), "hidden_size", 0)
            or getattr(config, "hidden_size", 768)
        )
        if image_size is not None:
            if isinstance(image_size, (int, float)):
                self.image_size = (int(image_size), int(image_size))
            else:
                self.image_size = (int(image_size[0]), int(image_size[1]))
        else:
            vision_config = getattr(config, "vision_config", None)
            detected_size = (
                getattr(vision_config, "image_size", None)
                or getattr(config, "image_size", None)
                or 224
            )
            self.image_size = (int(detected_size), int(detected_size))
        self.register_buffer("image_mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = F.interpolate(images.float(), size=self.image_size, mode="bilinear", align_corners=False)
        images = (images - self.image_mean) / self.image_std
        with torch.no_grad():
            if self.return_patch_tokens:
                value = self.model.vision_model(pixel_values=images).last_hidden_state
            else:
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
    std_penalty: torch.Tensor | None = None
    
    
class PlanConditionedAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int, max_views: int = 4):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.view_gate = nn.Parameter(torch.ones(max_views))
        self.scale = hidden_dim ** -0.5
        
    def forward(self, x: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        # x: [batch, history, views, seq_len, hidden_dim]
        # query: [batch, hidden_dim]
        b, h, v, s, d = x.shape
        q = self.query_proj(query).view(b, 1, 1, 1, d)
        k = self.key_proj(x)
        val = self.value_proj(x)
        
        # [batch, history, views, seq_len]
        attn = (q * k).sum(dim=-1) * self.scale
        
        # view gate
        gate = torch.sigmoid(self.view_gate[:v]).view(1, 1, -1, 1)
        attn = attn + torch.log(gate + 1e-6)
        
        # softmax over views and seq_len
        attn = attn.view(b, h, v * s)
        attn = F.softmax(attn, dim=-1).view(b, h, v, s, 1)
        
        # pool
        pooled = (attn * val).sum(dim=(2, 3)) # [batch, history, hidden_dim]
        return pooled


class MultiViewController(nn.Module):
    """Behavior-cloning controller and PPO actor-critic over EE chunks."""

    # Version 4 predicts tanh-bounded local increments, conditions them on the
    # active graph operation, and integrates them around the last observed EE
    # pose. The public action remains an absolute
    # [xyz, roll, pitch, yaw, gripper] target matching dataset and simulator.
    action_representation_version = 4
    # Version 3 trains on unit-range RGB for every decoding path.
    behavior_cloning_version = 3
    critic_version = 3
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
        
        # R6: MLP critic
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # R6: Attention pooling
        self.attention_pool = PlanConditionedAttentionPooling(hidden_dim, max_views)
        if hasattr(self.image_encoder, "return_patch_tokens"):
            self.image_encoder.return_patch_tokens = True
            
        self.pose_step_scale = (0.02, 0.02, 0.02, 0.10, 0.10, 0.10)
        # Standard deviations are in the pre-tanh local-delta space.  A value
        # of 0.35 explores without allowing a sampled target to escape the
        # graph-independent Cartesian safety envelope.
        self.exploration_std = (0.35,) * 6
        self.log_std = nn.Parameter(torch.tensor(self.exploration_std).log())
        # R5: state-conditioned std head
        self.std_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 6)
        )

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
        for layer in self.value_head:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
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
        for layer in self.value_head:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

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
        task = self.task_embedding(task_index.long().reshape(batch))
        if plan_context is not None:
            context = plan_context.long().reshape(batch, 3)
            skill = context[:, 0].clamp(0, self.skill_embedding.num_embeddings - 1)
            entity = context[:, 1].clamp(0, self.entity_embedding.num_embeddings - 1)
            operation = context[:, 2].clamp(0, self.operation_embedding.num_embeddings - 1)
            task = task + self.skill_embedding(skill) + self.entity_embedding(entity) + self.operation_embedding(operation)

        flat = images.reshape(batch * history * views, channels, height, width)
        enc_out = self.image_encoder(flat)
        if enc_out.ndim == 3: # [N, seq_len, dim]
            _, seq_len, vision_dim = enc_out.shape
            vision = enc_out.view(batch, history, views, seq_len, vision_dim)
            view_ids = torch.arange(views, device=images.device)
            vision = vision + self.view_embedding(view_ids).view(1, 1, views, 1, vision_dim)
            wrist_features = vision[:, -1, min(2, views - 1), :, :].mean(dim=-2).detach()
            vision = self.vision_projection(vision) # [batch, history, views, seq_len, hidden_dim]
            # Pool over views and seq_len
            vision = self.attention_pool(vision, task) # [b, h, d]
        else: # [N, dim]
            _, vision_dim = enc_out.shape
            vision = enc_out.view(batch, history, views, vision_dim)
            view_ids = torch.arange(views, device=images.device)
            vision = vision + self.view_embedding(view_ids).view(1, 1, views, vision_dim)
            wrist_features = vision[:, -1, min(2, views - 1), :].detach()
            vision = self.vision_projection(vision.mean(dim=2))

        state = state[..., : self.state_dim]
        if state.shape[-1] < self.state_dim:
            state = F.pad(state, (0, self.state_dim - state.shape[-1]))
        state_features = self.state_projection(state.float())
        temporal, _ = self.temporal(torch.cat((vision, state_features), dim=-1))
        return self.fusion(torch.cat((temporal[:, -1], task), dim=-1)), wrist_features

    def policy(self, images: torch.Tensor, state: torch.Tensor, task_index: torch.Tensor, plan_context=None) -> ControllerPolicyOutput:
        features, wrist_features = self._features(images, state, task_index, plan_context)
        raw = self.policy_head(features).reshape(-1, self.action_horizon, self.action_dim)
        pose_mean = self._local_pose_chunk(raw[..., :6], state)
        
        if hasattr(self, "std_head"):
            log_std = self.std_head(features)
            std = log_std.clamp(float(torch.tensor(0.05).log()), float(torch.tensor(0.75).log())).exp().view(-1, 1, 6).expand_as(raw[..., :6])
            std_penalty = (wrist_features.norm(dim=-1) * std.mean(dim=(1, 2))).mean() * 0.01
        else:
            std = self.log_std.clamp(float(torch.tensor(0.05).log()), float(torch.tensor(0.75).log())).exp().view(1, 1, 6).expand_as(raw[..., :6])
            std_penalty = torch.zeros((), device=raw.device)
            
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
            std_penalty,
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
        return logprob, entropy, output.value, output.std_penalty

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
