"""Autoregressive reward training for EAI action-token trajectories."""

from __future__ import annotations

import torch

from domiknows.reinforcement.reinforcement_program import ReinforcementProgram
from domiknows.reinforcement.rewards import as_reward_tensor, call_reward_function
from domiknows.reinforcement.sampling import importance_weighted_loss, reinforce_loss
from domiknows.program.model.base import Mode


class AutoregressiveSequenceReinforcementProgram(ReinforcementProgram):
    """A ReinforcementProgram whose samples are genuine policy rollouts.

    The generic program samples every graph variable from one fixed set of
    logits. For an autoregressive generator those logits are teacher-forced and
    later positions are conditioned on gold prefixes. This specialization
    instead recomputes logits after each sampled prefix before applying the
    existing DomiKnowS reward estimators.
    """

    def __init__(self, *args, autoregressive_head, eos_label, max_steps, **kwargs):
        super().__init__(*args, **kwargs)
        self.autoregressive_head = autoregressive_head
        self.eos_label = int(eos_label)
        self.max_steps = int(max_steps)

    def _sample_trajectories(self, text):
        device = next(self.autoregressive_head.parameters()).device
        sample_count = int(self.num_samples)
        prefixes = torch.full(
            (sample_count, 1),
            self.eos_label,
            dtype=torch.long,
            device=device,
        )
        finished = torch.zeros(sample_count, dtype=torch.bool, device=device)
        trajectory_logprob = torch.zeros(sample_count, device=device)
        trajectories = [[] for _ in range(sample_count)]

        for _step in range(self.max_steps):
            if getattr(self.autoregressive_head, "supports_batched_prefixes", False):
                logits = self.autoregressive_head.sequence_logits(text, prefixes)[:, -1, :]
                if logits.shape[0] != sample_count:
                    raise ValueError("generator did not preserve the rollout batch")
            else:
                # The GRU head carries a single-example encoder hidden state and
                # cannot broadcast it across rollout prefixes.
                logits = torch.stack(
                    [
                        self.autoregressive_head.sequence_logits(
                            text, prefixes[index : index + 1]
                        )[0, -1, :]
                        for index in range(sample_count)
                    ],
                    dim=0,
                )
            distribution = torch.distributions.Categorical(logits=logits)
            sampled = distribution.sample()
            active = ~finished
            sampled = torch.where(active, sampled, torch.full_like(sampled, self.eos_label))
            chosen_logprob = torch.log_softmax(logits, dim=-1).gather(
                -1, sampled.unsqueeze(-1)
            ).squeeze(-1)
            trajectory_logprob = trajectory_logprob + torch.where(
                active, chosen_logprob, torch.zeros_like(chosen_logprob)
            )

            sampled_values = sampled.detach().cpu().tolist()
            active_values = active.detach().cpu().tolist()
            for index, (label, is_active) in enumerate(zip(sampled_values, active_values)):
                if is_active:
                    trajectories[index].append(int(label))

            prefixes = torch.cat([prefixes, sampled.unsqueeze(-1)], dim=-1)
            finished = finished | (sampled == self.eos_label)
            if bool(finished.all()):
                break

        return trajectories, trajectory_logprob

    def reinforcement_loss(self, datanode, reward_fn, data_item):
        if reward_fn is None:
            return super().reinforcement_loss(datanode, reward_fn, data_item)

        trajectories, logprob = self._sample_trajectories(data_item.get("text", ""))
        reward_values = []
        for trajectory in trajectories:
            reward = call_reward_function(
                reward_fn,
                trajectory,
                data_item=data_item,
                datanode=datanode,
            )
            reward_values.append(as_reward_tensor(reward, dtype=torch.float32).mean().item())
        rewards = torch.tensor(reward_values, dtype=logprob.dtype, device=logprob.device)

        if self.estimator == "importance_weighted":
            loss = importance_weighted_loss(logprob, rewards)
        else:
            loss = reinforce_loss(logprob, rewards, baseline=self.baseline)
        return loss, rewards.mean().item()

    def train_epoch(self, dataset, **kwargs):
        """Train directly from rollouts without an unused gold-prefix forward."""
        del kwargs
        self.model.mode(Mode.TRAIN)
        self.model.reset()
        for data_item in dataset:
            reward_fn = self._resolve_reward_fn(data_item)
            with self._autocast_ctx():
                loss, _reward = self.reinforcement_loss(None, reward_fn, data_item)
            self._backward_and_step(loss)
            yield (loss, None, None)

    def test_epoch(self, dataset, **kwargs):
        del kwargs
        self.model.mode(Mode.TEST)
        self.model.reset()
        with torch.no_grad(), self._autocast_ctx():
            for data_item in dataset:
                reward_fn = self._resolve_reward_fn(data_item)
                loss, _reward = self.reinforcement_loss(None, reward_fn, data_item)
                yield (loss, None, None)
