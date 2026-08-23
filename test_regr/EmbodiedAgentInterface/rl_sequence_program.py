"""Autoregressive reward training for EAI action-token trajectories."""

from __future__ import annotations

import torch
from torch.nn import functional as F

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

    def __init__(
        self,
        *args,
        autoregressive_head,
        eos_label,
        max_steps,
        supervised_weight=0.5,
        policy_dfa=None,
        policy_dfa_factory=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.autoregressive_head = autoregressive_head
        self.eos_label = int(eos_label)
        self.max_steps = int(max_steps)
        self.supervised_weight = float(supervised_weight)
        self.policy_dfa = policy_dfa
        self.policy_dfa_factory = policy_dfa_factory
        if self.supervised_weight < 0.0:
            raise ValueError("supervised_weight must be non-negative")

    def supervised_anchor_loss(self, data_item):
        """Teacher-forced Stage 1 loss used to prevent RL policy collapse."""
        labels = data_item.get("target_action_labels")
        if labels is None:
            return None
        device = next(self.autoregressive_head.parameters()).device
        labels = torch.as_tensor(labels, dtype=torch.long, device=device).reshape(1, -1)
        start = torch.full(
            (labels.shape[0], 1), self.eos_label, dtype=torch.long, device=device
        )
        prefixes = torch.cat([start, labels[:, :-1]], dim=1)
        logits = self.autoregressive_head.sequence_logits(
            data_item.get("text", ""), prefixes
        )
        positions = torch.arange(labels.shape[1], device=device).unsqueeze(0)
        eos_positions = torch.where(
            labels == self.eos_label,
            positions,
            torch.full_like(positions, labels.shape[1]),
        )
        first_eos = eos_positions.min(dim=1).values
        mask = positions <= first_eos.unsqueeze(1)
        return F.cross_entropy(logits[mask], labels[mask])

    def _sample_trajectories(self, text, policy_dfa=None):
        device = next(self.autoregressive_head.parameters()).device
        sample_count = int(self.num_samples)
        prefixes = torch.full(
            (sample_count, 1),
            self.eos_label,
            dtype=torch.long,
            device=device,
        )
        finished = torch.zeros(sample_count, dtype=torch.bool, device=device)
        proposal_logprob = torch.zeros(sample_count, dtype=torch.float32, device=device)
        trajectories = [[] for _ in range(sample_count)]
        if policy_dfa is None:
            policy_dfa = getattr(self, "policy_dfa", None)
        dfa_states = (
            [policy_dfa.start_state for _ in range(sample_count)]
            if policy_dfa is not None
            else None
        )

        # REINFORCE does not differentiate through the categorical sample.  Do
        # rollout generation without autograd, otherwise a causal LM retains one
        # complete graph for every sampled prefix until the final optimizer step
        # (num_samples * max_steps model graphs for one training example).
        was_training = self.autoregressive_head.training
        self.autoregressive_head.eval()
        try:
            with torch.no_grad():
                for step in range(self.max_steps):
                    logits = AutoregressiveSequenceReinforcementProgram._prefix_logits(
                        self, text, prefixes
                    )[:, -1, :]
                    logits = AutoregressiveSequenceReinforcementProgram._mask_policy_logits(
                        self, logits, dfa_states, step, finished, policy_dfa=policy_dfa
                    )
                    distribution = torch.distributions.Categorical(logits=logits)
                    sampled = distribution.sample()
                    active = ~finished
                    proposal_logprob = proposal_logprob + torch.where(
                        active,
                        distribution.log_prob(sampled),
                        torch.zeros_like(proposal_logprob),
                    )
                    sampled = torch.where(
                        active, sampled, torch.full_like(sampled, self.eos_label)
                    )

                    sampled_values = sampled.cpu().tolist()
                    active_values = active.cpu().tolist()
                    for index, (label, is_active) in enumerate(
                        zip(sampled_values, active_values)
                    ):
                        if is_active:
                            trajectories[index].append(int(label))
                            if dfa_states is not None:
                                next_state = policy_dfa.step(dfa_states[index], int(label))
                                if next_state is None:
                                    raise RuntimeError(
                                        f"sampled label {label} has no policy DFA transition"
                                    )
                                dfa_states[index] = next_state

                    prefixes = torch.cat([prefixes, sampled.unsqueeze(-1)], dim=-1)
                    finished = finished | (sampled == self.eos_label)
                    if bool(finished.all()):
                        break

            # Re-score each sampled trajectory in one teacher-forced model pass
            # per rollout.  These are the only graphs needed by the policy loss.
            trajectory_logprob = (
                AutoregressiveSequenceReinforcementProgram._trajectory_logprob(
                    self, text, trajectories, device, policy_dfa=policy_dfa
                )
            )
        finally:
            self.autoregressive_head.train(was_training)

        return trajectories, trajectory_logprob, proposal_logprob.detach()

    def _mask_policy_logits(
        self, logits, dfa_states, step, finished=None, policy_dfa=None
    ):
        """Apply the graph-compiled DFA to one logit row per trajectory."""
        if policy_dfa is None:
            policy_dfa = getattr(self, "policy_dfa", None)
        if policy_dfa is None:
            return logits
        masked = torch.full_like(logits, float("-inf"))
        for index, state in enumerate(dfa_states):
            if finished is not None and bool(finished[index]):
                allowed = {self.eos_label}
            else:
                allowed = {
                    int(label)
                    for label in policy_dfa.allowed_tokens(
                        state, remaining_steps=self.max_steps - int(step)
                    )
                }
            if not allowed:
                raise RuntimeError(
                    f"policy DFA has no productive token at rollout step {step}, state={state!r}"
                )
            allowed_index = torch.tensor(sorted(allowed), dtype=torch.long, device=logits.device)
            masked[index, allowed_index] = logits[index, allowed_index]
        return masked

    def _prefix_logits(self, text, prefixes):
        sample_count = prefixes.shape[0]
        if getattr(self.autoregressive_head, "supports_batched_prefixes", False):
            logits = self.autoregressive_head.sequence_logits(text, prefixes)
            if logits.shape[0] != sample_count:
                raise ValueError("generator did not preserve the rollout batch")
            return logits
        # The GRU head carries a single-example encoder hidden state and cannot
        # broadcast it across rollout prefixes.
        return torch.cat(
            [
                self.autoregressive_head.sequence_logits(
                    text, prefixes[index : index + 1]
                )
                for index in range(sample_count)
            ],
            dim=0,
        )

    def _trajectory_logprob(self, text, trajectories, device, policy_dfa=None):
        lengths = torch.tensor(
            [len(trajectory) for trajectory in trajectories],
            dtype=torch.long,
            device=device,
        )
        max_length = int(lengths.max().item()) if lengths.numel() else 0
        if max_length == 0:
            raise ValueError("RL rollout produced no action tokens")

        targets = torch.full(
            (len(trajectories), max_length),
            self.eos_label,
            dtype=torch.long,
            device=device,
        )
        for index, trajectory in enumerate(trajectories):
            targets[index, : len(trajectory)] = torch.tensor(
                trajectory, dtype=torch.long, device=device
            )
        starts = torch.full(
            (len(trajectories), 1),
            self.eos_label,
            dtype=torch.long,
            device=device,
        )
        teacher_forced_prefixes = torch.cat([starts, targets[:, :-1]], dim=1)
        logits = AutoregressiveSequenceReinforcementProgram._prefix_logits(
            self, text, teacher_forced_prefixes
        )
        if policy_dfa is None:
            policy_dfa = getattr(self, "policy_dfa", None)
        if policy_dfa is not None:
            masked_steps = []
            states = [policy_dfa.start_state for _ in trajectories]
            for step in range(max_length):
                active = step < lengths
                step_logits = AutoregressiveSequenceReinforcementProgram._mask_policy_logits(
                    self,
                    logits[:, step, :],
                    states,
                    step,
                    ~active,
                    policy_dfa=policy_dfa,
                )
                masked_steps.append(step_logits)
                for index, trajectory in enumerate(trajectories):
                    if step >= len(trajectory):
                        continue
                    label = int(trajectory[step])
                    next_state = policy_dfa.step(states[index], label)
                    if next_state is None:
                        raise RuntimeError(
                            f"trajectory label {label} violates policy DFA at step {step}"
                        )
                    states[index] = next_state
            logits = torch.stack(masked_steps, dim=1)
        selected = torch.log_softmax(logits, dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        mask = torch.arange(max_length, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        return (selected * mask).sum(dim=-1)

    def reinforcement_loss(self, datanode, reward_fn, data_item):
        if reward_fn is None:
            return super().reinforcement_loss(datanode, reward_fn, data_item)

        policy_dfa = getattr(self, "policy_dfa", None)
        factory = getattr(self, "policy_dfa_factory", None)
        if factory is not None:
            policy_dfa = factory(data_item)
        trajectories, logprob, proposal_logprob = self._sample_trajectories(
            data_item.get("text", ""), policy_dfa=policy_dfa
        )
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
            loss = importance_weighted_loss(
                logprob, rewards, proposal_logprob=proposal_logprob
            )
        else:
            loss = reinforce_loss(logprob, rewards, baseline=self.baseline)
        if self.supervised_weight:
            anchor_loss = self.supervised_anchor_loss(data_item)
            if anchor_loss is not None:
                loss = loss + self.supervised_weight * anchor_loss
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
