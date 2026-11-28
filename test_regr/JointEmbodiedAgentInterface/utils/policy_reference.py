"""A frozen Stage-1 policy without duplicating the quantized VLM backbone.

Only optimizer-owned floating-point parameters are stored, on CPU. Reference
forwards run before the differentiable replay; swapping tensor storage avoids
in-place writes/version changes to parameters used by autograd.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math

import torch


@dataclass
class PolicyReplay:
    logprob: torch.Tensor
    # Normalized log probabilities over the same allowed labels at each prefix.
    log_probs: tuple[torch.Tensor, ...]

    def kl_from(self, reference):
        """Mean token KL(reference || current), including unchosen labels."""
        if len(self.log_probs) != len(reference.log_probs):
            raise ValueError("reference replay length differs")
        terms = []
        for current, frozen in zip(self.log_probs, reference.log_probs):
            if current.shape != frozen.shape:
                raise ValueError("reference replay support differs")
            frozen = frozen.detach().to(current)
            terms.append((frozen.exp() * (frozen - current)).sum())
        return torch.stack(terms).mean().clamp_min(0) if terms else self.logprob * 0


@contextmanager
def deterministic_policy(model, *, inference=False, training=False):
    """Disable dropout while preserving training-mode activation checkpointing.

    Call again during backward so checkpointed layers recompute the same policy.
    Reference forwards alone may enter inference mode (they have no live graph).
    """
    if inference and training:
        raise ValueError("policy scope must choose inference or training")
    modes = [(module, module.training) for module in model.modules()]
    attention_dropout = []
    if inference:
        model.eval()
    elif training:
        model.train()
    for module, _ in modes:
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            module.training = False
        if isinstance(getattr(module, "attention_dropout", None), (float, int)):
            attention_dropout.append((module, module.attention_dropout))
            module.attention_dropout = 0.0
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training
        for module, dropout in attention_dropout:
            module.attention_dropout = dropout


def reference_fingerprint(parameters):
    digest = hashlib.sha256()
    for name, tensor in sorted(parameters.items()):
        digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}".encode())
        digest.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FrozenPolicyReference:
    def __init__(self, planner, optimizer, *, state=None):
        owned = {id(p) for group in optimizer.param_groups for p in group["params"]}
        parameters = dict(planner.named_parameters())
        expected = {name: p for name, p in parameters.items() if id(p) in owned and p.requires_grad}
        if not expected:
            raise ValueError("frozen planner reference has no trainable parameters")
        if state is None:
            self.parameters = {name: p.detach().cpu().clone() for name, p in expected.items()}
        else:
            if state.get("version") != 1 or set(state.get("parameters", {})) != set(expected):
                raise ValueError("frozen planner reference parameter names differ")
            self.parameters = {name: p.detach().cpu().clone() for name, p in state["parameters"].items()}
            for name, tensor in self.parameters.items():
                if tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
                    raise ValueError("frozen planner reference parameter shape/dtype differs")
        self.fingerprint = reference_fingerprint(self.parameters)
        if state is not None and self.fingerprint != state.get("fingerprint"):
            raise ValueError("frozen planner reference fingerprint differs")
        self._active = False

    def state_dict(self):
        return {"version": 1, "fingerprint": self.fingerprint, "parameters": self.parameters}

    @contextmanager
    def activate(self, planner):
        if self._active:
            raise RuntimeError("nested frozen policy activation")
        self._active = True
        originals = []
        try:
            with torch.no_grad(), deterministic_policy(planner, inference=True):
                for name, parameter in planner.named_parameters():
                    if name in self.parameters:
                        originals.append((parameter, parameter.data))
                        parameter.data = self.parameters[name].to(device=parameter.device, copy=True)
                yield
        finally:
            for parameter, original in originals:
                parameter.data = original
            self._active = False


def regularized_policy_replay(planner, reference, domain, context, labels, dfa, advantage, **stopping):
    with reference.activate(planner):
        frozen = planner.replay_labels_policy(domain, context, labels, dfa, **stopping)
    with deterministic_policy(planner):
        current = planner.replay_labels_policy(domain, context, labels, dfa, **stopping)
    policy_loss = -current.logprob * advantage.to(current.logprob) / max(1, len(current.log_probs))
    return policy_loss, current.kl_from(frozen)


def resolve_regularization(args, resume_payload=None):
    """Keep historical resumes exact; never recapture an RL policy as CE."""
    saved = (resume_payload or {}).get("stage2_planner_regularization")
    if saved is not None and saved.get("version") != 1:
        raise ValueError("unsupported Stage 2 planner regularization version")
    resuming = (resume_payload or {}).get("stage") == "stage2"
    weight = getattr(args, "stage2_planner_kl_weight", None)
    reward = getattr(args, "stage2_planner_reward", None)
    replay = getattr(args, "stage2_reference_replay_per_update", None)
    defaults = (saved["kl_weight"], saved["reward_policy"], saved.get("replay_per_update", 0)) if saved else (
        (0.0, "dense", 0) if resuming else (0.05, "success-gated", 2))
    weight = defaults[0] if weight is None else float(weight)
    reward = defaults[1] if reward is None else reward
    replay = (defaults[2] if weight else 0) if replay is None else int(replay)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("Stage 2 planner KL weight must be finite and nonnegative")
    if replay < 0 or (replay and not weight):
        raise ValueError("reference replay requires a positive KL weight and nonnegative count")
    if resuming and (weight, reward, replay) != defaults:
        raise ValueError("Stage 2 planner regularization differs from checkpoint; start from the CE checkpoint")
    if saved and weight and not saved.get("reference"):
        raise ValueError("Stage 2 checkpoint is missing its frozen CE reference")
    return weight, reward, replay, saved.get("reference") if saved else None


def balanced_reference_example(examples, cursor):
    """Cycle task groups before reusing a group's next training observation."""
    groups = {}
    for example in examples:
        task = example.get("task", "") if isinstance(example, dict) else getattr(example, "task", "")
        groups.setdefault(str(task), []).append(example)
    keys = sorted(groups)
    if not keys:
        raise ValueError("reference replay requires training observations")
    group = groups[keys[cursor % len(keys)]]
    return group[(cursor // len(keys)) % len(group)]
