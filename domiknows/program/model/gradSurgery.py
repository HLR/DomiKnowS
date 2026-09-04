"""R5 Phase C — gradient surgery between the supervised and constraint losses.

Why this exists *now* and not earlier
-------------------------------------
With one isolated ``ModuleLearner`` per concept over a frozen encoder, the
supervised and constraint gradients touch almost disjoint parameters, so there
is nothing to conflict and surgery would be dead weight. R4's shared trunk puts
them in the **same** parameters, where they can point in opposing directions and
partially cancel — and neither loss reports it: both totals fall while the
shared parameters receive a near-zero resultant.

Diagnostic first, deliberately
------------------------------
Resolving conflict costs a second backward pass on every step. That is only
worth paying if conflict actually occurs, so :func:`conflict_report` is usable
on its own (``grad_surgery='diagnose'``) and is what should decide whether to
turn a resolver on. If the measured conflict rate is ~0 for a task, the right
answer is to leave surgery off rather than ship a cost that buys nothing.

What counts as "shared"
-----------------------
Rather than requiring the caller to name the trunk, a parameter is treated as
shared exactly when **both** losses produce a gradient for it. That is the
definition that matters — a parameter only one loss reaches cannot conflict —
and it needs no knowledge of the architecture.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

#: Selectable strategies. ``'diagnose'`` measures without changing the update.
GRAD_SURGERY = ('none', 'diagnose', 'pcgrad', 'cagrad')


@dataclass
class ConflictStats:
    """Running conflict measurement between two loss gradients."""

    steps: int = 0
    conflicts: int = 0
    cosine_sum: float = 0.0
    shared_params: int = 0

    @property
    def conflict_rate(self) -> float:
        return float('nan') if not self.steps else self.conflicts / self.steps

    @property
    def mean_cosine(self) -> float:
        return float('nan') if not self.steps else self.cosine_sum / self.steps

    def record(self, cosine: float, shared: int):
        self.steps += 1
        self.shared_params = shared
        self.cosine_sum += cosine
        if cosine < 0:
            self.conflicts += 1

    def render(self) -> str:
        if not self.steps:
            return 'gradient conflict: no measurement taken'
        return (f'gradient conflict: rate={self.conflict_rate:.3f} '
                f'(mean cos={self.mean_cosine:+.4f}) over {self.steps} step(s), '
                f'{self.shared_params} shared parameter tensor(s)')


def _flatten(grads: Sequence[Optional[torch.Tensor]]) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads])


def cosine(g_a: torch.Tensor, g_b: torch.Tensor) -> float:
    """Cosine between two flattened gradients; 0.0 when either vanishes."""
    na, nb = g_a.norm(), g_b.norm()
    if float(na) == 0.0 or float(nb) == 0.0:
        return 0.0
    return float(torch.dot(g_a, g_b) / (na * nb))


def pcgrad(g_a: torch.Tensor, g_b: torch.Tensor) -> torch.Tensor:
    """PCGrad (Yu et al. 2020) for two objectives.

    When the gradients conflict, project each out of the *other's* conflicting
    direction before summing, so neither can cancel the component of the other
    that it disagrees with. Non-conflicting gradients are summed unchanged, so
    this is a no-op exactly when there is nothing to fix.
    """
    inner = torch.dot(g_a, g_b)
    if float(inner) >= 0:
        return g_a + g_b
    a_sq, b_sq = g_a.dot(g_a), g_b.dot(g_b)
    projected_a = g_a - (inner / b_sq.clamp_min(1e-12)) * g_b
    projected_b = g_b - (inner / a_sq.clamp_min(1e-12)) * g_a
    return projected_a + projected_b


def cagrad(g_a: torch.Tensor, g_b: torch.Tensor, c: float = 0.5,
           grid: int = 101) -> torch.Tensor:
    """CAGrad (Liu et al. 2021) for two objectives.

    Seeks an update that stays within a ``c``-ball of the average gradient while
    maximising the *worst-case* per-objective improvement — so unlike a plain
    sum it cannot let one objective be dominated. The mixing weight is scalar
    for two tasks and every term reduces to the three dot products, so a grid
    search over ``w`` is exact enough and costs nothing next to a backward pass.
    """
    a = float(g_a.dot(g_a))
    b = float(g_a.dot(g_b))
    d = float(g_b.dot(g_b))

    g0_sq = 0.25 * (a + 2 * b + d)              # ||(g_a+g_b)/2||^2
    if g0_sq <= 0:
        return g_a + g_b
    phi = (c ** 2) * g0_sq

    best_w, best_value = 0.5, None
    for i in range(grid):
        w = i / (grid - 1)
        gw_sq = w * w * a + 2 * w * (1 - w) * b + (1 - w) * (1 - w) * d
        if gw_sq <= 0:
            continue
        gw_g0 = 0.5 * (w * (a + b) + (1 - w) * (b + d))
        value = gw_g0 + (phi ** 0.5) * (gw_sq ** 0.5)
        if best_value is None or value < best_value:
            best_value, best_w = value, w

    g_w = best_w * g_a + (1 - best_w) * g_b
    norm = float(g_w.norm())
    if norm == 0.0:
        return g_a + g_b
    # d = g0 + (sqrt(phi)/||g_w||) g_w, rescaled to the summed-gradient scale so
    # the effective learning rate matches the untouched path.
    direction = 0.5 * (g_a + g_b) + ((phi ** 0.5) / norm) * g_w
    return direction * 2.0


def _resolve(method: str, g_a: torch.Tensor, g_b: torch.Tensor,
             cagrad_c: float) -> torch.Tensor:
    if method == 'pcgrad':
        return pcgrad(g_a, g_b)
    if method == 'cagrad':
        return cagrad(g_a, g_b, c=cagrad_c)
    return g_a + g_b


def conflict_report(model_loss: torch.Tensor, constraint_loss: torch.Tensor,
                    parameters: Sequence[torch.nn.Parameter],
                    method: str = 'diagnose', stats: Optional[ConflictStats] = None,
                    cagrad_c: float = 0.5) -> Optional[float]:
    """Measure (and optionally resolve) conflict, writing the result to ``.grad``.

    The two gradients are taken **separately** — the single fused
    ``(mloss + beta*closs).backward()`` cannot be decomposed after the fact —
    which is the real cost of this pass and why it is opt-in.

    Parameters reached by only one loss keep that loss's gradient untouched:
    they cannot conflict, so there is nothing to resolve for them.

    Returns the cosine between the two gradients over the shared parameters, or
    None when they share none.
    """
    if method not in GRAD_SURGERY:
        raise ValueError(f'grad_surgery must be one of {GRAD_SURGERY}')

    params = [p for p in parameters if p.requires_grad]
    if not params:
        return None

    grads_a = torch.autograd.grad(model_loss, params, retain_graph=True,
                                  allow_unused=True)
    grads_b = torch.autograd.grad(constraint_loss, params, retain_graph=True,
                                  allow_unused=True)

    shared = [i for i, (a, b) in enumerate(zip(grads_a, grads_b))
              if a is not None and b is not None]

    # Parameters only one loss reaches: accumulate that gradient as-is.
    for i, (param, a, b) in enumerate(zip(params, grads_a, grads_b)):
        if i in shared:
            continue
        contribution = a if a is not None else b
        if contribution is None:
            continue
        param.grad = contribution if param.grad is None else param.grad + contribution

    if not shared:
        return None

    flat_a = _flatten([grads_a[i] for i in shared])
    flat_b = _flatten([grads_b[i] for i in shared])
    similarity = cosine(flat_a, flat_b)
    if stats is not None:
        stats.record(similarity, len(shared))

    combined = _resolve(method, flat_a, flat_b, cagrad_c)

    offset = 0
    for i in shared:
        param = params[i]
        size = param.numel()
        piece = combined[offset:offset + size].view_as(param)
        offset += size
        param.grad = piece if param.grad is None else param.grad + piece

    return similarity
