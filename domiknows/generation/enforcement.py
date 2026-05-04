"""Runtime enforcement layer that bridges DomiKnowS constraints to generation.

This module connects the declarative constraint definitions (DFA and latent
formulas) to the actual generation pipeline.  It provides:

**Marking helpers** — annotate DomiKnowS logical constraint objects at
graph-build time to indicate how they should be enforced:

- :func:`mark_for_dfa`    — hard token-level enforcement via DFA masking.
- :func:`mark_for_latent` — soft enforcement via a differentiable window loss.
- :func:`mark_for_both`   — convenience wrapper for both at once.

**Discovery** — walk a built DomiKnowS graph and collect all annotated
constraints into a :class:`GenerationEnforcement` bundle:

- :func:`discover_generation_enforcement`

**Data structures**:

- :class:`LatentWindowSpec` — parameters for one soft window-formula loss term.
- :class:`GenerationEnforcement` — all hard DFA constraints + compiled soft
  loss callable.

Workflow overview::

    # 1. Build graph, mark constraints
    with Graph(...) as graph:
        lc = ifL(...)
        mark_for_dfa(lc, constraint=forbidden_token("bad"))
        mark_for_latent(lc, LatentWindowSpec(if_label=3, formula=my_formula, window=5))

    # 2. Discover enforcement bundle
    enforcement = discover_generation_enforcement(graph, bundle)

    # 3. Use at generation time
    combined_dfa = constraints_to_dfa(enforcement.dfa_constraints, vocabulary)
    loss = enforcement.latent_loss(token_probs)
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .constraints import GenerationConstraint
from .graph_discovery import discover_generation_constraints
from .latent_constraints import Formula, window_formula_loss


@dataclass(frozen=True)
class LatentWindowSpec:
    """Parameters for a single soft window-formula loss term.

    A *window formula loss* evaluates a propositional :class:`~.latent_constraints.Formula`
    over every sliding window of *window* consecutive token-probability vectors
    and returns a scalar penalty.  This dataclass bundles all hyperparameters
    needed to instantiate that computation.

    Attributes:
        if_label (int): Vocabulary label index that conditions the window;
            the loss is only accumulated for windows where this label is
            active (see :func:`~.latent_constraints.window_formula_loss`).
        formula (Formula): Propositional formula describing the desired
            relationship between token probabilities within a window.
        window (int): Width of the sliding window in tokens (≥ 1).
        weight (float): Scalar multiplier applied to the computed loss before
            summing with other terms.  Defaults to ``1.0``.
        reduction (str): How to reduce per-window losses: ``"mean"``,
            ``"sum"``, or ``"none"`` (return per-window tensor).
            Defaults to ``"mean"``.
    """

    if_label: int
    formula: Formula
    window: int
    weight: float = 1.0
    reduction: str = "mean"

    def __post_init__(self):
        """Validate *window* and *reduction* at construction time."""
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if self.reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")


@dataclass(frozen=True)
class GenerationEnforcement:
    """All enforcement information extracted from a built DomiKnowS graph.

    Produced by :func:`discover_generation_enforcement` and consumed at
    generation / training time.

    Attributes:
        dfa_constraints: Tuple of :class:`~.constraints.GenerationConstraint`
            objects collected from DFA-marked logical constraints.  Pass to
            :func:`~.constraints.constraints_to_dfa` to get a single
            combined DFA for hard token masking.
        latent_specs: Tuple of :class:`LatentWindowSpec` instances collected
            from latent-marked logical constraints.  Exposed for inspection;
            the compiled callable is in *latent_loss*.
        latent_loss: A callable ``(probs: Tensor) -> Tensor`` that sums the
            weighted window-formula losses for all *latent_specs* and returns
            a scalar.  Returns ``0.0`` when *latent_specs* is empty.
    """

    dfa_constraints: tuple[GenerationConstraint, ...]
    latent_specs: tuple[LatentWindowSpec, ...]
    latent_loss: Callable[[torch.Tensor], torch.Tensor]


def mark_for_dfa(lc, constraint: GenerationConstraint | None = None):
    """Annotate a DomiKnowS logical constraint for hard DFA enforcement.

    Attaches the ``_generation_dfa_constraint`` attribute to *lc* so that
    :func:`discover_generation_enforcement` will include it when building the
    DFA constraint tuple.  Call this at graph-build time, inside a
    ``with Graph(...):`` block.

    Args:
        lc: A DomiKnowS logical constraint object (e.g. the result of
            ``ifL(...)``, ``atMostAL(...)``, etc.).
        constraint: The :class:`~.constraints.GenerationConstraint` instance
            that captures the same property as *lc* but in DFA form.  When
            ``None``, the attribute is set to ``True`` as a plain presence
            flag and the constraint will be discovered via graph introspection
            by :func:`~.graph_discovery.discover_generation_constraints`.

    Returns:
        *lc* unchanged (for chaining).
    """
    setattr(lc, "_generation_dfa_constraint", constraint if constraint is not None else True)
    return lc


def mark_for_latent(lc, spec: LatentWindowSpec):
    """Annotate a DomiKnowS logical constraint for soft latent window loss.

    Appends *spec* to the ``_generation_latent_specs`` list on *lc* so that
    :func:`discover_generation_enforcement` collects it into
    :attr:`GenerationEnforcement.latent_specs`.  Multiple calls accumulate
    specs; they are all summed (weighted) by the compiled loss callable.

    Args:
        lc: A DomiKnowS logical constraint object.
        spec: A :class:`LatentWindowSpec` describing the window formula and
            its hyperparameters.

    Returns:
        *lc* unchanged (for chaining).

    Raises:
        TypeError: If *spec* is not a :class:`LatentWindowSpec` instance.
    """
    if not isinstance(spec, LatentWindowSpec):
        raise TypeError("spec must be a LatentWindowSpec")
    # Append to any existing specs already attached to this constraint.
    specs = list(getattr(lc, "_generation_latent_specs", ()))
    specs.append(spec)
    setattr(lc, "_generation_latent_specs", tuple(specs))
    return lc


def mark_for_both(
    lc,
    constraint: GenerationConstraint | None = None,
    spec: LatentWindowSpec | None = None,
):
    """Annotate a logical constraint for both DFA and optional latent loss.

    Convenience wrapper that calls :func:`mark_for_dfa` and, when *spec* is
    provided, :func:`mark_for_latent` in a single step.

    Args:
        lc: A DomiKnowS logical constraint object.
        constraint: Forwarded to :func:`mark_for_dfa`; see that function for
            semantics.
        spec: If not ``None``, forwarded to :func:`mark_for_latent`.

    Returns:
        *lc* unchanged (for chaining).
    """
    mark_for_dfa(lc, constraint=constraint)
    if spec is not None:
        mark_for_latent(lc, spec)
    return lc


def discover_generation_enforcement(
    graph,
    bundle,
    *,
    on_unsupported: str = "warn",
) -> GenerationEnforcement:
    """Walk a built DomiKnowS graph and collect all enforcement annotations.

    Combines the results of
    :func:`~.graph_discovery.discover_generation_constraints` (DFA) and the
    private :func:`_discover_latent_specs` (latent) passes into a single
    :class:`GenerationEnforcement` bundle that is ready for use at generation
    or training time.

    Args:
        graph: A DomiKnowS :class:`~domiknows.graph.Graph` instance that was
            built with :meth:`~.encoder.GenerationEncoder.build_graph` and
            optionally annotated with :func:`mark_for_dfa` /
            :func:`mark_for_latent`.
        bundle: The :class:`~.encoder.GenerationBundle` returned alongside
            *graph*.
        on_unsupported: Behaviour when a DFA-marked constraint cannot be
            automatically converted to a :class:`~.constraints.GenerationConstraint`.
            ``"warn"`` (default) emits a warning; ``"raise"`` raises;
            ``"ignore"`` silently skips.

    Returns:
        A :class:`GenerationEnforcement` with the collected DFA constraints,
        latent specs, and a compiled latent loss callable.
    """
    dfa_constraints = discover_generation_constraints(graph, bundle, on_unsupported=on_unsupported)
    latent_specs = _discover_latent_specs(graph)
    return GenerationEnforcement(
        dfa_constraints=dfa_constraints,
        latent_specs=latent_specs,
        latent_loss=_compile_latent_loss(latent_specs),
    )


def _discover_latent_specs(graph) -> tuple[LatentWindowSpec, ...]:
    """Collect all :class:`LatentWindowSpec` instances from a DomiKnowS graph.

    Iterates over head logical constraints (``headLC=True``) and gathers every
    :class:`LatentWindowSpec` stored in their ``_generation_latent_specs``
    attribute by :func:`mark_for_latent`.

    Args:
        graph: A built DomiKnowS graph with a ``logicalConstrains`` mapping.

    Returns:
        Tuple of all :class:`LatentWindowSpec` objects found, in iteration
        order.
    """
    specs: list[LatentWindowSpec] = []
    for lc in graph.logicalConstrains.values():
        # Only top-level (head) constraints carry enforcement annotations.
        if not getattr(lc, "headLC", True):
            continue
        specs.extend(getattr(lc, "_generation_latent_specs", ()))
    return tuple(specs)


def _compile_latent_loss(specs: tuple[LatentWindowSpec, ...]):
    """Compile a list of :class:`LatentWindowSpec` objects into a loss callable.

    Returns a closure that, given a token-probability tensor, computes the
    weighted sum of all window-formula losses and returns a scalar tensor.

    Args:
        specs: Tuple of :class:`LatentWindowSpec` objects to include.

    Returns:
        A callable ``latent_loss(probs: Tensor) -> Tensor`` where *probs* is
        a float tensor of shape ``(T, num_labels)`` and the return value is a
        scalar.  Returns ``probs.new_zeros(())`` when *specs* is empty so
        gradients still flow correctly in a training loop.
    """
    def latent_loss(probs: torch.Tensor) -> torch.Tensor:
        # Coerce non-tensor inputs (e.g. numpy arrays) for flexibility.
        if not isinstance(probs, torch.Tensor):
            probs = torch.as_tensor(probs, dtype=torch.float32)
        # Return a zero scalar on the same device when there are no specs.
        if not specs:
            return probs.new_zeros(())

        total = None
        for spec in specs:
            # Compute per-spec loss and apply the weight multiplier.
            loss = window_formula_loss(
                probs,
                if_label=spec.if_label,
                formula=spec.formula,
                window=spec.window,
                reduction=spec.reduction,
            ) * float(spec.weight)
            # Accumulate: first spec initialises total, subsequent ones add.
            total = loss if total is None else total + loss
        return total

    return latent_loss
