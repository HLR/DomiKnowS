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
- :class:`GenerationEnforcement` — compiled hard DFA + compiled soft
  loss callable.

Workflow overview::

    # 1. Build graph, mark constraints
    with Graph(...) as graph:
        lc = ifL(...)
        mark_for_dfa(lc)
        mark_for_latent(lc, LatentWindowSpec(if_label=3, formula=my_formula, window=5))

    # 2. Discover enforcement bundle
    enforcement = discover_generation_enforcement(graph, bundle)

    # 3. Use at generation time
    combined_dfa = enforcement.dfa
    loss = enforcement.latent_loss(token_probs)
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias
import warnings

import torch

from ..dfa.core import DFA
from ..dfa.graph_discovery import constraints_to_dfa_from_graph
from .constraints import Formula, LabelRef, LatentLossBreakdown, evaluate_latent_loss, window_formula_loss
from .potentials import LatentTransitionPotential, combine_transition_potentials, forbid_hmm_transition


GraphLatentCompiler: TypeAlias = Callable[[object, object], "GraphLatentCompilerResult | None"]


@dataclass(frozen=True)
class LatentWindowSpec:
    """Parameters for a single soft window-formula loss term.

    A *window formula loss* evaluates a propositional :class:`~.constraints.Formula`
    over every sliding window of *window* consecutive token-probability vectors
    and returns a scalar penalty.  This dataclass bundles all hyperparameters
    needed to instantiate that computation.

    Attributes:
        if_label (int): Vocabulary label index that conditions the window;
            the loss is only accumulated for windows where this label is
            active (see :func:`~.constraints.window_formula_loss`).
        formula (Formula): Propositional formula describing the desired
            relationship between token probabilities within a window.
        window (int): Width of the sliding window in tokens (≥ 1).
        weight (float): Scalar multiplier applied to the computed loss before
            summing with other terms.  Defaults to ``1.0``.
        reduction (str): How to reduce per-window losses: ``"mean"``,
            ``"sum"``, or ``"none"`` (return per-window tensor).
            Defaults to ``"mean"``.
    """

    if_label: int | LabelRef
    formula: Formula
    window: int
    weight: float = 1.0
    reduction: str = "mean"
    name: str | None = None
    concept: str = "generated_token"
    mask_policy: str = "lengths"
    empty_window_policy: str = "penalize"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """Validate *window* and *reduction* at construction time."""
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if self.reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        if self.empty_window_policy not in {"penalize", "ignore"}:
            raise ValueError("empty_window_policy must be 'penalize' or 'ignore'")


@dataclass(frozen=True)
class GenerationEnforcement:
    """All enforcement information extracted from a built DomiKnowS graph.

    Produced by :func:`discover_generation_enforcement` and consumed at
    generation / training time.

    Attributes:
        dfa: Combined hard-enforcement DFA compiled from supported
            DomiKnowS graph constraints.
        latent_specs: Tuple of :class:`LatentWindowSpec` instances collected
            from latent-marked logical constraints.  Exposed for inspection;
            the compiled callable is in *latent_loss*.
        latent_loss: A callable ``(probs: Tensor) -> Tensor`` that sums the
            weighted window-formula losses for all *latent_specs* and returns
            a scalar.  Returns ``0.0`` when *latent_specs* is empty.
    """

    dfa: DFA
    latent_specs: tuple[LatentWindowSpec, ...]
    latent_loss: Callable[..., torch.Tensor]
    latent_breakdown: Callable[..., LatentLossBreakdown]
    transition_potentials: tuple[LatentTransitionPotential, ...] = ()


@dataclass(frozen=True)
class GraphLatentCompilerResult:
    """Result returned by a custom graph-to-latent compiler hook.

    Custom compilers inspect one DomiKnowS logical constraint and may emit
    latent window specs, transition potentials, or an unsupported reason.
    """

    latent_specs: tuple[LatentWindowSpec, ...] = ()
    transition_potentials: tuple[LatentTransitionPotential, ...] = ()
    relevant: bool = False
    supported: bool = False
    reason: str | None = None
    compiler_name: str | None = None


def graph_latent_compiler_result(
    *,
    latent_specs: LatentWindowSpec | Sequence[LatentWindowSpec] = (),
    transition_potentials: LatentTransitionPotential | Sequence[LatentTransitionPotential] = (),
    relevant: bool | None = None,
    supported: bool | None = None,
    reason: str | None = None,
    compiler_name: str | None = None,
) -> GraphLatentCompilerResult:
    """Create a normalized :class:`GraphLatentCompilerResult` for custom hooks."""

    specs = _as_tuple(latent_specs, LatentWindowSpec, "latent_specs")
    potentials = _as_tuple(transition_potentials, LatentTransitionPotential, "transition_potentials")
    if supported is None:
        supported = bool(specs or potentials)
    if relevant is None:
        relevant = bool(supported or reason)
    return GraphLatentCompilerResult(
        latent_specs=specs,
        transition_potentials=potentials,
        relevant=bool(relevant),
        supported=bool(supported),
        reason=reason,
        compiler_name=compiler_name,
    )


def mark_for_dfa(lc):
    """Annotate a DomiKnowS logical constraint for hard DFA enforcement.

    Attaches the ``_generation_dfa_constraint`` flag to *lc* so that
    :func:`discover_generation_enforcement` will include it when compiling the
    hard DFA.  Call this at graph-build time, inside a
    ``with Graph(...):`` block.

    Args:
        lc: A DomiKnowS logical constraint object (e.g. the result of
            ``ifL(...)``, ``atMostAL(...)``, etc.).
    Returns:
        *lc* unchanged (for chaining).
    """
    setattr(lc, "_generation_dfa_constraint", True)
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
    spec: LatentWindowSpec | None = None,
):
    """Annotate a logical constraint for both DFA and optional latent loss.

    Convenience wrapper that calls :func:`mark_for_dfa` and, when *spec* is
    provided, :func:`mark_for_latent` in a single step.

    Args:
        lc: A DomiKnowS logical constraint object.
        spec: If not ``None``, forwarded to :func:`mark_for_latent`.

    Returns:
        *lc* unchanged (for chaining).
    """
    mark_for_dfa(lc)
    if spec is not None:
        mark_for_latent(lc, spec)
    return lc


def discover_generation_enforcement(
    graph,
    bundle,
    *,
    on_unsupported: str = "warn",
    latent_mode: str = "marked",
    extra_latent_compilers: Sequence[GraphLatentCompiler] = (),
) -> GenerationEnforcement:
    """Walk a built DomiKnowS graph and collect all enforcement annotations.

    Combines the graph-only DFA compiler with the private latent discovery
    passes into a single :class:`GenerationEnforcement` bundle that is ready
    for use at generation or training time.

    Args:
        graph: A DomiKnowS :class:`~domiknows.graph.Graph` instance that was
            built with :meth:`~.dfa.encoder.GenerationEncoder.build_graph` and
            optionally annotated with :func:`mark_for_dfa` /
            :func:`mark_for_latent`.
        bundle: The :class:`~.dfa.encoder.GenerationBundle` returned alongside
            *graph*.
        on_unsupported: Behaviour when a generation-relevant graph constraint
            cannot be compiled to DFA.
            ``"warn"`` (default) emits a warning; ``"raise"`` raises;
            ``"ignore"`` silently skips.

    Returns:
        A :class:`GenerationEnforcement` with the compiled DFA, latent specs,
        and a compiled latent loss callable.
    """
    dfa = constraints_to_dfa_from_graph(graph, bundle, on_unsupported=on_unsupported)
    compiler_results = _run_custom_latent_compilers(
        graph,
        bundle,
        extra_latent_compilers,
        on_unsupported=on_unsupported,
        skip_builtin_supported=True,
    )
    latent_specs = discover_latent_window_specs(
        graph,
        bundle,
        mode=latent_mode,
        on_unsupported=on_unsupported,
        extra_compilers=extra_latent_compilers,
        _compiler_results=compiler_results,
    )
    transition_potentials = discover_transition_potentials(
        graph,
        bundle,
        on_unsupported=on_unsupported,
        extra_compilers=extra_latent_compilers,
        _compiler_results=compiler_results,
    )
    return GenerationEnforcement(
        dfa=dfa,
        latent_specs=latent_specs,
        latent_loss=_compile_latent_loss(latent_specs),
        latent_breakdown=_compile_latent_breakdown(latent_specs),
        transition_potentials=transition_potentials,
    )


def discover_latent_window_specs(
    graph,
    bundle=None,
    *,
    mode: str = "marked",
    on_unsupported: str = "warn",
    extra_compilers: Sequence[GraphLatentCompiler] = (),
    _compiler_results: Sequence[GraphLatentCompilerResult] | None = None,
) -> tuple[LatentWindowSpec, ...]:
    """Discover soft latent window specs from graph annotations or safe patterns.

    ``mode="marked"`` preserves the original behavior.  ``"auto"`` compiles
    adjacent ``is_next_rel`` rules over known generation/HMM/WFA enum concepts.
    ``"marked_and_auto"`` returns both, deduplicated by normalized spec fields.
    """
    mode = str(mode).lower().replace("-", "_")
    if mode not in {"marked", "auto", "marked_and_auto"}:
        raise ValueError("mode must be 'marked', 'auto', or 'marked_and_auto'")
    if on_unsupported not in {"ignore", "warn", "error"}:
        raise ValueError("on_unsupported must be 'ignore', 'warn', or 'error'")
    extra_compilers = tuple(extra_compilers)

    specs: list[LatentWindowSpec] = []
    seen: set = set()
    if mode in {"marked", "marked_and_auto"}:
        for spec in _discover_marked_latent_specs(graph):
            _append_unique_spec(specs, seen, spec)
    if mode in {"auto", "marked_and_auto"}:
        if bundle is None:
            raise ValueError("bundle is required for auto latent discovery")
        for lc_name, lc in graph.logicalConstrains.items():
            if not getattr(lc, "headLC", True):
                continue
            spec = _auto_latent_spec_from_lc(lc, bundle)
            if spec is not None:
                _append_unique_spec(specs, seen, spec)
            elif _is_latent_relevant(lc, bundle) and not getattr(lc, "_generation_latent_specs", ()):
                _handle_latent_unsupported(lc_name, lc, on_unsupported)
    compiler_results = (
        tuple(_compiler_results)
        if _compiler_results is not None
        else _run_custom_latent_compilers(
            graph,
            bundle,
            extra_compilers,
            on_unsupported=on_unsupported,
            skip_builtin_supported=True,
        )
    )
    for result in compiler_results:
        for spec in result.latent_specs:
            _append_unique_spec(specs, seen, spec)
    return tuple(specs)


def _discover_marked_latent_specs(graph) -> tuple[LatentWindowSpec, ...]:
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
    def latent_loss(probs: torch.Tensor | Mapping[str, torch.Tensor], **kwargs) -> torch.Tensor:
        # Coerce non-tensor inputs (e.g. numpy arrays) for flexibility.
        if not isinstance(probs, torch.Tensor) and not isinstance(probs, Mapping):
            probs = torch.as_tensor(probs, dtype=torch.float32)
        # Return a zero scalar on the same device when there are no specs.
        if not specs:
            reference = next(iter(probs.values())) if isinstance(probs, Mapping) else probs
            if not isinstance(reference, torch.Tensor):
                reference = torch.as_tensor(reference, dtype=torch.float32)
            return reference.new_zeros(())

        return evaluate_latent_loss(specs, probs, **kwargs).total

    return latent_loss


def _compile_latent_breakdown(specs: tuple[LatentWindowSpec, ...]):
    def latent_breakdown(probs: torch.Tensor | Mapping[str, torch.Tensor], **kwargs) -> LatentLossBreakdown:
        if not isinstance(probs, torch.Tensor) and not isinstance(probs, Mapping):
            probs = torch.as_tensor(probs, dtype=torch.float32)
        return evaluate_latent_loss(specs, probs, **kwargs)

    return latent_breakdown


def discover_transition_potentials(
    graph,
    bundle,
    *,
    strength: float = 0.0,
    on_unsupported: str = "warn",
    extra_compilers: Sequence[GraphLatentCompiler] = (),
    _compiler_results: Sequence[GraphLatentCompilerResult] | None = None,
) -> tuple[LatentTransitionPotential, ...]:
    """Discover adjacent latent-state transition potentials from graph rules.

    V1 recognizes rules shaped like ``state_i(t) => not state_j(t+1)`` under
    the bundle's ``is_next_rel`` relation.  The returned HMM-style potentials
    also broadcast over WFA transition tensors.
    """
    if on_unsupported not in {"ignore", "warn", "error"}:
        raise ValueError("on_unsupported must be 'ignore', 'warn', or 'error'")
    potentials: list[LatentTransitionPotential] = []
    seen: set = set()
    for lc_name, lc in graph.logicalConstrains.items():
        if not getattr(lc, "headLC", True):
            continue
        match = _transition_potential_from_lc(lc, bundle, strength=strength)
        if match is None:
            continue
        _append_unique_potential(potentials, seen, match, compiler_name="builtin", strength=strength)
    compiler_results = (
        tuple(_compiler_results)
        if _compiler_results is not None
        else _run_custom_latent_compilers(
            graph,
            bundle,
            extra_compilers,
            on_unsupported=on_unsupported,
            skip_builtin_supported=True,
        )
    )
    for result in compiler_results:
        for potential in result.transition_potentials:
            _append_unique_potential(potentials, seen, potential, compiler_name=result.compiler_name)
    return tuple(potentials)


def combined_transition_potential_from_graph(
    graph,
    bundle,
    *,
    strength: float = 0.0,
    on_unsupported: str = "warn",
    extra_compilers: Sequence[GraphLatentCompiler] = (),
) -> LatentTransitionPotential | None:
    """Discover and combine graph transition potentials in one call."""
    return combine_transition_potentials(
        discover_transition_potentials(
            graph,
            bundle,
            strength=strength,
            on_unsupported=on_unsupported,
            extra_compilers=extra_compilers,
        )
    )


def _append_unique_spec(specs: list[LatentWindowSpec], seen: set, spec: LatentWindowSpec) -> None:
    key = (
        spec.name,
        spec.concept,
        _label_key(spec.if_label),
        _formula_key(spec.formula),
        spec.window,
        spec.weight,
        spec.reduction,
        spec.empty_window_policy,
    )
    if key not in seen:
        seen.add(key)
        specs.append(spec)


def _append_unique_potential(
    potentials: list[LatentTransitionPotential],
    seen: set,
    potential: LatentTransitionPotential,
    *,
    compiler_name: str | None = None,
    strength: float | None = None,
) -> None:
    key = _potential_key(potential, compiler_name=compiler_name, strength=strength)
    object_key = ("object_id", id(potential))
    if key in seen or object_key in seen:
        return
    seen.add(key)
    seen.add(object_key)
    potentials.append(potential)


def _potential_key(
    potential: LatentTransitionPotential,
    *,
    compiler_name: str | None = None,
    strength: float | None = None,
):
    values = potential.values
    if callable(values):
        value_key = ("callable", id(values))
    else:
        try:
            tensor = torch.as_tensor(values)
            value_key = (
                "tensor",
                tuple(tensor.shape),
                tuple(float(item) for item in tensor.detach().cpu().reshape(-1).tolist()),
            )
        except Exception:
            try:
                value_key = ("repr", repr(values))
            except Exception:
                value_key = ("object", id(values))
    return (potential.name, potential.log_space, value_key, compiler_name, strength)


def _as_tuple(value, expected_type: type, name: str) -> tuple:
    if value is None:
        return ()
    if isinstance(value, expected_type):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = tuple(value)
        invalid = [item for item in items if not isinstance(item, expected_type)]
        if invalid:
            raise TypeError(f"{name} must contain only {expected_type.__name__} instances")
        return items
    raise TypeError(f"{name} must be a {expected_type.__name__} or a sequence of them")


def _run_custom_latent_compilers(
    graph,
    bundle,
    extra_compilers: Sequence[GraphLatentCompiler],
    *,
    on_unsupported: str,
    skip_builtin_supported: bool,
) -> tuple[GraphLatentCompilerResult, ...]:
    if on_unsupported not in {"ignore", "warn", "error"}:
        raise ValueError("on_unsupported must be 'ignore', 'warn', or 'error'")
    compilers = tuple(extra_compilers or ())
    if not compilers:
        return ()
    if bundle is None:
        raise ValueError("bundle is required when extra latent compilers are supplied")

    results: list[GraphLatentCompilerResult] = []
    for lc_name, lc in graph.logicalConstrains.items():
        if not getattr(lc, "headLC", True):
            continue
        if skip_builtin_supported and _latent_lc_has_builtin_support(lc, bundle):
            continue

        unsupported: list[GraphLatentCompilerResult] = []
        for compiler in compilers:
            result = compiler(lc, bundle)
            if result is None:
                continue
            if not isinstance(result, GraphLatentCompilerResult):
                compiler_name = getattr(compiler, "__name__", compiler.__class__.__name__)
                raise TypeError(
                    f"graph latent compiler {compiler_name!r} must return "
                    "GraphLatentCompilerResult or None"
                )
            if result.supported:
                results.append(result)
                unsupported = []
                break
            if result.relevant:
                unsupported.append(result)

        for result in unsupported:
            _handle_custom_latent_unsupported(lc_name, lc, result, on_unsupported)
    return tuple(results)


def _latent_lc_has_builtin_support(lc, bundle) -> bool:
    if getattr(lc, "_generation_latent_specs", ()):
        return True
    return (
        _auto_latent_spec_from_lc(lc, bundle) is not None
        or _transition_potential_from_lc(lc, bundle, strength=0.0) is not None
    )


def _handle_custom_latent_unsupported(
    lc_name: str,
    lc,
    result: GraphLatentCompilerResult,
    on_unsupported: str,
) -> None:
    compiler = result.compiler_name or "custom graph-to-latent compiler"
    reason = result.reason or "no supported latent output"
    message = (
        f"{compiler} reported DomiKnowS logical constraint {lc_name} "
        f"({lc.__class__.__name__}) as relevant but unsupported: {reason}"
    )
    if on_unsupported == "error":
        raise ValueError(message)
    if on_unsupported == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def _label_key(label):
    if isinstance(label, LabelRef):
        return ("ref", label.concept, label.label)
    return ("label", int(label))


def _formula_key(formula):
    if isinstance(formula, LabelRef):
        return ("ref", formula.concept, formula.label)
    if isinstance(formula, int):
        return ("label", int(formula))
    if isinstance(formula, tuple):
        return tuple(_formula_key(item) if index else item for index, item in enumerate(formula))
    return repr(formula)


def _auto_latent_spec_from_lc(lc, bundle) -> LatentWindowSpec | None:
    if getattr(lc, "__class__", None).__name__ != "ifL":
        return None
    relation_var = _relation_variable(lc, bundle)
    if relation_var is None:
        return None
    nested = next((item for item in getattr(lc, "e", ()) if getattr(item, "__class__", None).__name__ == "ifL"), None)
    if nested is None:
        return None
    antecedent = _path_label_ref_from_expr(nested, bundle, relation_var, getattr(bundle, "current_token", None))
    if antecedent is None:
        return None
    formula = _formula_from_expr(nested, bundle, relation_var, getattr(bundle, "next_token", None))
    if formula is None:
        return None
    return LatentWindowSpec(
        if_label=antecedent,
        formula=formula,
        window=1,
        concept=antecedent.concept,
        name=f"auto_{antecedent.concept}_{antecedent.label}_next",
        empty_window_policy="ignore",
        metadata={"source": "graph_auto"},
    )


def _transition_potential_from_lc(lc, bundle, *, strength: float) -> LatentTransitionPotential | None:
    if getattr(lc, "__class__", None).__name__ != "ifL":
        return None
    relation_var = _relation_variable(lc, bundle)
    if relation_var is None:
        return None
    nested = next((item for item in getattr(lc, "e", ()) if getattr(item, "__class__", None).__name__ == "ifL"), None)
    if nested is None or len(getattr(nested, "e", ())) < 2:
        return None
    current_role = getattr(bundle, "current_token", None)
    next_role = getattr(bundle, "next_token", None)
    antecedent = _path_label_ref_from_expr(nested, bundle, relation_var, current_role)
    consequent = nested.e[-1]
    if getattr(consequent, "__class__", None).__name__ != "notL":
        return None
    blocked = _path_label_ref_from_expr(consequent, bundle, relation_var, next_role)
    if antecedent is None or blocked is None or antecedent.concept != blocked.concept:
        return None
    if antecedent.concept not in {"latent_state", "wfa_state"}:
        return None
    state_count = len(getattr(bundle, "state_names", ()))
    if not state_count:
        return None
    potential = forbid_hmm_transition(antecedent.label, blocked.label, state_count, strength=strength)
    return LatentTransitionPotential(potential.values, name=f"{antecedent.concept}_{antecedent.label}_not_{blocked.label}")


def _relation_variable(lc, bundle) -> str | None:
    relation = getattr(bundle, "is_next_rel", None)
    if relation is None:
        return None
    elements = list(getattr(lc, "e", ()))
    for index, item in enumerate(elements):
        concept_tuple = _concept_tuple(item)
        if concept_tuple is None or concept_tuple[0] is not relation:
            continue
        if index + 1 < len(elements) and _is_v(elements[index + 1]):
            return elements[index + 1].name
    return None


def _formula_from_expr(expr, bundle, relation_var: str, role) -> Formula | None:
    cls_name = getattr(expr, "__class__", None).__name__
    if cls_name in {"andL", "orL"}:
        op = "and" if cls_name == "andL" else "or"
        children = []
        for child in getattr(expr, "e", ()):
            formula = _formula_from_expr(child, bundle, relation_var, role)
            if formula is not None:
                children.append(formula)
        return (op, *children) if children else None
    if cls_name == "notL":
        return None
    return _path_label_ref_from_expr(expr, bundle, relation_var, role)


def _path_label_ref_from_expr(expr, bundle, relation_var: str, role) -> LabelRef | None:
    elements = list(getattr(expr, "e", ())) if hasattr(expr, "e") else [expr]
    for index, item in enumerate(elements):
        concept_tuple = _concept_tuple(item)
        if concept_tuple is None:
            continue
        next_item = elements[index + 1] if index + 1 < len(elements) else None
        if role is not None and (not _is_v(next_item) or next_item.v != (relation_var, role)):
            continue
        concept_name = _concept_name(concept_tuple[0], bundle)
        if concept_name is None or concept_tuple[2] is None:
            continue
        return LabelRef(concept_name, int(concept_tuple[2]))
    for child in elements:
        if hasattr(child, "e"):
            found = _path_label_ref_from_expr(child, bundle, relation_var, role)
            if found is not None:
                return found
    return None


def _concept_name(concept, bundle) -> str | None:
    mapping = {
        "generated_token": getattr(bundle, "generated_token", None),
        "latent_state": getattr(bundle, "latent_state", None),
        "forward_state": getattr(bundle, "forward_state", None),
        "backward_state": getattr(bundle, "backward_state", None),
        "transition_pair": getattr(bundle, "transition_pair", None),
        "wfa_state": getattr(bundle, "wfa_state", None),
        "wfa_transition_pair": getattr(bundle, "wfa_transition_pair", None),
    }
    for name, candidate in mapping.items():
        if candidate is not None and concept is candidate:
            return name
    return None


def _is_latent_relevant(lc, bundle) -> bool:
    for item in _walk_lc(lc):
        concept_tuple = _concept_tuple(item)
        if concept_tuple is not None and _concept_name(concept_tuple[0], bundle) is not None:
            return True
    return False


def _handle_latent_unsupported(lc_name: str, lc, on_unsupported: str) -> None:
    message = (
        f"DomiKnowS logical constraint {lc_name} ({lc.__class__.__name__}) references "
        "generation latent concepts but is not supported by latent auto-discovery"
    )
    if on_unsupported == "error":
        raise ValueError(message)
    if on_unsupported == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def _concept_tuple(item):
    return item if isinstance(item, tuple) and len(item) == 4 else None


def _is_v(item) -> bool:
    return hasattr(item, "_fields") and set(getattr(item, "_fields", ())) >= {"name", "v", "relVarInfo"}


def _walk_lc(lc):
    yield lc
    for item in getattr(lc, "e", ()):
        yield item
        if hasattr(item, "e"):
            yield from _walk_lc(item)
