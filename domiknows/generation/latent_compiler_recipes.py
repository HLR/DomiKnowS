"""Packaged custom graph-to-latent compiler recipes.

The recipes in this module are small opt-in factories around the public
``extra_latent_compilers`` hook in :mod:`domiknows.generation.enforcement`.
They do not broaden built-in DomiKnowS graph discovery by default; callers
explicitly pass the returned compilers to discovery functions.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from .enforcement import (
    GraphLatentCompiler,
    GraphLatentCompilerResult,
    LatentWindowSpec,
    graph_latent_compiler_result,
)
from .latent_constraints import Formula, LabelRef
from .latent_potentials import LatentTransitionPotential, forbid_hmm_transition


LabelLike: TypeAlias = int | str | LabelRef
RecipeMatcher: TypeAlias = Callable[[object, object], object | None]


@dataclass(frozen=True)
class WindowRecipeMatch:
    """Matcher output for recipes that create one :class:`LatentWindowSpec`."""

    if_label: LabelLike
    formula: Formula | LabelLike
    window: int | None = None
    weight: float | None = None
    reduction: str | None = None
    name: str | None = None
    concept: str | None = None
    mask_policy: str | None = None
    empty_window_policy: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CooccurrenceRecipeMatch:
    """Matcher output for nearby AND/OR co-occurrence latent losses."""

    if_label: LabelLike
    labels: Sequence[LabelLike]
    mode: str = "or"
    window: int | None = None
    weight: float | None = None
    reduction: str | None = None
    name: str | None = None
    concept: str | None = None
    mask_policy: str | None = None
    empty_window_policy: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ForbiddenTransitionRecipeMatch:
    """Matcher output for latent-state transition potential recipes."""

    from_state: int | str
    to_state: int | str
    state_count: int | None = None
    strength: float | None = None
    name: str | None = None


@dataclass(frozen=True)
class UnsupportedRecipeMatch:
    """Matcher output for relevant project LCs that a recipe cannot compile."""

    reason: str
    relevant: bool = True


def adjacent_implication_recipe(
    matcher: RecipeMatcher | None = None,
    *,
    lc_class_name: str | Sequence[str] | None = None,
    if_label: LabelLike | None = None,
    then_label: LabelLike | None = None,
    if_token: str | None = None,
    then_token: str | None = None,
    concept: str = "generated_token",
    weight: float = 1.0,
    reduction: str = "mean",
    name: str | None = None,
    empty_window_policy: str = "ignore",
    compiler_name: str = "adjacent_implication_recipe",
) -> GraphLatentCompiler:
    """Create a compiler for ``A(t) => B(t+1)`` style latent losses."""

    return _window_recipe(
        matcher,
        lc_class_name=lc_class_name,
        default_if=if_label if if_label is not None else if_token,
        default_formula=then_label if then_label is not None else then_token,
        fixed_window=1,
        concept=concept,
        weight=weight,
        reduction=reduction,
        name=name,
        empty_window_policy=empty_window_policy,
        metadata={"recipe": "adjacent_implication"},
        compiler_name=compiler_name,
        missing_reason="adjacent implication recipe requires if_label/if_token and then_label/then_token",
    )


def bounded_lookahead_recipe(
    matcher: RecipeMatcher | None = None,
    *,
    window: int,
    lc_class_name: str | Sequence[str] | None = None,
    if_label: LabelLike | None = None,
    then_label: LabelLike | None = None,
    if_token: str | None = None,
    then_token: str | None = None,
    concept: str = "generated_token",
    weight: float = 1.0,
    reduction: str = "mean",
    name: str | None = None,
    empty_window_policy: str = "ignore",
    compiler_name: str = "bounded_lookahead_recipe",
) -> GraphLatentCompiler:
    """Create a compiler for ``A(t) => exists B in next window`` losses."""

    return _window_recipe(
        matcher,
        lc_class_name=lc_class_name,
        default_if=if_label if if_label is not None else if_token,
        default_formula=then_label if then_label is not None else then_token,
        fixed_window=int(window),
        concept=concept,
        weight=weight,
        reduction=reduction,
        name=name,
        empty_window_policy=empty_window_policy,
        metadata={"recipe": "bounded_lookahead"},
        compiler_name=compiler_name,
        missing_reason="bounded lookahead recipe requires if_label/if_token and then_label/then_token",
    )


def cooccurrence_recipe(
    matcher: RecipeMatcher | None = None,
    *,
    window: int,
    lc_class_name: str | Sequence[str] | None = None,
    if_label: LabelLike | None = None,
    candidate_labels: Sequence[LabelLike] = (),
    if_token: str | None = None,
    candidate_tokens: Sequence[str] = (),
    mode: str = "or",
    concept: str = "generated_token",
    weight: float = 1.0,
    reduction: str = "mean",
    name: str | None = None,
    empty_window_policy: str = "ignore",
    compiler_name: str = "cooccurrence_recipe",
) -> GraphLatentCompiler:
    """Create a compiler for nearby AND/OR token or state co-occurrence."""

    default_if = if_label if if_label is not None else if_token
    default_labels: tuple[LabelLike, ...] = tuple(candidate_labels) + tuple(candidate_tokens)

    def default_match(lc, bundle):
        if not _class_matches(lc, lc_class_name):
            return None
        if default_if is None or not default_labels:
            return UnsupportedRecipeMatch("cooccurrence recipe requires an if label/token and candidate labels/tokens")
        return CooccurrenceRecipeMatch(default_if, default_labels, mode=mode)

    active_matcher = matcher or default_match

    def compiler(lc, bundle):
        if matcher is not None and not _class_matches(lc, lc_class_name):
            return None
        match = active_matcher(lc, bundle)
        if match is None:
            return None
        if isinstance(match, GraphLatentCompilerResult):
            return match
        if isinstance(match, UnsupportedRecipeMatch):
            return _unsupported_result(match, compiler_name)
        if isinstance(match, LatentWindowSpec):
            return graph_latent_compiler_result(latent_specs=match, compiler_name=compiler_name)
        if not isinstance(match, CooccurrenceRecipeMatch):
            raise TypeError("cooccurrence recipe matcher must return CooccurrenceRecipeMatch, LatentWindowSpec, GraphLatentCompilerResult, UnsupportedRecipeMatch, or None")
        formula = _cooccurrence_formula(match, bundle, concept=match.concept or concept)
        spec = LatentWindowSpec(
            if_label=_resolve_label(match.if_label, bundle, match.concept or concept),
            formula=formula,
            window=match.window or int(window),
            weight=float(match.weight if match.weight is not None else weight),
            reduction=match.reduction or reduction,
            name=match.name or name,
            concept=match.concept or concept,
            mask_policy=match.mask_policy or "lengths",
            empty_window_policy=match.empty_window_policy or empty_window_policy,
            metadata={**{"recipe": "cooccurrence"}, **dict(match.metadata)},
        )
        return graph_latent_compiler_result(latent_specs=spec, compiler_name=compiler_name)

    return compiler


def forbidden_transition_potential_recipe(
    matcher: RecipeMatcher | None = None,
    *,
    lc_class_name: str | Sequence[str] | None = None,
    from_state: int | str | None = None,
    to_state: int | str | None = None,
    state_count: int | None = None,
    strength: float = 0.0,
    name: str | None = None,
    compiler_name: str = "forbidden_transition_potential_recipe",
) -> GraphLatentCompiler:
    """Create a compiler for latent/WFA ``state_i -> state_j`` reweighting."""

    def default_match(lc, bundle):
        if not _class_matches(lc, lc_class_name):
            return None
        if from_state is None or to_state is None:
            return UnsupportedRecipeMatch("forbidden transition recipe requires from_state and to_state")
        return ForbiddenTransitionRecipeMatch(from_state, to_state, state_count=state_count, strength=strength, name=name)

    active_matcher = matcher or default_match

    def compiler(lc, bundle):
        if matcher is not None and not _class_matches(lc, lc_class_name):
            return None
        match = active_matcher(lc, bundle)
        if match is None:
            return None
        if isinstance(match, GraphLatentCompilerResult):
            return match
        if isinstance(match, UnsupportedRecipeMatch):
            return _unsupported_result(match, compiler_name)
        if isinstance(match, LatentTransitionPotential):
            return graph_latent_compiler_result(transition_potentials=match, compiler_name=compiler_name)
        if not isinstance(match, ForbiddenTransitionRecipeMatch):
            raise TypeError("forbidden transition recipe matcher must return ForbiddenTransitionRecipeMatch, LatentTransitionPotential, GraphLatentCompilerResult, UnsupportedRecipeMatch, or None")
        count = match.state_count or state_count or len(getattr(bundle, "state_names", ()))
        if not count:
            return graph_latent_compiler_result(
                relevant=True,
                supported=False,
                reason="forbidden transition recipe requires state_count or bundle.state_names",
                compiler_name=compiler_name,
            )
        src = _resolve_state(match.from_state, bundle, count)
        dst = _resolve_state(match.to_state, bundle, count)
        potential = forbid_hmm_transition(src, dst, int(count), strength=float(match.strength if match.strength is not None else strength))
        display_name = match.name or name or f"recipe_forbid_{src}_to_{dst}"
        potential = LatentTransitionPotential(potential.values, name=display_name)
        return graph_latent_compiler_result(transition_potentials=potential, compiler_name=compiler_name)

    return compiler


def common_latent_compiler_recipes(
    *,
    adjacent_matcher: RecipeMatcher | None = None,
    bounded_lookahead_matcher: RecipeMatcher | None = None,
    cooccurrence_matcher: RecipeMatcher | None = None,
    forbidden_transition_matcher: RecipeMatcher | None = None,
    adjacent_lc_class_name: str | Sequence[str] | None = None,
    adjacent_if_token: str | None = None,
    adjacent_then_token: str | None = None,
    bounded_lc_class_name: str | Sequence[str] | None = None,
    bounded_if_token: str | None = None,
    bounded_then_token: str | None = None,
    bounded_window: int = 3,
    cooccurrence_lc_class_name: str | Sequence[str] | None = None,
    cooccurrence_if_token: str | None = None,
    cooccurrence_candidate_tokens: Sequence[str] = (),
    cooccurrence_window: int = 3,
    cooccurrence_mode: str = "or",
    forbidden_transition_lc_class_name: str | Sequence[str] | None = None,
    forbidden_from_state: int | str | None = None,
    forbidden_to_state: int | str | None = None,
) -> tuple[GraphLatentCompiler, ...]:
    """Return a tuple of configured common recipe compilers.

    Unconfigured recipes are omitted, so calling this with no arguments returns
    an empty tuple rather than installing broad process-wide behavior.
    """

    recipes: list[GraphLatentCompiler] = []
    if adjacent_matcher is not None or adjacent_lc_class_name is not None:
        recipes.append(
            adjacent_implication_recipe(
                adjacent_matcher,
                lc_class_name=adjacent_lc_class_name,
                if_token=adjacent_if_token,
                then_token=adjacent_then_token,
            )
        )
    if bounded_lookahead_matcher is not None or bounded_lc_class_name is not None:
        recipes.append(
            bounded_lookahead_recipe(
                bounded_lookahead_matcher,
                window=bounded_window,
                lc_class_name=bounded_lc_class_name,
                if_token=bounded_if_token,
                then_token=bounded_then_token,
            )
        )
    if cooccurrence_matcher is not None or cooccurrence_lc_class_name is not None:
        recipes.append(
            cooccurrence_recipe(
                cooccurrence_matcher,
                window=cooccurrence_window,
                lc_class_name=cooccurrence_lc_class_name,
                if_token=cooccurrence_if_token,
                candidate_tokens=cooccurrence_candidate_tokens,
                mode=cooccurrence_mode,
            )
        )
    if forbidden_transition_matcher is not None or forbidden_transition_lc_class_name is not None:
        recipes.append(
            forbidden_transition_potential_recipe(
                forbidden_transition_matcher,
                lc_class_name=forbidden_transition_lc_class_name,
                from_state=forbidden_from_state,
                to_state=forbidden_to_state,
            )
        )
    return tuple(recipes)


def _window_recipe(
    matcher: RecipeMatcher | None,
    *,
    lc_class_name: str | Sequence[str] | None,
    default_if: LabelLike | None,
    default_formula: Formula | LabelLike | None,
    fixed_window: int,
    concept: str,
    weight: float,
    reduction: str,
    name: str | None,
    empty_window_policy: str,
    metadata: Mapping[str, object],
    compiler_name: str,
    missing_reason: str,
) -> GraphLatentCompiler:
    def default_match(lc, bundle):
        if not _class_matches(lc, lc_class_name):
            return None
        if default_if is None or default_formula is None:
            return UnsupportedRecipeMatch(missing_reason)
        return WindowRecipeMatch(default_if, default_formula)

    active_matcher = matcher or default_match

    def compiler(lc, bundle):
        if matcher is not None and not _class_matches(lc, lc_class_name):
            return None
        match = active_matcher(lc, bundle)
        if match is None:
            return None
        if isinstance(match, GraphLatentCompilerResult):
            return match
        if isinstance(match, UnsupportedRecipeMatch):
            return _unsupported_result(match, compiler_name)
        if isinstance(match, LatentWindowSpec):
            return graph_latent_compiler_result(latent_specs=match, compiler_name=compiler_name)
        if not isinstance(match, WindowRecipeMatch):
            raise TypeError("window recipe matcher must return WindowRecipeMatch, LatentWindowSpec, GraphLatentCompilerResult, UnsupportedRecipeMatch, or None")
        spec_concept = match.concept or concept
        spec = LatentWindowSpec(
            if_label=_resolve_label(match.if_label, bundle, spec_concept),
            formula=_resolve_formula(match.formula, bundle, spec_concept),
            window=match.window or fixed_window,
            weight=float(match.weight if match.weight is not None else weight),
            reduction=match.reduction or reduction,
            name=match.name or name,
            concept=spec_concept,
            mask_policy=match.mask_policy or "lengths",
            empty_window_policy=match.empty_window_policy or empty_window_policy,
            metadata={**dict(metadata), **dict(match.metadata)},
        )
        return graph_latent_compiler_result(latent_specs=spec, compiler_name=compiler_name)

    return compiler


def _class_matches(lc, lc_class_name: str | Sequence[str] | None) -> bool:
    if lc_class_name is None:
        return True
    names = (lc_class_name,) if isinstance(lc_class_name, str) else tuple(lc_class_name)
    return lc.__class__.__name__ in names


def _unsupported_result(match: UnsupportedRecipeMatch, compiler_name: str) -> GraphLatentCompilerResult:
    return graph_latent_compiler_result(
        relevant=match.relevant,
        supported=False,
        reason=match.reason,
        compiler_name=compiler_name,
    )


def _cooccurrence_formula(match: CooccurrenceRecipeMatch, bundle, *, concept: str) -> Formula:
    mode = str(match.mode).lower()
    if mode not in {"and", "or"}:
        raise ValueError("cooccurrence mode must be 'and' or 'or'")
    labels = tuple(_resolve_label(label, bundle, concept) for label in match.labels)
    if not labels:
        raise ValueError("cooccurrence recipe requires at least one candidate label")
    return labels[0] if len(labels) == 1 else (mode, *labels)


def _resolve_formula(formula: Formula | LabelLike, bundle, concept: str) -> Formula:
    if isinstance(formula, tuple):
        if not formula:
            raise ValueError("formula tuple must not be empty")
        op = formula[0]
        return (op, *(_resolve_formula(child, bundle, concept) for child in formula[1:]))
    return _resolve_label(formula, bundle, concept)


def _resolve_label(label: LabelLike, bundle, concept: str) -> int | LabelRef:
    if isinstance(label, LabelRef):
        return label
    if isinstance(label, str):
        index = _resolve_label_index(label, bundle, concept)
    else:
        index = int(label)
    return index if concept == "generated_token" else LabelRef(concept, index)


def _resolve_label_index(label: str, bundle, concept: str) -> int:
    if concept == "generated_token":
        return int(bundle.vocabulary.label_for_token(label))
    names = tuple(getattr(bundle, "state_names", ()))
    if names and label in names:
        return names.index(label)
    try:
        return int(label)
    except ValueError as exc:
        raise ValueError(f"cannot resolve label {label!r} for concept {concept!r}") from exc


def _resolve_state(state: int | str, bundle, state_count: int) -> int:
    if isinstance(state, str):
        names = tuple(getattr(bundle, "state_names", ()))
        if names and state in names:
            return names.index(state)
        try:
            index = int(state)
        except ValueError as exc:
            raise ValueError(f"unknown state {state!r}") from exc
    else:
        index = int(state)
    if index < 0 or index >= int(state_count):
        raise ValueError(f"state index {index} is out of range for {state_count} states")
    return index


__all__ = [
    "CooccurrenceRecipeMatch",
    "ForbiddenTransitionRecipeMatch",
    "RecipeMatcher",
    "UnsupportedRecipeMatch",
    "WindowRecipeMatch",
    "adjacent_implication_recipe",
    "bounded_lookahead_recipe",
    "common_latent_compiler_recipes",
    "cooccurrence_recipe",
    "forbidden_transition_potential_recipe",
]
