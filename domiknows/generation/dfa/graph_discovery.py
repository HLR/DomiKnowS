"""Compile DomiKnowS generation graph constraints into DFA fragments.

After a generation graph is built with :class:`~.dfa.encoder.GenerationEncoder`
and constraints are applied (either via :func:`~.dfa.generation_constraints.apply_all_constraints`
or :func:`~.enforcement.mark_for_dfa`), this module walks the graph's logical
constraint nodes and compiles recognised regular fragments directly into DFA
objects for token-level hard decoding.

Recognised DomiKnowS constraint shapes
----------------------------------------
- ``ifL(is_before_rel, ifL(eos, eos))``           → EOS-closure DFA
- ``atMostAL(notL(eos), N)``                      → max non-EOS DFA
- ``atLeastAL(token_value, N)`` / ``existsAL``    → required-token / count DFAs
- ``atMostAL(token_value, 0)``                    → forbidden-token DFA
- ``exactAL`` / ``atLeastAL`` / ``atMostAL`` over token sets or negated tokens
- ``ifL(existsAL(token), atMostAL(notL(eos), N))``→ conditional max-non-EOS DFA
- regular boolean forms: ``andL``, ``orL``, ``notL``, ``nandL``, ``norL``,
  ``xorL``, ``iffL``/``equivalenceL``, and ``ifL(A, B)`` when both sides are regular
- supported ``is_before_rel`` endpoint paths for after-trigger restrictions
  and simple ordered-token existence patterns

Any other shape that still references generation concepts is considered
*unsupported* and handled according to the ``on_unsupported`` policy.

Public API
----------
:func:`constraints_to_dfa_from_graph`
    Primary entry point — compiles supported graph constraints into one DFA.

:func:`analyze_generation_constraints`
    Debugging entry point — reports supported constraints and unsupported
    reasons without changing the discovery APIs.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Iterable

from .core import DFA, complement_dfa, product_dfa, union_dfa
from ._constraints import (
    accept_all_dfa,
    after_token_allowed_dfa,
    conditional_max_non_eos_dfa,
    eos_closure_dfa,
    forbidden_token_dfa,
    max_non_eos_dfa,
    ordered_tokens_dfa,
    required_token_dfa,
    token_set_count_dfa,
)


@dataclass(frozen=True)
class GenerationConstraintAnalysis:
    """Analysis record for one head DomiKnowS logical constraint."""

    lc_name: str
    lc_type: str
    relevant: bool
    supported: bool
    dfas: tuple[DFA, ...] = ()
    reason: str | None = None


def _discover_generation_dfas(
    graph,
    bundle,
    *,
    on_unsupported: str = "warn",
) -> tuple[DFA, ...]:
    """Compile supported generation-relevant graph constraints into DFA fragments."""
    dfas: list[DFA] = []
    for analysis in analyze_generation_constraints(graph, bundle, on_unsupported=on_unsupported):
        dfas.extend(analysis.dfas)
    return tuple(dfas)


def analyze_generation_constraints(
    graph,
    bundle,
    *,
    on_unsupported: str = "warn",
) -> tuple[GenerationConstraintAnalysis, ...]:
    """Analyze head graph constraints for DFA-enforceable generation fragments.

    The analysis is intentionally conservative: it reports supported regular
    sequence fragments as DFA objects and gives a reason for generation-relevant
    constraints that are not faithfully compilable to DFA.
    """
    if on_unsupported not in {"ignore", "warn", "error"}:
        raise ValueError("on_unsupported must be 'ignore', 'warn', or 'error'")

    analyses: list[GenerationConstraintAnalysis] = []
    for lc_name, lc in graph.logicalConstrains.items():
        if not getattr(lc, "headLC", True):
            continue
        relevant = _is_generation_relevant(lc, bundle)
        if _is_latent_marked(lc) and not hasattr(lc, "_generation_dfa_constraint"):
            analyses.append(
                GenerationConstraintAnalysis(
                    lc_name=lc_name,
                    lc_type=lc.__class__.__name__,
                    relevant=relevant,
                    supported=False,
                    reason="constraint is marked for latent enforcement only",
                )
            )
            continue
        discovered = _marked_dfas(lc, bundle)
        if discovered is False:
            discovered = _match_lc_many(lc, bundle)
        if discovered:
            analyses.append(
                GenerationConstraintAnalysis(
                    lc_name=lc_name,
                    lc_type=lc.__class__.__name__,
                    relevant=relevant,
                    supported=True,
                    dfas=tuple(discovered),
                )
            )
            continue
        if relevant:
            reason = _unsupported_reason(lc, bundle)
            _handle_unsupported(lc_name, lc, on_unsupported, reason=reason)
            analyses.append(
                GenerationConstraintAnalysis(
                    lc_name=lc_name,
                    lc_type=lc.__class__.__name__,
                    relevant=True,
                    supported=False,
                    reason=reason,
                )
            )
        else:
            analyses.append(
                GenerationConstraintAnalysis(
                    lc_name=lc_name,
                    lc_type=lc.__class__.__name__,
                    relevant=False,
                    supported=False,
                    reason="constraint does not reference generation concepts",
                )
            )
    return tuple(analyses)


def constraints_to_dfa_from_graph(graph, bundle, *, on_unsupported: str = "warn"):
    """Compile supported DomiKnowS graph constraints into a single DFA.

    Args:
        graph: A built DomiKnowS graph.
        bundle: The :class:`~.dfa.encoder.GenerationBundle` returned alongside
            *graph*.
        on_unsupported: Policy for unrecognised generation-relevant constraints.

    Returns:
        A :class:`~.dfa.DFA` accepting all sequences that satisfy every
        discovered constraint.
    """
    return _combine_dfas(_discover_generation_dfas(graph, bundle, on_unsupported=on_unsupported), bundle.vocabulary)


def _combine_dfas(dfas: Iterable[DFA], vocabulary) -> DFA:
    """Intersect DFA fragments, returning accept-all when no graph constraints match."""
    dfas = tuple(dfas)
    if dfas:
        return product_dfa(dfas)
    return accept_all_dfa(vocabulary)


def _match_lc_many(lc, bundle) -> tuple[DFA, ...] | None:
    """Attempt to match *lc* against the known generation constraint shapes.

    Dispatches on the DomiKnowS logical-constraint class name and delegates to
    the appropriate ``_match_*`` helper.  Returns ``None`` when no pattern
    matches; returns a tuple of one or more :class:`DFA` instances otherwise.
    """
    cls_name = lc.__class__.__name__
    if cls_name == "andL":
        return _match_and_lc(lc, bundle)
    if cls_name == "orL":
        return _match_or_lc(lc, bundle)
    if cls_name == "notL":
        return _match_not_lc(lc, bundle)
    if cls_name == "nandL":
        matched = _match_and_lc(lc, bundle)
        return _negate_many(matched) if matched else None
    if cls_name == "norL":
        matched = _match_or_lc(lc, bundle)
        return _negate_many(matched) if matched else None
    if cls_name == "xorL":
        return _match_xor_lc(lc, bundle)
    if cls_name in {"equivalenceL", "iffL"}:
        return _match_equivalence_lc(lc, bundle)
    if cls_name == "ifL":
        return _as_dfa_tuple(_match_if_lc(lc, bundle))
    if cls_name == "atMostAL":
        return _as_dfa_tuple(_match_at_most_lc(lc, bundle))
    if cls_name == "atLeastAL":
        return _as_dfa_tuple(_match_at_least_lc(lc, bundle))
    if cls_name == "exactAL":
        return _as_dfa_tuple(_match_exact_lc(lc, bundle))
    if cls_name == "existsAL":
        return _as_dfa_tuple(_match_exists_lc(lc, bundle))
    return None


def _as_dfa_tuple(dfa: DFA | None) -> tuple[DFA, ...] | None:
    """Normalize an optional single DFA to the discovery tuple shape."""
    if dfa is None:
        return None
    return (dfa,)


def _marked_dfas(lc, bundle) -> tuple[DFA, ...] | None | bool:
    """Resolve the ``_generation_dfa_constraint`` marker on *lc*.

    Returns ``False`` when no marker is present (caller should fall back to
    structural matching), ``None`` when the marker cannot be resolved, or a
    tuple of DFAs when the marker triggers a successful structural match.
    """
    marker = getattr(lc, "_generation_dfa_constraint", False)
    if marker is False:
        return False
    if marker is True:
        return _match_lc_many(lc, bundle)
    return None


def _is_latent_marked(lc) -> bool:
    """Return ``True`` if *lc* has any latent window specs attached."""
    return bool(getattr(lc, "_generation_latent_specs", ()))


def _match_and_lc(lc, bundle) -> tuple[DFA, ...] | None:
    """Match an ``andL`` as a conjunction of supported generation children.

    Generation-relevant unsupported children make the whole ``andL``
    unsupported.  Non-generation children are ignored for hard decoding and
    remain available to normal DomiKnowS loss/verification.
    """
    dfas: list[DFA] = []
    for child in getattr(lc, "e", ()):
        child_match = _match_lc_many(child, bundle) if hasattr(child, "e") else None
        if child_match is None:
            if _is_generation_relevant(child, bundle):
                return None
            continue
        dfas.extend(child_match)
    return tuple(dfas)


def _match_or_lc(lc, bundle) -> tuple[DFA, ...] | None:
    """Match an ``orL`` as a union of fully supported generation branches."""
    branches: list[DFA] = []
    for child in getattr(lc, "e", ()):
        if not _is_generation_relevant(child, bundle):
            return None
        child_match = _match_lc_many(child, bundle) if hasattr(child, "e") else None
        if not child_match:
            return None
        branches.append(product_dfa(child_match) if len(child_match) > 1 else child_match[0])
    if not branches:
        return None
    return (union_dfa(branches),)


def _match_not_lc(lc, bundle) -> tuple[DFA, ...] | None:
    """Match a regular negation by complementing the child DFA."""
    children = [child for child in getattr(lc, "e", ()) if hasattr(child, "e")]
    if len(children) != 1:
        return None
    matched = _match_lc_many(children[0], bundle)
    return _negate_many(matched) if matched else None


def _negate_many(matched: tuple[DFA, ...] | None) -> tuple[DFA, ...] | None:
    if not matched:
        return None
    child = matched[0] if len(matched) == 1 else product_dfa(matched)
    return (complement_dfa(child),)


def _match_xor_lc(lc, bundle) -> tuple[DFA, ...] | None:
    branches: list[DFA] = []
    for child in getattr(lc, "e", ()):
        child_match = _match_lc_many(child, bundle) if hasattr(child, "e") else None
        if not child_match:
            return None
        branches.append(product_dfa(child_match) if len(child_match) > 1 else child_match[0])
    if len(branches) != 2:
        return None
    left, right = branches
    return (
        union_dfa(
            (
                product_dfa((left, complement_dfa(right))),
                product_dfa((complement_dfa(left), right)),
            )
        ),
    )


def _match_equivalence_lc(lc, bundle) -> tuple[DFA, ...] | None:
    branches: list[DFA] = []
    for child in getattr(lc, "e", ()):
        child_match = _match_lc_many(child, bundle) if hasattr(child, "e") else None
        if not child_match:
            return None
        branches.append(product_dfa(child_match) if len(child_match) > 1 else child_match[0])
    if len(branches) != 2:
        return None
    left, right = branches
    return (
        union_dfa(
            (
                product_dfa((left, right)),
                product_dfa((complement_dfa(left), complement_dfa(right))),
            )
        ),
    )


def _match_if_lc(lc, bundle) -> DFA | None:
    """Try to match an ``ifL`` constraint to a known generation shape.

    Recognised patterns:
    - EOS-closure: ``ifL(is_before_rel, ifL(eos_x, eos_y))``
    - Conditional max non-EOS: ``ifL(existsAL(token), atMostAL(notL(eos), N))``
    """
    if _is_eos_closure(lc, bundle):
        return eos_closure_dfa(bundle.vocabulary)
    after_allowed = _match_before_implication(lc, bundle)
    if after_allowed is not None:
        return after_allowed
    if len(lc.e) == 2:
        token = _exists_token(lc.e[0], bundle)
        max_count = _non_eos_at_most_count(lc.e[1], bundle)
        if token is not None and max_count is not None:
            return conditional_max_non_eos_dfa(bundle.vocabulary, token, max_count)
    if len(lc.e) == 2:
        antecedent = _match_lc_many(lc.e[0], bundle) if hasattr(lc.e[0], "e") else None
        consequent = _match_lc_many(lc.e[1], bundle) if hasattr(lc.e[1], "e") else None
        if antecedent and consequent:
            a = antecedent[0] if len(antecedent) == 1 else product_dfa(antecedent)
            b = consequent[0] if len(consequent) == 1 else product_dfa(consequent)
            return union_dfa((complement_dfa(a), b))
    return None


def _match_at_most_lc(lc, bundle) -> DFA | None:
    """Try to match an ``atMostAL`` constraint to a known generation shape.

    Recognised patterns:
    - Max non-EOS: ``atMostAL(notL(eos), N)``
    - Forbidden token: ``atMostAL(token_value, 0)``
    - Generic token-set count cap.
    """
    max_count = _non_eos_at_most_count(lc, bundle)
    if max_count is not None:
        return max_non_eos_dfa(bundle.vocabulary, max_count)
    token = _direct_token(lc.e, bundle)
    limit = _last_int(lc.e)
    if token is not None and limit == 0:
        return forbidden_token_dfa(bundle.vocabulary, token)
    predicate = _token_predicate_from_count_lc(lc, bundle)
    if predicate is not None and limit is not None:
        tokens, negated = predicate
        return token_set_count_dfa(bundle.vocabulary, tokens, max_count=limit, negated=negated)
    return None


def _match_at_least_lc(lc, bundle) -> DFA | None:
    """Try to match an ``atLeastAL`` constraint to a known generation shape.

    Recognised pattern:
    - Required token: ``atLeastAL(token_value, N)`` where N ≥ 1.
    - Generic token-set count floor.
    """
    token = _direct_token(lc.e, bundle)
    min_count = _last_int(lc.e)
    if token is not None and min_count is not None and min_count >= 1:
        return required_token_dfa(bundle.vocabulary, token, min_count=min_count)
    predicate = _token_predicate_from_count_lc(lc, bundle)
    if predicate is not None and min_count is not None:
        tokens, negated = predicate
        return token_set_count_dfa(bundle.vocabulary, tokens, min_count=min_count, negated=negated)
    return None


def _match_exact_lc(lc, bundle) -> DFA | None:
    """Match ``exactAL(predicate, N)`` for regular token predicates."""
    limit = _last_int(lc.e)
    if limit is None:
        return None
    predicate = _token_predicate_from_count_lc(lc, bundle)
    if predicate is None:
        return None
    tokens, negated = predicate
    return token_set_count_dfa(bundle.vocabulary, tokens, exact_count=limit, negated=negated)


def _match_exists_lc(lc, bundle) -> DFA | None:
    """Try to match an ``existsAL`` constraint to a known generation shape.

    Recognised patterns:
    - Required token (min 1): ``existsAL(token_value)``.
    - Ordered pair existence: ``existsAL(andL(before, A(first), B(second)))``.
    """
    token = _exists_token(lc, bundle)
    if token is not None:
        return required_token_dfa(bundle.vocabulary, token, 1)
    ordered = _match_ordered_pair_exists(lc, bundle)
    if ordered is not None:
        return ordered
    predicate = _token_predicate_from_count_lc(lc, bundle)
    if predicate is not None:
        tokens, negated = predicate
        return token_set_count_dfa(bundle.vocabulary, tokens, min_count=1, negated=negated)
    return None


def _is_eos_closure(lc, bundle) -> bool:
    """Return ``True`` if *lc* matches the EOS-closure ``ifL`` pattern."""
    if len(lc.e) < 3:
        return False
    rel = _concept_tuple(lc.e[0])
    if rel is None or rel[0] is not bundle.is_before_rel:
        return False
    nested = next((item for item in lc.e if getattr(item, "__class__", None).__name__ == "ifL"), None)
    if nested is None:
        return False
    tokens = [_token_from_tuple(item, bundle) for item in nested.e if _concept_tuple(item) is not None]
    return tokens == [bundle.vocabulary.eos_token, bundle.vocabulary.eos_token]


def _non_eos_at_most_count(lc, bundle) -> int | None:
    """Extract the count from an ``atMostAL(notL(eos), N)`` sub-expression."""
    if getattr(lc, "__class__", None).__name__ != "atMostAL":
        return None
    limit = _last_int(lc.e)
    if limit is None:
        return None
    if not lc.e or getattr(lc.e[0], "__class__", None).__name__ != "notL":
        return None
    token = _direct_token(lc.e[0].e, bundle)
    if token == bundle.vocabulary.eos_token:
        return limit
    return None


def _exists_token(lc, bundle) -> str | None:
    """Extract the token from an ``existsAL(token_value)`` sub-expression."""
    if getattr(lc, "__class__", None).__name__ != "existsAL":
        return None
    return _direct_token(lc.e, bundle)


def _token_predicate_from_count_lc(lc, bundle) -> tuple[tuple[str, ...], bool] | None:
    """Return a token-set predicate from an accumulated counting LC."""
    elements = [item for item in getattr(lc, "e", ()) if not isinstance(item, int)]
    if len(elements) == 1 and hasattr(elements[0], "e"):
        return _token_predicate_from_expr(elements[0], bundle)
    return _token_predicate_from_elements(elements, bundle)


def _token_predicate_from_expr(expr, bundle) -> tuple[tuple[str, ...], bool] | None:
    """Parse regular token predicates used inside count constraints."""
    cls_name = getattr(expr, "__class__", None).__name__
    if cls_name == "notL":
        children = [child for child in getattr(expr, "e", ()) if hasattr(child, "e") or _concept_tuple(child)]
        if len(children) == 1 and hasattr(children[0], "e"):
            child = _token_predicate_from_expr(children[0], bundle)
        else:
            child = _token_predicate_from_elements(getattr(expr, "e", ()), bundle)
        if child is None:
            return None
        tokens, negated = child
        return tokens, not negated
    if cls_name == "orL":
        if not any(hasattr(child, "e") for child in getattr(expr, "e", ())):
            tokens = _tokens_from_flat_elements(getattr(expr, "e", ()), bundle)
            return (tuple(sorted(tokens)), False) if tokens is not None else None
        sets = []
        for child in getattr(expr, "e", ()):
            child_pred = _token_predicate_from_expr(child, bundle) if hasattr(child, "e") else None
            if child_pred is None:
                return None
            sets.append(_predicate_token_set(child_pred, bundle))
        return tuple(sorted(set().union(*sets))), False
    if cls_name == "andL":
        if not any(hasattr(child, "e") for child in getattr(expr, "e", ())):
            tokens = _tokens_from_flat_elements(getattr(expr, "e", ()), bundle)
            if tokens is None:
                return None
            if not tokens:
                return (), False
            current = {tokens[0]}
            for token in tokens[1:]:
                current &= {token}
            return tuple(sorted(current)), False
        sets = []
        for child in getattr(expr, "e", ()):
            child_pred = _token_predicate_from_expr(child, bundle) if hasattr(child, "e") else None
            if child_pred is None:
                return None
            sets.append(_predicate_token_set(child_pred, bundle))
        if not sets:
            return None
        current = set(sets[0])
        for token_set in sets[1:]:
            current &= set(token_set)
        return tuple(sorted(current)), False
    if cls_name in {"atLeastAL", "atMostAL", "exactAL", "existsAL"}:
        return None
    return _token_predicate_from_elements(getattr(expr, "e", ()), bundle)


def _predicate_token_set(predicate: tuple[tuple[str, ...], bool], bundle) -> set[str]:
    tokens, negated = predicate
    token_set = set(tokens)
    all_tokens = set(bundle.vocabulary.labels)
    return all_tokens - token_set if negated else token_set


def _token_predicate_from_elements(elements: Iterable, bundle) -> tuple[tuple[str, ...], bool] | None:
    """Return a pathless positive token predicate from raw LC elements."""
    elements = list(elements)
    if _has_any_path(elements):
        return None
    tokens = [_token_from_tuple(item, bundle) for item in elements if _concept_tuple(item) is not None]
    tokens = [token for token in tokens if token is not None]
    if len(tokens) != 1:
        return None
    return (tokens[0],), False


def _match_before_implication(lc, bundle) -> DFA | None:
    """Match ``ifL(before, ifL(trigger(first), allowed(second)))``."""
    before_var = _before_relation_variable(lc, bundle)
    if before_var is None:
        return None
    nested = next((item for item in lc.e if getattr(item, "__class__", None).__name__ == "ifL"), None)
    if nested is None:
        return None
    trigger = _path_token_predicate_from_flat(nested.e, bundle, before_var, bundle.first_token)
    allowed = _path_token_predicate_from_flat(nested.e, bundle, before_var, bundle.second_token)
    if trigger is None or allowed is None:
        return None
    trigger_tokens, trigger_negated = trigger
    allowed_tokens, allowed_negated = allowed
    if trigger_negated or allowed_negated:
        return None
    return after_token_allowed_dfa(bundle.vocabulary, trigger_tokens, allowed_tokens)


def _match_ordered_pair_exists(lc, bundle) -> DFA | None:
    """Match ``existsAL(andL(before, A(first), B(second)))``."""
    children = [item for item in getattr(lc, "e", ()) if hasattr(item, "e")]
    if len(children) != 1 or getattr(children[0], "__class__", None).__name__ != "andL":
        return None
    and_lc = children[0]
    before_var = _before_relation_variable(and_lc, bundle)
    if before_var is None:
        return None
    first_pred = _path_token_predicate_from_flat(and_lc.e, bundle, before_var, bundle.first_token)
    second_pred = _path_token_predicate_from_flat(and_lc.e, bundle, before_var, bundle.second_token)
    if first_pred is None or second_pred is None:
        return None
    first_tokens, first_negated = first_pred
    second_tokens, second_negated = second_pred
    if first_negated or second_negated or len(first_tokens) != 1 or len(second_tokens) != 1:
        return None
    first = first_tokens[0]
    second = second_tokens[0]
    return ordered_tokens_dfa(bundle.vocabulary, (first, second))


def _before_relation_variable(lc, bundle) -> str | None:
    """Return the relation variable name if *lc* contains ``is_before_rel``."""
    elements = list(getattr(lc, "e", ()))
    for index, item in enumerate(elements):
        concept_tuple = _concept_tuple(item)
        if concept_tuple is None or concept_tuple[0] is not bundle.is_before_rel:
            continue
        if index + 1 < len(elements) and _is_v(elements[index + 1]):
            return elements[index + 1].name
    return None


def _path_token_predicate_from_flat(elements, bundle, before_var: str, role) -> tuple[tuple[str, ...], bool] | None:
    """Parse one or more token predicates in a flat LC element list by path role."""
    elements = list(elements)
    tokens = []
    for index, item in enumerate(elements):
        token = _token_from_tuple(item, bundle)
        if token is None:
            continue
        next_item = elements[index + 1] if index + 1 < len(elements) else None
        if _is_v(next_item) and next_item.v == (before_var, role):
            tokens.append(token)
    if not tokens:
        return None
    return tuple(sorted(set(tokens))), False


def _direct_token(elements: Iterable, bundle) -> str | None:
    """Return the single token referenced by *elements*, or ``None``."""
    elements = list(elements)
    if _has_any_path(elements):
        return None
    tokens = [_token_from_tuple(item, bundle) for item in elements if _concept_tuple(item) is not None]
    tokens = [token for token in tokens if token is not None]
    if len(tokens) != 1:
        return None
    return tokens[0]


def _tokens_from_flat_elements(elements: Iterable, bundle) -> list[str] | None:
    elements = list(elements)
    if _has_any_path(elements):
        return None
    tokens = [_token_from_tuple(item, bundle) for item in elements if _concept_tuple(item) is not None]
    tokens = [token for token in tokens if token is not None]
    return tokens if tokens else None


def _has_any_path(elements: Iterable) -> bool:
    return any(_is_v(item) and item.v is not None for item in elements)


def _is_v(item) -> bool:
    return (
        hasattr(item, "_fields")
        and set(getattr(item, "_fields", ())) >= {"name", "v", "relVarInfo"}
    )


def _token_from_tuple(item, bundle) -> str | None:
    """Resolve a concept 4-tuple to its surface-form token string."""
    concept_tuple = _concept_tuple(item)
    if concept_tuple is None:
        return None
    concept, _name, label, _cardinality = concept_tuple
    if concept is not bundle.generated_token or label is None:
        return None
    try:
        return bundle.vocabulary.token_for_label(int(label))
    except (ValueError, IndexError):
        return None


def _concept_tuple(item):
    """Return *item* if it is a valid concept 4-tuple, else ``None``."""
    if isinstance(item, tuple) and len(item) == 4:
        return item
    return None


def _last_int(elements: Iterable) -> int | None:
    """Return the last integer found in *elements*, or ``None``."""
    ints = [item for item in elements if isinstance(item, int)]
    return int(ints[-1]) if ints else None


def _is_generation_relevant(lc, bundle) -> bool:
    """Return ``True`` if *lc* references any generation-specific concept."""
    for item in _walk_lc(lc):
        concept_tuple = _concept_tuple(item)
        if concept_tuple is None:
            continue
        concept = concept_tuple[0]
        if concept is bundle.generated_token or concept is bundle.is_before_rel:
            return True
    return False


def _walk_lc(lc):
    """Yield *lc* and all of its descendants in depth-first order."""
    yield lc
    for item in getattr(lc, "e", ()):
        yield item
        if hasattr(item, "e"):
            yield from _walk_lc(item)


def _unsupported_reason(lc, bundle) -> str:
    """Return a compact reason why a generation-relevant LC is unsupported."""
    cls_name = getattr(lc, "__class__", None).__name__
    if cls_name in {"sumL", "iotaL", "queryL", "sameL", "differentL"}:
        return f"{cls_name} depends on numeric selection/query semantics, not a regular token language"
    if cls_name in {"greaterL", "greaterEqL", "lessL", "lessEqL", "equalCountsL", "notEqualCountsL"}:
        return f"{cls_name} comparative count semantics are not compiled to DFA in this pass"
    for item in _walk_lc(lc):
        if getattr(item, "__class__", None).__name__ == "eqL":
            return "eqL path filters require DataNode/path execution outside DFA decoding"
        if _is_v(item) and item.v is not None and not _is_supported_generation_path(item.v, bundle):
            return f"path {item.v!r} is outside supported generation sequence paths"
    return "constraint is generation-relevant but outside the supported regular DFA fragment"


def _is_supported_generation_path(path, bundle) -> bool:
    if not isinstance(path, tuple) or len(path) != 2:
        return False
    _var_name, role = path
    return role is bundle.first_token or role is bundle.second_token


def _handle_unsupported(lc_name: str, lc, on_unsupported: str, *, reason: str | None = None) -> None:
    """Emit a warning or raise an error for an unrecognised generation constraint."""
    message = (
        f"DomiKnowS logical constraint {lc_name} ({lc.__class__.__name__}) references "
        "generation concepts but is not supported by generation DFA discovery"
    )
    if reason:
        message = f"{message}: {reason}"
    if on_unsupported == "error":
        raise ValueError(message)
    if on_unsupported == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=3)
