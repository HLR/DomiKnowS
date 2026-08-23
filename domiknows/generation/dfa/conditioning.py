"""Context-conditioned hard constraints for graph-defined generation.

Some generation constraints are structurally stable but depend on facts that
vary per example.  An object token, for example, may be legal only when its
object type occurs in the current task world.  Rebuilding the full graph DFA
for every example is unnecessary: this module records the dependency on a
DomiKnowS logical constraint and binds the compiled DFA to concrete context
values at decode time.

The resulting :class:`ContextualDFA` exposes the normal DFA interface.  It is
therefore usable by greedy/beam decoding, sampling, and differentiable
rescoring without decoder-specific masks.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .core import State, Symbol
from .vocabulary import TokenVocabulary


@dataclass(frozen=True)
class ContextualTokenConstraintSpec:
    """A graph-declared mapping from output labels to required context values."""

    name: str
    context_key: str
    label_to_value: tuple[tuple[int, str], ...]
    trigger_labels: tuple[int, ...] = ()
    allow_missing_context: bool = False

    @property
    def mapping(self) -> dict[int, str]:
        return dict(self.label_to_value)


def mark_for_contextual_dfa(
    lc,
    *,
    context_key: str,
    token_to_value: Mapping[int | str, str],
    vocabulary: TokenVocabulary | None = None,
    name: str | None = None,
    allow_missing_context: bool = False,
    trigger_tokens: Iterable[int | str] = (),
):
    """Mark a graph LC as a per-example token-availability constraint.

    ``token_to_value`` maps each guarded output token to the value that must
    occur in ``context[context_key]``.  String token names require a
    :class:`TokenVocabulary`; integer labels can be used directly.
    """
    if not context_key:
        raise ValueError("context_key must not be empty")
    normalized: dict[int, str] = {}
    for token, value in token_to_value.items():
        if isinstance(token, str):
            if vocabulary is None:
                raise ValueError("string token names require a TokenVocabulary")
            label = int(vocabulary.label_for_token(token))
        else:
            label = int(token)
        required = str(value).strip()
        if not required:
            raise ValueError(f"context value for token {token!r} must not be empty")
        normalized[label] = required
    if not normalized:
        raise ValueError("token_to_value must not be empty")
    normalized_triggers = []
    for token in trigger_tokens:
        if isinstance(token, str):
            if vocabulary is None:
                raise ValueError("string trigger names require a TokenVocabulary")
            normalized_triggers.append(int(vocabulary.label_for_token(token)))
        else:
            normalized_triggers.append(int(token))

    spec = ContextualTokenConstraintSpec(
        name=name or getattr(lc, "name", None) or getattr(lc, "lcName", "contextual_token"),
        context_key=str(context_key),
        label_to_value=tuple(sorted(normalized.items())),
        trigger_labels=tuple(sorted(set(normalized_triggers))),
        allow_missing_context=bool(allow_missing_context),
    )
    specs = list(getattr(lc, "_generation_contextual_dfa_specs", ()))
    specs.append(spec)
    setattr(lc, "_generation_contextual_dfa_specs", tuple(specs))
    return lc


def declare_contextual_token_constraint(
    graph,
    bundle,
    *,
    tokens: Iterable[str],
    context_key: str,
    token_to_value: Mapping[str, str] | None = None,
    concept_name: str = "contextually_available_token",
    constraint_name: str = "contextual_token_availability",
    allow_missing_context: bool = False,
    trigger_tokens: Iterable[str] = (),
):
    """Declare a reusable graph LC and its context-to-DFA binding metadata.

    The graph schema gains a Boolean sub-concept on token positions.  The LC
    states that every guarded generated token must satisfy that availability
    concept.  The LC is inactive for ordinary DataNode inference because its
    truth is supplied by request context rather than a sensor; its contextual
    DFA marker is the executable hard-enforcement form.
    """
    from domiknows.graph import Concept
    from domiknows.graph.logicalConstrain import ifL, orL

    guarded = tuple(dict.fromkeys(str(token) for token in tokens))
    if not guarded:
        raise ValueError("tokens must not be empty")
    known = set(bundle.vocabulary.tokens)
    unknown = [token for token in guarded if token not in known]
    if unknown:
        raise ValueError(f"contextual constraint contains unknown tokens: {unknown!r}")
    mapping = dict(token_to_value or {token: token for token in guarded})
    triggers = tuple(dict.fromkeys(str(token) for token in trigger_tokens))
    unknown_triggers = [token for token in triggers if token not in known]
    if unknown_triggers:
        raise ValueError(
            f"contextual constraint contains unknown triggers: {unknown_triggers!r}"
        )
    missing = [token for token in guarded if token not in mapping]
    if missing:
        raise ValueError(f"token_to_value is missing guarded tokens: {missing!r}")

    with graph:
        # Keep availability as a peer Boolean concept. Some traditional
        # DomiKnowS graphs attach an EnumConcept below ``token``; adding a
        # second subclass after that enum is not supported by their registry.
        availability = Concept(name=concept_name)
        if triggers:
            trigger_calls = [
                bundle.context.token_value(
                    token,
                    f"context_trigger_{index}",
                    path=("context_edge", bundle.first_token),
                )
                for index, token in enumerate(triggers)
            ]
            trigger_predicate = (
                trigger_calls[0]
                if len(trigger_calls) == 1
                else orL(*trigger_calls)
            )
            guarded_next_calls = [
                bundle.context.token_value(
                    token,
                    f"context_guarded_{index}",
                    path=("context_edge", bundle.second_token),
                )
                for index, token in enumerate(guarded)
            ]
            guarded_next = (
                guarded_next_calls[0]
                if len(guarded_next_calls) == 1
                else orL(*guarded_next_calls)
            )
            antecedent = bundle.context.is_before_rel("context_edge")
            consequent = ifL(
                trigger_predicate,
                ifL(guarded_next, availability("context_guarded_token")),
            )
        else:
            calls = [
                bundle.context.token_value(token, f"guarded_{index}")
                for index, token in enumerate(guarded)
            ]
            antecedent = calls[0] if len(calls) == 1 else orL(*calls)
            consequent = availability("guarded_token")
        lc = ifL(
            antecedent,
            consequent,
            active=False,
            name=constraint_name,
        )
        mark_for_contextual_dfa(
            lc,
            context_key=context_key,
            token_to_value={token: mapping[token] for token in guarded},
            vocabulary=bundle.vocabulary,
            name=constraint_name,
            allow_missing_context=allow_missing_context,
            trigger_tokens=triggers,
        )
    return availability, lc


def discover_contextual_token_constraints(graph) -> tuple[ContextualTokenConstraintSpec, ...]:
    """Return all context-conditioned token specs declared on head graph LCs."""
    discovered: list[ContextualTokenConstraintSpec] = []
    for lc in getattr(graph, "logicalConstrains", {}).values():
        if not getattr(lc, "headLC", True):
            continue
        discovered.extend(getattr(lc, "_generation_contextual_dfa_specs", ()))
    return tuple(discovered)


def _context_values(context: Mapping[str, Any], spec: ContextualTokenConstraintSpec):
    if spec.context_key not in context or context.get(spec.context_key) is None:
        return None
    value = context.get(spec.context_key)
    if isinstance(value, str):
        return frozenset({value})
    try:
        return frozenset(str(item) for item in value)
    except TypeError as exc:
        raise TypeError(
            f"context value {spec.context_key!r} must be a string or iterable"
        ) from exc


class ContextualDFA:
    """A base DFA specialized by graph-declared per-example token facts."""

    def __init__(self, base_dfa, specs, context: Mapping[str, Any]):
        self.base_dfa = base_dfa
        self.specs = tuple(specs)
        self.context = context
        self.alphabet = base_dfa.alphabet
        allowed_values = [(spec, _context_values(context, spec)) for spec in self.specs]
        self._forbidden_by_spec: list[frozenset[int]] = []
        for spec, values in allowed_values:
            if values is None and spec.allow_missing_context:
                self._forbidden_by_spec.append(frozenset())
                continue
            values = values or frozenset()
            self._forbidden_by_spec.append(frozenset(
                label for label, required in spec.label_to_value if required not in values
            ))
        self._trigger_bit_by_spec: list[int | None] = []
        trigger_count = 0
        for spec in self.specs:
            if spec.trigger_labels:
                self._trigger_bit_by_spec.append(trigger_count)
                trigger_count += 1
            else:
                self._trigger_bit_by_spec.append(None)
        mask_count = 1 << trigger_count
        self.states = frozenset(
            (state, mask) for state in base_dfa.states for mask in range(mask_count)
        )
        self.transitions = {}
        self.start_state = (base_dfa.start_state, 0)
        self.accepting_states = frozenset(
            (state, mask)
            for state in base_dfa.accepting_states
            for mask in range(mask_count)
        )
        base_dead = getattr(base_dfa, "dead_states", frozenset())
        self.dead_states = frozenset(
            (state, mask) for state in base_dead for mask in range(mask_count)
        )
        self.forbidden_symbols = frozenset().union(
            *(
                forbidden
                for spec, forbidden in zip(self.specs, self._forbidden_by_spec)
                if not spec.trigger_labels
            )
        )
        self._allowed_cache: dict[tuple[State, int | None], frozenset[Symbol]] = {}
        self._reach_cache: dict[tuple[State, int | None], bool] = {}

    def step(self, state: State, symbol: Symbol) -> State | None:
        try:
            base_state, pending_mask = state
        except (TypeError, ValueError):
            return None
        for spec, forbidden, trigger_bit in zip(
            self.specs, self._forbidden_by_spec, self._trigger_bit_by_spec
        ):
            active = trigger_bit is None or bool(pending_mask & (1 << trigger_bit))
            if active and symbol in forbidden:
                return None
        next_base = self.base_dfa.step(base_state, symbol)
        if next_base is None:
            return None
        next_mask = 0
        for spec, trigger_bit in zip(self.specs, self._trigger_bit_by_spec):
            if symbol in spec.trigger_labels:
                next_mask |= 1 << trigger_bit
        return next_base, next_mask

    def is_accepting(self, state: State) -> bool:
        return state in self.accepting_states

    def accepts(self, sequence: Iterable[Symbol]) -> bool:
        state = self.start_state
        for symbol in sequence:
            state = self.step(state, symbol)
            if state is None:
                return False
        return self.is_accepting(state)

    def can_reach_accepting(self, state: State, max_steps: int | None = None) -> bool:
        limit = None if max_steps is None else int(max_steps)
        key = (state, limit)
        if key in self._reach_cache:
            return self._reach_cache[key]
        if state in self.dead_states or (limit is not None and limit < 0):
            self._reach_cache[key] = False
            return False
        if self.is_accepting(state):
            self._reach_cache[key] = True
            return True
        queue = deque([(state, 0)])
        seen = {state}
        while queue:
            current, depth = queue.popleft()
            if limit is not None and depth >= limit:
                continue
            try:
                base_current, _pending_mask = current
            except (TypeError, ValueError):
                continue
            for symbol in self.base_dfa.allowed_tokens(
                base_current, remaining_steps=None
            ):
                nxt = self.step(current, symbol)
                if nxt is None or nxt in seen or nxt in self.dead_states:
                    continue
                if self.is_accepting(nxt):
                    self._reach_cache[key] = True
                    return True
                seen.add(nxt)
                queue.append((nxt, depth + 1))
        self._reach_cache[key] = False
        return False

    def allowed_tokens(self, state: State, remaining_steps: int | None = None) -> set[Symbol]:
        limit = None if remaining_steps is None else int(remaining_steps)
        key = (state, limit)
        cached = self._allowed_cache.get(key)
        if cached is not None:
            return set(cached)
        try:
            base_state, _pending_mask = state
        except (TypeError, ValueError):
            return set()
        base_allowed = self.base_dfa.allowed_tokens(
            base_state, remaining_steps=remaining_steps
        )
        allowed = set()
        for symbol in base_allowed:
            nxt = self.step(state, symbol)
            if nxt is None:
                continue
            next_limit = None if limit is None else limit - 1
            if self.can_reach_accepting(nxt, next_limit):
                allowed.add(symbol)
        self._allowed_cache[key] = frozenset(allowed)
        return allowed

    def __getattr__(self, name):
        return getattr(self.base_dfa, name)


def bind_contextual_dfa(base_dfa, graph, context: Mapping[str, Any]):
    """Bind all contextual graph constraints to one example's facts."""
    specs = discover_contextual_token_constraints(graph)
    if not specs:
        return base_dfa
    return ContextualDFA(base_dfa, specs, context)


__all__ = [
    "ContextualDFA",
    "ContextualTokenConstraintSpec",
    "bind_contextual_dfa",
    "declare_contextual_token_constraint",
    "discover_contextual_token_constraints",
    "mark_for_contextual_dfa",
]
