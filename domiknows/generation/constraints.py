"""Declarative generation constraints for DomiKnowS-guided text generation.

Each constraint can be compiled into one of two representations:

* **DFA** (default) — a :class:`~.learners.DFA` over the vocabulary's label
  alphabet that accepts exactly the sequences satisfying the constraint.  DFA
  constraints can be combined via :func:`constraints_to_dfa`, which takes their
  product (intersection).
* **DomiKnowS logical constraint** — a DomiKnowS expression that encodes the
  same property for training-time loss computation.  Only supported by a
  subset of constraint classes (``supports_domiknows = True``).

Constraint classes
------------------
- :class:`EosClosureConstraint`  — once EOS is produced, all later tokens must also be EOS.
- :class:`MaxNonEosConstraint`   — caps the total number of non-EOS tokens.
- :class:`RequiredTokenConstraint` — requires a token to appear at least *n* times.
- :class:`ForbiddenTokenConstraint` — forbids a specific token.
- :class:`OrderedTokensConstraint` — enforces an appearance ordering over a list of tokens.
- :class:`ConditionalMaxNonEosConstraint` — caps non-EOS count only when a trigger token appears.
- :class:`TokenSetCountConstraint` — counts tokens matching a finite token set or its complement.
- :class:`AfterTokenAllowedConstraint` — after a trigger token, only an allowed token set may appear.
- :class:`ComplementGenerationConstraint` — accepts exactly when its child rejects.
- :class:`AllOfGenerationConstraint` — accepts sequences satisfying every child constraint.
- :class:`AnyOfGenerationConstraint` — accepts sequences satisfying at least one child constraint.

Convenience factory functions
-----------------------------
:func:`no_token_after_eos`, :func:`max_non_eos`, :func:`required_token`,
:func:`forbidden_token`, :func:`ordered_tokens`,
:func:`if_token_present_then_at_most_non_eos`

To obtain a single combined DFA from a collection of constraints use
:func:`constraints_to_dfa`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from .learners.dfa.core import DFA, complement_dfa, product_dfa, union_dfa
from .vocabulary import TokenVocabulary


class DomiKnowSGenerationContext(Protocol):
    """Protocol for DomiKnowS compilation context.

    Constraint classes call methods on this object when
    :meth:`GenerationConstraint.apply_domiknows` is invoked.  Implementations
    are responsible for translating the semantic queries below into concrete
    DomiKnowS logical expressions.

    Attributes:
        vocabulary: The :class:`~.vocabulary.TokenVocabulary` for the current
            generation task.
    """

    vocabulary: TokenVocabulary

    def token_value(self, token: str, variable: str, path=None): ...
    """Return a DomiKnowS predicate that is true when *variable* holds *token*."""

    def non_eos(self, variable: str): ...
    """Return a DomiKnowS predicate that is true when *variable* is not EOS."""


class GenerationConstraint:
    """Abstract base class for all generation constraints.

    Subclasses must implement :meth:`to_dfa` when ``supports_dfa = True``
    and :meth:`apply_domiknows` when ``supports_domiknows = True``.

    Class attributes:
        supports_dfa (bool): Whether :meth:`to_dfa` is implemented.  Almost
            all constraints set this to ``True``.
        supports_domiknows (bool): Whether :meth:`apply_domiknows` is
            implemented.  Defaults to ``False``.
        name (str): Human-readable description of the constraint, used in
            error messages and logging.
    """

    supports_dfa = True
    supports_domiknows = False
    name = "generation constraint"

    def to_dfa(self, vocabulary: TokenVocabulary) -> DFA:
        """Compile this constraint into a :class:`~.learners.DFA`.

        Args:
            vocabulary: The vocabulary whose label alphabet the DFA should
                operate over.

        Returns:
            A :class:`~.learners.DFA` that accepts all and only the label
            sequences satisfying this constraint.

        Raises:
            NotImplementedError: If the subclass has not implemented DFA
                compilation.
        """
        raise NotImplementedError

    def apply_domiknows(self, context: DomiKnowSGenerationContext):
        """Compile this constraint into a DomiKnowS logical expression.

        Args:
            context: A :class:`DomiKnowSGenerationContext` that provides the
                vocabulary and predicate builders.

        Returns:
            A DomiKnowS logical constraint expression.

        Raises:
            NotImplementedError: If this constraint does not support DomiKnowS
                compilation (``supports_domiknows = False``).
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support DomiKnowS compilation")


def _complete_transitions(states, alphabet, fn):
    """Build a complete DFA transition table from a step function.

    Args:
        states: Iterable of state identifiers.
        alphabet: Iterable of symbol identifiers.
        fn: Callable ``(state, symbol) -> next_state`` defining the transition.

    Returns:
        A dict mapping ``(state, symbol)`` pairs to next states, covering
        every combination of state and symbol.
    """
    return {(state, symbol): fn(state, symbol) for state in states for symbol in alphabet}


@dataclass(frozen=True)
class EosClosureConstraint(GenerationConstraint):
    """Once EOS is generated, every later token must also be EOS.

    DFA states:
    - ``"open"``  — no EOS seen yet; all tokens allowed.
    - ``"eos"``   — EOS has been generated; only EOS is allowed.
    - ``"dead"``  — a non-EOS token appeared after an EOS (constraint violated).

    Attributes:
        name (str): Human-readable constraint description.
        supports_domiknows (bool): ``True`` — DomiKnowS compilation is available.
    """

    name: str = "no non-EOS tokens can follow an EOS token"
    supports_domiknows = True

    def to_dfa(self, vocabulary: TokenVocabulary) -> DFA:
        """Build the EOS-closure DFA.

        Args:
            vocabulary: Token vocabulary providing the label alphabet and
                :attr:`~.vocabulary.TokenVocabulary.eos_label`.

        Returns:
            A three-state DFA that rejects any sequence containing a
            non-EOS token after an EOS token.
        """
        alphabet = frozenset(vocabulary.alphabet)
        eos = vocabulary.eos_label
        states = frozenset({"open", "eos", "dead"})

        def step(state, symbol):
            # Absorbing dead state — stay dead once violated.
            if state == "dead":
                return "dead"
            # After EOS only EOS is allowed; anything else kills the sequence.
            if state == "eos":
                return "eos" if symbol == eos else "dead"
            # Open: first EOS transitions to the "eos" state.
            return "eos" if symbol == eos else "open"

        return DFA(
            states=states,
            alphabet=alphabet,
            transitions=_complete_transitions(states, alphabet, step),
            start_state="open",
            accepting_states=frozenset({"open", "eos"}),
            dead_states=frozenset({"dead"}),
        )

    def apply_domiknows(self, context: DomiKnowSGenerationContext):
        """Return a DomiKnowS ``ifL`` expression enforcing EOS closure.

        Encodes: for every pair (x, y) where x is before y, if x is EOS then
        y must also be EOS.

        Args:
            context: Active :class:`DomiKnowSGenerationContext`.

        Returns:
            A DomiKnowS ``ifL`` logical expression.
        """
        from domiknows.graph.logicalConstrain import ifL

        before = context.is_before_rel("before")
        return ifL(
            before,
            ifL(
                context.token_value(
                    context.vocabulary.eos_token,
                    "x",
                    path=("before", context.first_token),
                ),
                context.token_value(
                    context.vocabulary.eos_token,
                    "y",
                    path=("before", context.second_token),
                ),
            ),
        )


@dataclass(frozen=True)
class MaxNonEosConstraint(GenerationConstraint):
    """Cap the total number of non-EOS tokens in a generated sequence.

    DFA states are integers ``0 … max_count+1`` where state *i* means *i*
    non-EOS tokens have been seen.  State ``max_count+1`` is a dead state
    (one token over the budget).

    Attributes:
        max_count (int): Maximum number of non-EOS tokens allowed (≥ 0).
        name (str | None): Human-readable description; auto-generated when
            ``None``.
        supports_domiknows (bool): ``True`` — uses ``atMostAL``.
    """

    max_count: int
    name: str | None = None
    supports_domiknows = True

    def __post_init__(self):
        """Validate *max_count* and auto-generate *name* if needed."""
        if self.max_count < 0:
            raise ValueError("max_count must be non-negative")
        if self.name is None:
            object.__setattr__(self, "name", f"at most {self.max_count} non-EOS tokens are generated")

    def to_dfa(self, vocabulary: TokenVocabulary) -> DFA:
        """Build a counting DFA that rejects sequences with too many non-EOS tokens.

        Args:
            vocabulary: Token vocabulary supplying alphabet and EOS label.

        Returns:
            A DFA with ``max_count + 2`` states (0 … max_count inclusive,
            plus one dead state at ``max_count + 1``).
        """
        alphabet = frozenset(vocabulary.alphabet)
        eos = vocabulary.eos_label
        # State max_count+1 is the single dead/reject state.
        dead = self.max_count + 1
        states = frozenset(range(dead + 1))

        def step(state, symbol):
            if state == dead:
                return dead
            # EOS tokens are free — do not increment the counter.
            if symbol == eos:
                return state
            # Non-EOS token: advance counter, saturate at dead.
            return min(state + 1, dead)

        return DFA(
            states=states,
            alphabet=alphabet,
            transitions=_complete_transitions(states, alphabet, step),
            start_state=0,
            accepting_states=frozenset(range(self.max_count + 1)),
            dead_states=frozenset({dead}),
        )

    def apply_domiknows(self, context: DomiKnowSGenerationContext):
        """Return an ``atMostAL`` DomiKnowS expression.

        Args:
            context: Active :class:`DomiKnowSGenerationContext`.

        Returns:
            A DomiKnowS ``atMostAL`` expression over the non-EOS predicate.
        """
        from domiknows.graph.logicalConstrain import atMostAL

        return atMostAL(context.non_eos("x"), self.max_count)


@dataclass(frozen=True)
class RequiredTokenConstraint(GenerationConstraint):
    """Require a specific token to appear at least *min_count* times.

    DFA states are integers ``0 … min_count`` counting occurrences of *token*.
    The single accepting state is ``min_count`` (the budget is saturated — the
    counter never exceeds it since further matches are no-ops).

    Attributes:
        token (str): The surface-form token that must appear.
        min_count (int): Required minimum number of occurrences (≥ 1).
        name (str | None): Human-readable description; auto-generated when
            ``None``.
        supports_domiknows (bool): ``True`` — uses ``atLeastAL``.
    """

    token: str
    min_count: int = 1
    name: str | None = None
    supports_domiknows = True

    def __post_init__(self):
        """Validate *min_count* and auto-generate *name* if needed."""
        if self.min_count < 1:
            raise ValueError("min_count must be at least 1")
        if self.name is None:
            object.__setattr__(self, "name", f"at least {self.min_count} {self.token!r} token(s) are generated")

    def to_dfa(self, vocabulary: TokenVocabulary) -> DFA:
        """Build a counting DFA that accepts only when *token* has appeared
        at least *min_count* times.

        Args:
            vocabulary: Token vocabulary used to resolve *token* to a label ID.

        Returns:
            A DFA with ``min_count + 1`` states; only state ``min_count`` is
            accepting.
        """
        target = vocabulary.label_for_token(self.token)
        alphabet = frozenset(vocabulary.alphabet)
        states = frozenset(range(self.min_count + 1))

        def step(state, symbol):
            # Advance counter on matching token; saturate at min_count.
            if symbol == target:
                return min(state + 1, self.min_count)
            return state

        return DFA(
            states=states,
            alphabet=alphabet,
            transitions=_complete_transitions(states, alphabet, step),
            start_state=0,
            accepting_states=frozenset({self.min_count}),
        )

    def apply_domiknows(self, context: DomiKnowSGenerationContext):
        """Return an ``atLeastAL`` DomiKnowS expression.

        Args:
            context: Active :class:`DomiKnowSGenerationContext`.

        Returns:
            A DomiKnowS ``atLeastAL`` expression asserting the token appears
            at least *min_count* times.
        """
        from domiknows.graph.logicalConstrain import atLeastAL

        return atLeastAL(context.token_value(self.token, "x"), self.min_count)


@dataclass(frozen=True)
class ForbiddenTokenConstraint(GenerationConstraint):
    """Forbid a specific token from appearing anywhere in the output.

    DFA states:
    - ``"ok"``   — forbidden token not yet seen; generation may continue.
    - ``"dead"`` — forbidden token was produced; constraint violated.

    Attributes:
        token (str): Surface-form token that must not appear.
        name (str | None): Human-readable description; auto-generated when
            ``None``.
        supports_domiknows (bool): ``True`` — uses ``atMostAL`` with 0.
    """

    token: str
    name: str | None = None
    supports_domiknows = True

    def __post_init__(self):
        """Auto-generate *name* if not provided."""
        if self.name is None:
            object.__setattr__(self, "name", f"token {self.token!r} is forbidden")

    def to_dfa(self, vocabulary: TokenVocabulary) -> DFA:
        """Build a two-state DFA that rejects any sequence containing *token*.

        Args:
            vocabulary: Token vocabulary used to resolve *token* to a label ID.

        Returns:
            A DFA with states ``{"ok", "dead"}``, accepting only from ``"ok"``.
        """
        target = vocabulary.label_for_token(self.token)
        alphabet = frozenset(vocabulary.alphabet)
        states = frozenset({"ok", "dead"})

        def step(state, symbol):
            # Seeing the forbidden token (or already dead) goes to dead.
            if state == "dead" or symbol == target:
                return "dead"
            return "ok"

        return DFA(
            states=states,
            alphabet=alphabet,
            transitions=_complete_transitions(states, alphabet, step),
            start_state="ok",
            accepting_states=frozenset({"ok"}),
            dead_states=frozenset({"dead"}),
        )

    def apply_domiknows(self, context: DomiKnowSGenerationContext):
        """Return an ``atMostAL(..., 0)`` DomiKnowS expression.

        Args:
            context: Active :class:`DomiKnowSGenerationContext`.

        Returns:
            A DomiKnowS ``atMostAL`` expression asserting zero occurrences.
        """
        from domiknows.graph.logicalConstrain import atMostAL

        return atMostAL(context.token_value(self.token, "x"), 0)


@dataclass(frozen=True)
class OrderedTokensConstraint(GenerationConstraint):
    """Require a list of tokens to appear in a specific relative order.

    Each token in *tokens* must appear at some point after the previous one.
    They do not need to be adjacent.  The constraint is satisfied once all
    tokens have appeared (in order).

    DFA states are integers ``0 … len(tokens)`` tracking how many tokens from
    the required sequence have been seen so far.  Only state ``len(tokens)``
    (all tokens matched) is accepting.

    Attributes:
        tokens (tuple[str, ...]): Ordered sequence of required surface-form
            tokens.
        name (str | None): Human-readable description; auto-generated when
            ``None``.
        supports_domiknows (bool): ``False`` — ordering constraints are not yet
            expressible via DomiKnowS logical constraints.
    """

    tokens: tuple[str, ...]
    name: str | None = None
    supports_domiknows = False

    def __init__(self, tokens: Iterable[str], name: str | None = None):
        """Initialise with an ordered sequence of tokens.

        Args:
            tokens: Iterable of surface-form tokens that must appear in order.
            name: Optional human-readable description; auto-generated if
                ``None``.

        Raises:
            ValueError: If *tokens* is empty.
        """
        tokens = tuple(tokens)
        if not tokens:
            raise ValueError("tokens must not be empty")
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "name", name or f"tokens appear in order: {tokens!r}")

    def to_dfa(self, vocabulary: TokenVocabulary) -> DFA:
        """Build an order-tracking DFA.

        The DFA walks through the required token sequence left-to-right,
        advancing the state each time the next expected token is produced.
        Non-matching tokens leave the state unchanged.

        Args:
            vocabulary: Token vocabulary used to map each token to a label ID.

        Returns:
            A DFA with ``len(tokens) + 1`` states; only the final state is
            accepting.
        """
        # Resolve each surface-form token to its vocabulary label ID up front.
        targets = tuple(vocabulary.label_for_token(token) for token in self.tokens)
        alphabet = frozenset(vocabulary.alphabet)
        states = frozenset(range(len(targets) + 1))

        def step(state, symbol):
            # Advance when the next expected token is produced.
            if state < len(targets) and symbol == targets[state]:
                return state + 1
            # All other symbols leave the progress counter unchanged.
            return state

        return DFA(
            states=states,
            alphabet=alphabet,
            transitions=_complete_transitions(states, alphabet, step),
            start_state=0,
            accepting_states=frozenset({len(targets)}),
        )


@dataclass(frozen=True)
class ConditionalMaxNonEosConstraint(GenerationConstraint):
    """Cap non-EOS tokens only when a trigger token appears in the output.

    Semantics: *if* ``token`` is present in the generated sequence *then* the
    total number of non-EOS tokens must not exceed ``max_count``.

    DFA state is a tuple ``(seen: bool, count: int)`` plus a single dead state
    ``("dead", max_count+1)``:
    - *seen* — whether ``token`` has been observed.
    - *count* — number of non-EOS tokens produced so far.

    The constraint is only enforced (``seen=True, count > max_count → dead``)
    after the trigger token appears.

    Attributes:
        token (str): Trigger token whose presence activates the cap.
        max_count (int): Maximum non-EOS tokens allowed when trigger is seen.
        name (str | None): Human-readable description; auto-generated when
            ``None``.
        supports_domiknows (bool): ``True`` — uses ``ifL(existsAL, atMostAL)``.
    """

    token: str
    max_count: int
    name: str | None = None
    supports_domiknows = True

    def __post_init__(self):
        """Validate *max_count* and auto-generate *name* if needed."""
        if self.max_count < 0:
            raise ValueError("max_count must be non-negative")
        if self.name is None:
            object.__setattr__(
                self,
                "name",
                f"if {self.token!r} appears then at most {self.max_count} non-EOS tokens are generated",
            )

    def to_dfa(self, vocabulary: TokenVocabulary) -> DFA:
        """Build the conditional counting DFA.

        States are ``(seen: bool, count: int)`` pairs for all combinations of
        seen ∈ {False, True} and count ∈ {0 … max_count+1}, plus a dead state.

        Args:
            vocabulary: Token vocabulary supplying label IDs and EOS label.

        Returns:
            A DFA that accepts exactly when either the trigger has not been
            seen or the non-EOS count is within budget.
        """
        token_label = vocabulary.label_for_token(self.token)
        eos_label = vocabulary.eos_label
        alphabet = frozenset(vocabulary.alphabet)
        # Use a sentinel tuple as the dead/reject state to avoid collisions.
        dead = ("dead", self.max_count + 1)
        states = {dead}
        # Enumerate all (seen, count) states including one overflow level.
        for seen in (False, True):
            for count in range(self.max_count + 2):
                states.add((seen, count))

        def step(state, symbol):
            if state == dead:
                return dead
            seen, count = state
            # Flip seen flag if the trigger token is produced.
            seen = seen or symbol == token_label
            # Increment counter for every non-EOS token.
            if symbol != eos_label:
                count += 1
            # Once trigger is seen and budget is exceeded, reject.
            if seen and count > self.max_count:
                return dead
            return (seen, count)

        # Every state except dead is accepting.
        accepting = {state for state in states if state != dead}
        return DFA(
            states=frozenset(states),
            alphabet=alphabet,
            transitions=_complete_transitions(states, alphabet, step),
            start_state=(False, 0),
            accepting_states=frozenset(accepting),
            dead_states=frozenset({dead}),
        )

    def apply_domiknows(self, context: DomiKnowSGenerationContext):
        """Return an ``ifL(existsAL, atMostAL)`` DomiKnowS expression.

        Encodes: if there exists a position where the trigger token is produced,
        then at most *max_count* non-EOS tokens are produced overall.

        Args:
            context: Active :class:`DomiKnowSGenerationContext`.

        Returns:
            A DomiKnowS ``ifL`` logical expression.
        """
        from domiknows.graph.logicalConstrain import atMostAL, existsAL, ifL

        return ifL(
            existsAL(context.token_value(self.token, "x")),
            atMostAL(context.non_eos("y"), self.max_count),
        )


@dataclass(frozen=True)
class TokenSetCountConstraint(GenerationConstraint):
    """Count occurrences of a token-set predicate.

    The counted predicate is ``symbol in tokens`` by default, or
    ``symbol not in tokens`` when ``negated=True``.  Any combination of
    ``min_count`` and ``max_count`` may be supplied; ``exact_count`` is a
    convenience that sets both bounds to the same value.
    """

    tokens: tuple[str, ...]
    min_count: int | None = None
    max_count: int | None = None
    negated: bool = False
    name: str | None = None
    supports_domiknows = False

    def __init__(
        self,
        tokens: Iterable[str],
        *,
        min_count: int | None = None,
        max_count: int | None = None,
        exact_count: int | None = None,
        negated: bool = False,
        name: str | None = None,
    ):
        tokens = tuple(tokens)
        if exact_count is not None:
            if min_count is not None or max_count is not None:
                raise ValueError("exact_count cannot be combined with min_count or max_count")
            min_count = exact_count
            max_count = exact_count
        if min_count is None and max_count is None:
            raise ValueError("at least one count bound is required")
        if min_count is not None and min_count < 0:
            raise ValueError("min_count must be non-negative")
        if max_count is not None and max_count < 0:
            raise ValueError("max_count must be non-negative")
        if min_count is not None and max_count is not None and min_count > max_count:
            raise ValueError("min_count cannot exceed max_count")
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "min_count", min_count)
        object.__setattr__(self, "max_count", max_count)
        object.__setattr__(self, "negated", negated)
        if name is None:
            predicate = f"not in {tokens!r}" if negated else f"in {tokens!r}"
            bounds = []
            if min_count is not None:
                bounds.append(f"at least {min_count}")
            if max_count is not None:
                bounds.append(f"at most {max_count}")
            name = f"{' and '.join(bounds)} token(s) {predicate}"
        object.__setattr__(self, "name", name)

    def to_dfa(self, vocabulary: TokenVocabulary) -> DFA:
        """Build a finite counter DFA for this token-set predicate."""
        token_labels = frozenset(vocabulary.label_for_token(token) for token in self.tokens)
        alphabet = frozenset(vocabulary.alphabet)
        max_count = self.max_count
        min_count = self.min_count or 0
        if max_count is None:
            max_state = min_count
            dead = None
        else:
            max_state = max_count + 1
            dead = max_state
        states = frozenset(range(max_state + 1))

        def matches(symbol):
            in_set = symbol in token_labels
            return not in_set if self.negated else in_set

        def step(state, symbol):
            if dead is not None and state == dead:
                return dead
            if not matches(symbol):
                return state
            if max_count is None:
                return min(state + 1, min_count)
            return min(state + 1, dead)

        accepting = {
            state
            for state in states
            if state >= min_count and (max_count is None or state <= max_count)
        }
        return DFA(
            states=states,
            alphabet=alphabet,
            transitions=_complete_transitions(states, alphabet, step),
            start_state=0,
            accepting_states=frozenset(accepting),
            dead_states=frozenset({dead}) if dead is not None else frozenset(),
        )


@dataclass(frozen=True)
class AfterTokenAllowedConstraint(GenerationConstraint):
    """Require all tokens after a trigger token to be in an allowed set."""

    trigger_tokens: tuple[str, ...]
    allowed_tokens: tuple[str, ...]
    name: str | None = None
    supports_domiknows = False

    def __init__(
        self,
        trigger_tokens: Iterable[str],
        allowed_tokens: Iterable[str],
        name: str | None = None,
    ):
        trigger_tokens = tuple(trigger_tokens)
        allowed_tokens = tuple(allowed_tokens)
        if not trigger_tokens:
            raise ValueError("trigger_tokens must not be empty")
        if not allowed_tokens:
            raise ValueError("allowed_tokens must not be empty")
        object.__setattr__(self, "trigger_tokens", trigger_tokens)
        object.__setattr__(self, "allowed_tokens", allowed_tokens)
        object.__setattr__(
            self,
            "name",
            name or f"after {trigger_tokens!r}, only {allowed_tokens!r} may appear",
        )

    def to_dfa(self, vocabulary: TokenVocabulary) -> DFA:
        """Build a three-state DFA for this suffix restriction."""
        triggers = frozenset(vocabulary.label_for_token(token) for token in self.trigger_tokens)
        allowed = frozenset(vocabulary.label_for_token(token) for token in self.allowed_tokens)
        alphabet = frozenset(vocabulary.alphabet)
        states = frozenset({"open", "after", "dead"})

        def step(state, symbol):
            if state == "dead":
                return "dead"
            if state == "after":
                return "after" if symbol in allowed else "dead"
            return "after" if symbol in triggers else "open"

        return DFA(
            states=states,
            alphabet=alphabet,
            transitions=_complete_transitions(states, alphabet, step),
            start_state="open",
            accepting_states=frozenset({"open", "after"}),
            dead_states=frozenset({"dead"}),
        )


@dataclass(frozen=True)
class ComplementGenerationConstraint(GenerationConstraint):
    """Negation of a DFA-compilable child generation constraint."""

    child: GenerationConstraint
    name: str | None = None
    supports_domiknows = False

    def __init__(self, child: GenerationConstraint, name: str | None = None):
        object.__setattr__(self, "child", child)
        object.__setattr__(self, "name", name or f"not ({child.name})")

    def to_dfa(self, vocabulary: TokenVocabulary) -> DFA:
        """Compile the child DFA and complement it."""
        return complement_dfa(self.child.to_dfa(vocabulary))


@dataclass(frozen=True)
class AllOfGenerationConstraint(GenerationConstraint):
    """Conjunction of several generation constraints.

    The compiled DFA is the intersection/product of the child DFAs.  This is
    useful when a graph-level boolean formula needs to preserve a grouped
    ``andL`` branch, for example inside an ``orL``.
    """

    children: tuple[GenerationConstraint, ...]
    name: str | None = None
    supports_domiknows = False

    def __init__(self, children: Iterable[GenerationConstraint], name: str | None = None):
        children = tuple(children)
        if not children:
            raise ValueError("children must not be empty")
        object.__setattr__(self, "children", children)
        object.__setattr__(self, "name", name or "all generation constraints hold")

    def to_dfa(self, vocabulary: TokenVocabulary) -> DFA:
        """Compile child constraints and intersect their DFAs."""
        return product_dfa(child.to_dfa(vocabulary) for child in self.children if child.supports_dfa)


@dataclass(frozen=True)
class AnyOfGenerationConstraint(GenerationConstraint):
    """Disjunction of several generation constraints.

    The compiled DFA is the union of the child DFAs.  A sequence is accepted
    when at least one child constraint accepts it.
    """

    children: tuple[GenerationConstraint, ...]
    name: str | None = None
    supports_domiknows = False

    def __init__(self, children: Iterable[GenerationConstraint], name: str | None = None):
        children = tuple(children)
        if not children:
            raise ValueError("children must not be empty")
        object.__setattr__(self, "children", children)
        object.__setattr__(self, "name", name or "at least one generation constraint holds")

    def to_dfa(self, vocabulary: TokenVocabulary) -> DFA:
        """Compile child constraints and union their DFAs."""
        return union_dfa(child.to_dfa(vocabulary) for child in self.children if child.supports_dfa)


def no_token_after_eos() -> EosClosureConstraint:
    """Return a constraint that forbids non-EOS tokens after the first EOS."""
    return EosClosureConstraint()


def max_non_eos(max_count: int) -> MaxNonEosConstraint:
    """Return a constraint capping the total number of non-EOS tokens.

    Args:
        max_count: Maximum number of non-EOS tokens allowed (≥ 0).
    """
    return MaxNonEosConstraint(max_count)


def required_token(token: str, min_count: int = 1) -> RequiredTokenConstraint:
    """Return a constraint requiring *token* to appear at least *min_count* times.

    Args:
        token: Surface-form token that must be present.
        min_count: Required minimum occurrences (≥ 1).  Defaults to ``1``.
    """
    return RequiredTokenConstraint(token, min_count=min_count)


def forbidden_token(token: str) -> ForbiddenTokenConstraint:
    """Return a constraint that forbids *token* from appearing anywhere.

    Args:
        token: Surface-form token to forbid.
    """
    return ForbiddenTokenConstraint(token)


def ordered_tokens(tokens: Iterable[str]) -> OrderedTokensConstraint:
    """Return a constraint requiring *tokens* to appear in the given order.

    Args:
        tokens: Iterable of surface-form tokens that must appear in order.
    """
    return OrderedTokensConstraint(tokens)


def if_token_present_then_at_most_non_eos(token: str, max_count: int) -> ConditionalMaxNonEosConstraint:
    """Return a constraint capping non-EOS tokens when *token* appears.

    Args:
        token: Trigger token whose presence activates the cap.
        max_count: Maximum non-EOS tokens allowed once *token* appears.
    """
    return ConditionalMaxNonEosConstraint(token, max_count)


def all_of_constraints(constraints: Iterable[GenerationConstraint]) -> AllOfGenerationConstraint:
    """Return a composite constraint requiring all *constraints* to hold."""
    return AllOfGenerationConstraint(constraints)


def any_of_constraints(constraints: Iterable[GenerationConstraint]) -> AnyOfGenerationConstraint:
    """Return a composite constraint requiring at least one child to hold."""
    return AnyOfGenerationConstraint(constraints)


def constraints_to_dfa(constraints: Iterable[GenerationConstraint], vocabulary: TokenVocabulary) -> DFA:
    """Combine multiple constraints into a single intersection DFA.

    Compiles each constraint that supports DFA representation and takes their
    product (intersection) via :func:`~.learners.product_dfa`.  A sequence is
    accepted by the resulting DFA if and only if it satisfies *all* constraints.

    When no constraints support DFA compilation, returns a trivial
    accept-all DFA (single state ``"ok"``) so callers never receive ``None``.

    Args:
        constraints: Iterable of :class:`GenerationConstraint` instances.
            Constraints with ``supports_dfa = False`` are silently skipped.
        vocabulary: Token vocabulary providing the shared label alphabet.

    Returns:
        A single :class:`~.learners.DFA` accepting sequences that satisfy all
        DFA-compilable constraints in *constraints*.
    """
    dfas = [constraint.to_dfa(vocabulary) for constraint in constraints if constraint.supports_dfa]
    if not dfas:
        # No DFA-compilable constraints — return a trivial accept-all DFA.
        alphabet = frozenset(vocabulary.alphabet)
        return DFA(
            states=frozenset({"ok"}),
            alphabet=alphabet,
            transitions={("ok", symbol): "ok" for symbol in alphabet},
            start_state="ok",
            accepting_states=frozenset({"ok"}),
        )
    # Intersect all individual DFAs into one combined automaton.
    return product_dfa(dfas)
