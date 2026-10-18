"""Domain-independent compilation of grounded Boolean LCs for decoding.

The compiler knows logical structure and predicate identities. An adapter owns
token interpretation, semantic state and the events at which predicates become
known. No vocabulary, scene, action, or end-token convention is built in.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Hashable, Iterable, Mapping, Protocol

from ._lc_normalize import kind
from .core import DFA, Symbol
from .graph_discovery import _concept_tuple, _is_v
from .semantic import MissingSemanticBinding
from .productive import BoundedReachability


class PredicateConstraintCompilationError(ValueError):
    """A selected LC cannot be compiled using the supplied predicate bindings."""


class PredicateGroundingLimitExceeded(RuntimeError):
    """An event exceeded its exact grounding budget; satisfaction is unknown."""


@dataclass(frozen=True)
class PredicateBinding:
    concept: Any
    name: str
    domain: str
    phase: str = "immediate"


@dataclass(frozen=True)
class CompiledPredicateConstraint:
    name: str
    domain: str
    expression: tuple
    predicates: frozenset[str]
    phase: str = "immediate"
    variables: tuple[str, ...] = ()
    free_variables: tuple[str, ...] = ()


@dataclass(frozen=True)
class PredicateEvent:
    """A complete valuation at one phase, optionally over finite variable domains.

With variable_domains, observations use (predicate_name, entity, ...) keys.
Without it, legacy name -> bool observations describe one unary grounding.
Omitted tuples are missing observations, never implicitly false.
"""

    domain: str
    values: Mapping[str | tuple, bool]
    phase: str = "immediate"
    variable_domains: Mapping[str, Iterable[Hashable]] | None = None


def _assignments(variables, domains, assignment):
    for entities in product(*(domains[name] for name in variables)):
        yield {**assignment, **dict(zip(variables, entities))}


def _evaluate(expression, values, assignment=None, variable_domains=None, *, _budget=None):
    operator, *args = expression
    if operator == "predicate":
        if _budget is not None:
            _budget[0] -= 1
            if _budget[0] < 0:
                raise PredicateGroundingLimitExceeded("predicate grounding limit exceeded; satisfaction is unknown")
        key = args[0] if assignment is None or len(args) == 1 else (args[0], *(assignment[v] for v in args[1]))
        if key not in values or not isinstance(values[key], bool):
            raise MissingSemanticBinding(f"requires Boolean predicate {key!r}")
        return values[key]
    if operator == "count":
        count_operator, limit, operands = args
        count = 0
        for variables, child in operands:
            groundings = (None,) if assignment is None else _assignments(variables, variable_domains, assignment)
            for grounding in groundings:
                count += _evaluate(child, values, grounding, variable_domains, _budget=_budget)
        if count_operator == "atMostL":
            return count <= limit
        if count_operator in {"atLeastL", "existsL"}:
            return count >= limit
        return count == limit
    children = [_evaluate(arg, values, assignment, variable_domains, _budget=_budget) for arg in args]
    if operator == "notL":
        return not children[0]
    if operator in {"ifL", "forAllL"}:
        return not children[0] or children[1]
    if operator in {"andL", "nandL"}:
        return all(children) if operator == "andL" else not all(children)
    if operator in {"orL", "norL"}:
        return any(children) if operator == "orL" else not any(children)
    if operator == "xorL":
        return sum(children) % 2 == 1
    if operator in {"equivalenceL", "iffL"}:
        return all(value == children[0] for value in children)
    raise AssertionError(operator)


_COUNT_OPERATORS = frozenset({"atMostL", "atLeastL", "exactL", "existsL"})
_BOOLEAN_OPERATORS = frozenset({"ifL", "notL", "andL", "orL", "nandL", "norL",
                                "xorL", "equivalenceL", "iffL", "forAllL"})


def _visible_variables(expression):
    """Variables outside local count scopes (before or after scope binding)."""
    operator, *args = expression
    if operator == "predicate":
        return set(args[1])
    if operator in _COUNT_OPERATORS or operator == "count":
        return set()
    return set().union(*(_visible_variables(child) for child in args))


def _bind_count_scopes(expression, outside=frozenset()):
    operator, *args = expression
    if operator == "predicate":
        return expression
    if operator in _COUNT_OPERATORS:
        limit, *children = args
        operands = []
        for child in children:
            variables = _visible_variables(child) - outside
            operands.append((tuple(sorted(variables)), _bind_count_scopes(child, outside | variables)))
        return ("count", operator, limit, tuple(operands))
    visible = [_visible_variables(child) for child in args]
    children = []
    for index, child in enumerate(args):
        sibling_variables = set().union(*(v for j, v in enumerate(visible) if index != j))
        children.append(_bind_count_scopes(child, outside | sibling_variables))
    return (operator, *children)


def _free_variables(expression):
    operator, *args = expression
    if operator == "predicate":
        return set(args[1])
    if operator == "count":
        return set().union(*(_free_variables(child) - set(variables) for variables, child in args[2]))
    return set().union(*(_free_variables(child) for child in args))


@dataclass(frozen=True)
class PredicateCompilationCoverage:
    compiled: tuple[str, ...]
    inactive: tuple[str, ...]


@dataclass(frozen=True)
class PredicateConstraintProgram:
    rules: tuple[CompiledPredicateConstraint, ...] = ()
    inactive_constraints: tuple[str, ...] = ()
    max_groundings: int = 10000

    def __post_init__(self):
        if isinstance(self.max_groundings, bool) or not isinstance(self.max_groundings, int) or self.max_groundings < 1:
            raise ValueError("max_groundings must be a positive integer")

    @property
    def coverage(self) -> PredicateCompilationCoverage:
        return PredicateCompilationCoverage(tuple(rule.name for rule in self.rules), self.inactive_constraints)

    def project(self, predicates, *, domain=None, phases=None):
        """Keep complete rules observable in a sound abstraction.

        Rules mentioning any omitted predicate remain in the concrete monitor;
        their absence here relaxes the program rather than inventing values.
        The caller must supply the same event lifecycle and retained predicate
        meanings as the concrete adapter.
        """
        names = frozenset(predicates)
        phases = None if phases is None else frozenset(phases)
        return PredicateConstraintProgram(
            tuple(rule for rule in self.rules if rule.predicates <= names
                  and (domain is None or rule.domain == domain)
                  and (phases is None or rule.phase in phases)),
            self.inactive_constraints, self.max_groundings,
        )

    def check(self, event: PredicateEvent) -> bool:
        domains = None
        for rule in self.rules:
            if rule.domain != event.domain or rule.phase != event.phase:
                continue
            try:
                if event.variable_domains is None:
                    if len(rule.variables) > 1:
                        raise MissingSemanticBinding(f"requires variable_domains for variables {rule.variables!r}")
                    assignments = (None,)
                else:
                    if domains is None:
                        domains = {}
                    for variable in rule.variables:
                        if variable in domains:
                            continue
                        if variable not in event.variable_domains:
                            raise MissingSemanticBinding(f"requires a finite domain for variable {variable!r}")
                        source = event.variable_domains[variable]
                        if not isinstance(source, (tuple, list, set, frozenset, range)):
                            raise MissingSemanticBinding(f"variable {variable!r} needs a finite collection of distinct hashable entities")
                        try:
                            oversized = len(source) > self.max_groundings
                        except OverflowError:
                            oversized = True
                        if oversized:
                            raise PredicateGroundingLimitExceeded(f"domain for {variable!r} exceeds the grounding limit; satisfaction is unknown")
                        try:
                            members = tuple(source)
                            unique = set(members)
                        except TypeError as error:
                            raise MissingSemanticBinding(f"variable {variable!r} has unhashable entities") from error
                        if len(unique) != len(members):
                            raise MissingSemanticBinding(f"variable {variable!r} has duplicate entities")
                        domains[variable] = members
                    assignments = _assignments(rule.free_variables, domains, {})
                budget = [self.max_groundings]
                satisfied = True
                for assignment in assignments:
                    # Evaluate every grounding, even after a known violation:
                    # an absent observation must not silently become False.
                    value = _evaluate(rule.expression, event.values, assignment, domains, _budget=budget)
                    satisfied = satisfied and value
            except (MissingSemanticBinding, PredicateGroundingLimitExceeded) as error:
                raise type(error)(f"constraint {rule.name!r} at {event.domain!r}/{event.phase!r}: {error}") from error
            if not satisfied:
                return False
        return True

    def check_events(self, events: Iterable[PredicateEvent]) -> bool:
        return all(self.check(event) for event in events)


def compile_predicate_constraints(graph, bindings: Iterable[PredicateBinding], *, max_groundings=10000) -> PredicateConstraintProgram:
    """Compile active head LCs using explicit concept-to-predicate bindings.

Supports Boolean connectives, forAllL (universal implication), and local/head
atMostL/atLeastL/exactL/existsL counts. Direct named unary and relation variables
share finite, explicitly supplied event domains. Free variables are universal;
variables introduced only within a count operand are local to that operand.
Scalar events remain compatible with formulas using one variable.
Each rule must belong to one event domain. Immediate predicates may appear in
a deferred formula, but mixing different deferred phases is unsupported. Paths,
value/filter LCs, multiclass atoms and unregistered predicates fail by LC name.
The graph and its descendant subgraphs are snapshotted: recompile after editing
their constraints. Ancestors and siblings are outside the selected scope.
"""
    registry = {}
    names = set()
    for binding in bindings:
        if not binding.name or not binding.domain or not binding.phase:
            raise ValueError("predicate name, domain and phase must be nonempty")
        if id(binding.concept) in registry or (binding.domain, binding.name) in names:
            raise ValueError(f"duplicate predicate binding {binding.name!r}")
        registry[id(binding.concept)] = binding
        names.add((binding.domain, binding.name))

    def relation_roles(concept):
        return tuple(concept.getOntologyGraph().findConceptInfo(concept).get("relationAttrs", {}))

    def collect_relations(node, aliases):
        for index, element in enumerate(node.e):
            if hasattr(element, "e"):
                collect_relations(element, aliases)
            atom = _concept_tuple(element)
            if atom is None or index + 1 >= len(node.e):
                continue
            variable = node.e[index + 1]
            if not _is_v(variable) or not isinstance(variable.relVarInfo, dict) or variable.v is not None:
                continue
            roles = relation_roles(atom[0])
            names = tuple(variable.relVarInfo)
            if not isinstance(variable.name, str) or not variable.name:
                raise ValueError("direct relation variables need a nonempty identifier")
            if not roles or len(names) != len(roles):
                raise ValueError("relation arguments must name each role exactly once")
            for name, role in zip(names, roles):
                key = (variable.name, role)
                if key in aliases and aliases[key] != name:
                    raise ValueError("ambiguous relation variable mapping")
                aliases[key] = name

    def projection(path, aliases, *, reversed_role=False):
        if not isinstance(path, tuple) or len(path) != 2 or not isinstance(path[0], str):
            return None
        role = path[1]
        if reversed_role:
            role = getattr(role, "reversed", None)
        try:
            return aliases.get((path[0], role))
        except TypeError:
            return None

    def parse(node, domains, variables, used, phases, aliases):
        operator = kind(node)
        if operator not in _BOOLEAN_OPERATORS | _COUNT_OPERATORS:
            raise ValueError(f"unsupported logical operator {operator}")
        elements = list(node.e)
        limit = None
        if operator in _COUNT_OPERATORS:
            trailing = elements.pop() if elements and type(elements[-1]) is int else None
            fixed = getattr(node, "fixedLimit", None)
            explicit = getattr(node, "_explicitLimit", None)
            if fixed is not None and trailing is not None:
                raise ValueError(f"{operator} has a fixed count and cannot take a trailing limit")
            if explicit is not None and trailing is not None:
                raise ValueError("count has both explicit and positional limits")
            if fixed is not None:
                limit = fixed
            elif explicit is not None:
                limit = explicit
            else:
                limit = trailing if trailing is not None else 1
            if type(limit) is not int or limit < 0:
                raise ValueError("count limit must be a nonnegative integer")
        args = []
        position = 0
        while position < len(elements):
            element = elements[position]
            position += 1
            atom = _concept_tuple(element)
            if atom is not None:
                binding = registry.get(id(atom[0]))
                if binding is None:
                    raise ValueError(f"unregistered predicate {getattr(atom[0], 'name', atom[0])!r}")
                variable = elements[position] if position < len(elements) else None
                position += 1
                if not _is_v(variable) or atom[2] is not None or atom[3] != 1:
                    raise ValueError("only direct named Boolean predicate bindings are supported")
                if variable.v is not None:
                    if variable.name:
                        raise ValueError("named path aliases are unsupported")
                    # Graph validation rewrites relation('x','y') consumers
                    # to role projections. Resolve only those exact joins;
                    # this is not arbitrary DataNode path traversal.
                    projected = projection(variable.v, aliases)
                    if projected is not None:
                        argument_variables = (projected,)
                    elif relation_roles(atom[0]) and isinstance(variable.v, tuple):
                        argument_variables = tuple(projection(p, aliases, reversed_role=True) for p in variable.v)
                        if len(argument_variables) != len(relation_roles(atom[0])) or None in argument_variables:
                            raise ValueError("unsupported relation argument path")
                    else:
                        raise ValueError("unsupported predicate path")
                elif variable.relVarInfo is None:
                    argument_variables = (variable.name,)
                else:
                    info = variable.relVarInfo
                    if not isinstance(info, dict):
                        raise ValueError("unsupported relation variable mapping")
                    argument_variables = tuple(info)
                    if (position < len(elements) and isinstance(elements[position], tuple)
                            and elements[position] and elements[position][0] == "VarMaps"):
                        metadata = elements[position]
                        position += 1
                        if len(metadata) != 2 or not isinstance(metadata[1], dict) or tuple(metadata[1]) != argument_variables:
                            raise ValueError("unsupported relation variable mapping")
                        for name, role in zip(info, relation_roles(atom[0])):
                            mapped = metadata[1][name]
                            if not _is_v(mapped) or mapped.v != (variable.name, role):
                                raise ValueError("unsupported relation argument path")
                if any(not isinstance(name, str) or not name for name in argument_variables):
                    raise ValueError("variable names must be nonempty strings")
                domains.add(binding.domain)
                variables.update(argument_variables)
                used.add(binding.name)
                phases.add(binding.phase)
                args.append(("predicate", binding.name, argument_variables))
            elif hasattr(element, "e"):
                args.append(parse(element, domains, variables, used, phases, aliases))
            else:
                raise ValueError(f"unsupported predicate or binding {element!r}")
        arity = {"notL": 1, "ifL": 2, "forAllL": 2}.get(operator)
        if (arity is not None and len(args) != arity) or not args:
            raise ValueError(f"invalid {operator} arity {len(args)}")
        return (operator, limit, *args) if operator in _COUNT_OPERATORS else (operator, *args)

    def constraints():
        pending, graphs_seen, constraints_seen = [graph], set(), set()
        while pending:
            current = pending.pop()
            if id(current) in graphs_seen:
                continue
            graphs_seen.add(id(current))
            for key, lc in current.logicalConstrains.items():
                if id(lc) not in constraints_seen:
                    constraints_seen.add(id(lc))
                    yield key, lc
            pending.extend(reversed(tuple(getattr(current, "subgraphs", {}).values())))

    rules, inactive = [], []
    for key, lc in constraints():
        if not getattr(lc, "headLC", True):
            continue
        name = getattr(lc, "name", None) or str(key)
        if not getattr(lc, "active", True):
            inactive.append(name)
            continue
        try:
            domains, variables, used, phases = set(), set(), set(), set()
            aliases = {}
            collect_relations(lc, aliases)
            expression = parse(lc, domains, variables, used, phases, aliases)
            if len(domains) != 1:
                raise ValueError("cross-domain formulas are unsupported")
            expression = _bind_count_scopes(expression)
            deferred = phases - {"immediate"}
            if len(deferred) > 1:
                raise ValueError("mixed deferred evaluation phases are unsupported")
            rules.append(CompiledPredicateConstraint(name, next(iter(domains)), expression,
                                                    frozenset(used), next(iter(deferred), "immediate"),
                                                    tuple(sorted(variables)), tuple(sorted(_free_variables(expression)))))
        except ValueError as error:
            raise PredicateConstraintCompilationError(f"constraint {name!r}: {error}") from error
    return PredicateConstraintProgram(tuple(rules), tuple(inactive), max_groundings)


@dataclass(frozen=True)
class SemanticTransition:
    state: Hashable
    events: tuple[PredicateEvent, ...] = ()


class PredicateMonitorAdapter(Protocol):
    """Request-bound pure semantic state machine supplied by an application.

States must be immutable/hashable. step must emit valuations for every entity
whose rules become decidable at that point. A deferred entity must be evaluated
when it is finalized. Returning None rejects a transition. acceptance_events
provides the final valuations needed to validate an accepting state.
Finite-grounded events must include complete variable domains and observations
for that snapshot. The adapter owns batch completion and deferred phases; the
program does not infer an entity universe or accumulate prior event values.
"""

    start_state: Hashable

    def step(self, state: Hashable, symbol: Symbol) -> SemanticTransition | None: ...

    def is_accepting(self, state: Hashable) -> bool: ...

    def acceptance_events(self, state: Hashable) -> Iterable[PredicateEvent]: ...


@dataclass(frozen=True)
class PredicateMonitorState:
    grammar: Hashable
    semantic: Hashable


class PredicateMonitorDFA:
    """Lazy intersection of any DFA-like grammar and an application adapter.

With a finite token budget, filtering checks that a semantic completion remains
reachable. Without a budget it checks immediate transitions only. Symbols keep
their original hashable type.
"""

    def __init__(self, base_dfa, program: PredicateConstraintProgram, adapter: PredicateMonitorAdapter,
                 *, max_state_expansions=10000, completion_bound=None):
        self.base_dfa = base_dfa
        self.program = program
        self.rules = program.rules
        self.adapter = adapter
        self.alphabet = base_dfa.alphabet
        self.start_state = PredicateMonitorState(base_dfa.start_state, adapter.start_state)
        self._reachability = BoundedReachability(
            self.step, self.is_accepting, lambda state: base_dfa.allowed_tokens(state.grammar),
            max_state_expansions=max_state_expansions,
            completion_bound=completion_bound,
        )

    def step(self, state, symbol):
        if state is None:
            return None
        grammar = self.base_dfa.step(state.grammar, symbol)
        if grammar is None:
            return None
        transition = self.adapter.step(state.semantic, symbol)
        if transition is None or not self.program.check_events(transition.events):
            return None
        return PredicateMonitorState(grammar, transition.state)

    def allowed_tokens(self, state, remaining_steps=None):
        return self._reachability.allowed_tokens(state, remaining_steps)

    def can_reach_accepting(self, state, max_steps=None):
        return self._reachability.can_reach_accepting(state, max_steps)

    def is_accepting(self, state):
        return (state is not None and self.base_dfa.is_accepting(state.grammar)
                and self.adapter.is_accepting(state.semantic)
                and self.program.check_events(self.adapter.acceptance_events(state.semantic)))

    def accepts(self, sequence):
        state = self.start_state
        for symbol in sequence:
            state = self.step(state, symbol)
            if state is None:
                return False
        return self.is_accepting(state)


@dataclass(frozen=True)
class PredicateConstrainedDFA(DFA):
    """Immutable grammar and compiled program, awaiting an application adapter."""

    constraint_program: PredicateConstraintProgram = PredicateConstraintProgram()

    @classmethod
    def from_dfa(cls, dfa, program, **kwargs):
        return cls(states=dfa.states, alphabet=dfa.alphabet, transitions=dfa.transitions,
                   start_state=dfa.start_state, accepting_states=dfa.accepting_states,
                   dead_states=dfa.dead_states, constraint_program=program, **kwargs)

    def bind(self, adapter: PredicateMonitorAdapter, *, max_state_expansions=10000,
             completion_bound=None) -> PredicateMonitorDFA:
        return PredicateMonitorDFA(self, self.constraint_program, adapter,
                                   max_state_expansions=max_state_expansions,
                                   completion_bound=completion_bound)


def find_predicate_monitor(dfa) -> PredicateMonitorDFA | None:
    """Find a bound monitor beneath transparent DFA wrappers or runtime overlays."""
    seen = set()
    while dfa is not None and id(dfa) not in seen:
        seen.add(id(dfa))
        if isinstance(dfa, PredicateMonitorDFA):
            return dfa
        dfa = getattr(dfa, "base_dfa", None)
    return None


def require_bound_predicate_dfa(dfa):
    """Reject context-free decoding of an unbound compiled semantic program.

Return the original DFA so application overlays stay intact. Ordinary grammar
DFAs and programs with no semantic rules remain valid without an adapter.
"""
    if find_predicate_monitor(dfa) is not None:
        return dfa
    current, seen = dfa, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, PredicateConstrainedDFA) and current.constraint_program.rules:
            raise MissingSemanticBinding("decoding requires a request-bound predicate monitor; bind the compiled DFA first")
        current = getattr(current, "base_dfa", None)
    return dfa
