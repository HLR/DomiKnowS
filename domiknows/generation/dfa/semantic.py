"""Request-bound, typed generation semantics for non-DFA logical constraints.

The graph DFA handles regular token languages.  This module handles graph
expressions that read request attributes or return numbers, selections, and
answers.  A bounded DFA-like wrapper makes Boolean results and explicitly
expected values available to the existing decoders without pretending that a
value-returning expression is itself a Boolean constraint.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Any, Callable, Iterable, Mapping

from ._lc_normalize import kind, normalize_lc
from .conditioning import bind_contextual_dfa
from .graph_discovery import (
    _COMPARATIVE_TYPES,
    _concept_tuple,
    _count_limit,
    _is_v,
    _last_int,
    _predicate_token_set,
    _token_predicate_from_expr,
    _token_from_tuple,
    _walk_lc,
    analyze_generation_constraints,
    constraints_to_dfa_from_graph,
)


_VALUE_TYPES = {"sumL": "number", "iotaL": "selection", "miotaL": "selection_set", "queryL": "answer"}
_FILTER_TYPES = {"eqL"}
_BOOLEAN_TYPES = {
    "fixedL", "sameL", "differentL", "atMostL", "atLeastL", "exactL", "existsL",
    "atMostAL", "atLeastAL", "exactAL", "existsAL", "forAllL",
    "andL", "orL", "notL", "nandL", "norL", "xorL", "ifL", "iffL", "equivalenceL",
    *_COMPARATIVE_TYPES,
}


class MissingSemanticBinding(ValueError):
    """The request has not supplied a fact needed to evaluate an LC."""


class UnsupportedSemanticShape(ValueError):
    """An LC cannot be interpreted from the declared generation snapshot."""


@dataclass(frozen=True)
class GenerationSemanticContext:
    """Request facts indexed by generated position.

    ``attributes[i]`` contains attributes of position ``i``; ``memberships``
    maps concept names to matching positions.  ``bindings`` maps LC variable
    names to positions. ``fixed_truth`` is the legacy shared truth map;
    ``fixed_truth_by_lc`` holds separately grounded observations for each
    ``fixedL``. ``membership_scores`` maps an ``miotaL`` name to its
    per-position scores. ``expected`` supplies targets for value-returning
    head LCs. ``position_count`` fixes the length of a DataNode snapshot.
    """

    attributes: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)
    memberships: Mapping[str, frozenset[int]] = field(default_factory=dict)
    bindings: Mapping[str, int] = field(default_factory=dict)
    fixed_truth: Mapping[int, bool] | None = None
    fixed_truth_by_lc: Mapping[str, Mapping[int, bool]] = field(default_factory=dict)
    membership_scores: Mapping[str, Mapping[int, float]] = field(default_factory=dict)
    expected: Mapping[str, Any] = field(default_factory=dict)
    request_values: Mapping[str, Any] = field(default_factory=dict)
    path_resolver: Callable[[Any, tuple[int, ...]], Iterable[int]] | None = None
    position_count: int | None = None

    @classmethod
    def from_data_nodes(cls, nodes, **kwargs):
        """Take a stable attribute snapshot from token-position DataNodes."""
        attributes = {}
        memberships = {}
        for index, node in enumerate(nodes):
            values = dict(node.getAttributes())
            values.setdefault("instanceID", node.instanceID)
            values.setdefault("instanceValue", node.instanceValue)
            attributes[index] = values
            concept_name = getattr(getattr(node, "ontologyNode", None), "name", None)
            if concept_name is not None:
                memberships.setdefault(concept_name, set()).add(index)
        for name, indexes in kwargs.pop("memberships", {}).items():
            memberships.setdefault(name, set()).update(indexes)
        return cls(attributes=attributes, memberships={
            name: frozenset(indexes) for name, indexes in memberships.items()
        }, position_count=len(attributes), **kwargs)


@dataclass(frozen=True)
class SemanticValue:
    kind: str
    value: Any
    valid: bool = True


def _lc_name(lc):
    return getattr(lc, "name", None) or getattr(lc, "lcName", None) or kind(lc)


def _scalar(value, source):
    """Read one DataNode label or score without silently reducing a vector."""
    if hasattr(value, "numel") and value.numel() != 1:
        raise MissingSemanticBinding(f"{source!r} must be a scalar")
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise MissingSemanticBinding(f"{source!r} must be a scalar")
        return _scalar(value[0], source)
    return value


def _observed_label(value, vocabulary, source):
    value = _scalar(value, source)
    if isinstance(value, bool):
        raise MissingSemanticBinding(f"{source!r} must be a compact label, not a Boolean")
    if isinstance(value, str):
        try:
            return vocabulary.label_for_token(value)
        except KeyError:
            pass
    try:
        label = int(value)
    except (TypeError, ValueError) as exc:
        raise MissingSemanticBinding(f"{source!r} is not a compact label") from exc
    if label not in vocabulary.alphabet or label != value and not isinstance(value, str):
        raise MissingSemanticBinding(f"{source!r} is not a valid compact label")
    return label


def _score(value, source):
    try:
        score = float(_scalar(value, source))
    except (TypeError, ValueError) as exc:
        raise MissingSemanticBinding(f"{source!r} must be a probability") from exc
    if not isfinite(score) or not 0 <= score <= 1:
        raise MissingSemanticBinding(f"{source!r} must be a probability in [0, 1]")
    return score


class _Evaluator:
    def __init__(self, bundle, labels, context: GenerationSemanticContext):
        self.bundle = bundle
        self.labels = tuple(int(label) for label in labels)
        self.context = context
        self.domain = frozenset(range(len(self.labels)))

    def _atom(self, atom):
        concept, _, label, _ = atom
        if concept is self.bundle.generated_token:
            if label is None:
                return self.domain
            return frozenset(i for i, value in enumerate(self.labels) if value == int(label))
        concept_name = getattr(concept, "name", None)
        if concept_name not in self.context.memberships:
            raise MissingSemanticBinding(f"missing membership set for concept {concept_name!r}")
        return frozenset(self.context.memberships[concept_name]) & self.domain

    def positions(self, lc):
        op = kind(lc)
        if op == "eqL":
            concept, attribute, required = lc.e
            atom = _concept_tuple(concept)
            base = self._atom(atom) if atom is not None else self._atom((concept, None, None, 1))
            selected = set()
            for index in base:
                attrs = self.context.attributes.get(index)
                if attrs is None or attribute not in attrs:
                    raise MissingSemanticBinding(f"missing attribute {attribute!r} at position {index}")
                if attrs[attribute] in required:
                    selected.add(index)
            return frozenset(selected)
        if op == "notL":
            segments = self._segments(lc.e)
            if len(segments) != 1:
                raise UnsupportedSemanticShape("notL requires one unary predicate")
            return self.domain - self.positions(segments[0])
        if op in {"andL", "orL"}:
            sets = [self.positions(item) for item in self._segments(lc.e)]
            if op == "andL":
                return frozenset.intersection(self.domain, *sets)
            return frozenset().union(*sets)
        sets = []
        elements = tuple(item for item in lc.e if not isinstance(item, int))
        index = 0
        while index < len(elements):
            item = elements[index]
            if hasattr(item, "e"):
                sets.append(self.positions(item))
                index += 1
                continue
            atom = _concept_tuple(item)
            if atom is None or index + 1 >= len(elements) or not _is_v(elements[index + 1]):
                raise UnsupportedSemanticShape(f"{op} has a non-unary or unbound predicate")
            variable = elements[index + 1]
            matching = self._atom(atom)
            if variable.v is not None:
                if self.context.path_resolver is None:
                    raise MissingSemanticBinding(f"{op} requires path_resolver for {variable.v!r}")
                matching &= frozenset(self.context.path_resolver(variable.v, self.labels))
            sets.append(matching)
            index += 2
        if not sets:
            raise UnsupportedSemanticShape(f"{op} has no position predicate")
        return frozenset.intersection(self.domain, *sets)

    def _numeric(self, lc):
        value = self.evaluate(lc)
        if value.kind == "number" and value.valid:
            return value.value
        raise UnsupportedSemanticShape(f"{kind(lc)} does not provide a valid number")

    @staticmethod
    def _segments(elements):
        """Split flattened LC operands into child expressions or tuple/V pairs."""
        operands = [item for item in elements if not isinstance(item, int)]
        groups = []
        index = 0
        while index < len(operands):
            item = operands[index]
            if hasattr(item, "e"):
                groups.append(item)
                index += 1
            elif (_concept_tuple(item) is not None and index + 1 < len(operands)
                  and _is_v(operands[index + 1])):
                groups.append(_FlatExpr(operands[index:index + 2]))
                index += 2
            else:
                raise UnsupportedSemanticShape("LC operands are not unary predicates or value expressions")
        return groups

    def _category(self, concept, position):
        if concept is self.bundle.generated_token:
            return self.bundle.vocabulary.token_for_label(self.labels[position])
        concept_name = getattr(concept, "name", None)
        attrs = self.context.attributes.get(position)
        if attrs is None:
            raise MissingSemanticBinding(f"missing {concept_name!r} value at position {position}")
        if concept_name in attrs:
            return attrs[concept_name]
        label_key = f"<{concept_name}>/label"
        if label_key in attrs:
            value = _scalar(attrs[label_key], label_key)
            enum = getattr(concept, "enum", None)
            if enum is not None:
                try:
                    return enum[int(value)]
                except (IndexError, ValueError, TypeError) as exc:
                    raise MissingSemanticBinding(
                        f"invalid {label_key!r} at position {position}"
                    ) from exc
            return value
        raise MissingSemanticBinding(f"missing {concept_name!r} value at position {position}")

    def _value(self, lc):
        op = kind(lc)
        if op == "sumL":
            segments = self._segments(lc.e)
            if not segments:
                raise UnsupportedSemanticShape("sumL has no arguments")
            count = sum(len(self.positions(segment)) for segment in segments)
            return SemanticValue("number", count)
        if op == "iotaL":
            selected = self.positions(lc)
            return SemanticValue("selection", next(iter(selected)) if len(selected) == 1 else None,
                                 valid=len(selected) == 1)
        if op == "miotaL":
            scores = self.context.membership_scores.get(_lc_name(lc))
            if scores is None:
                if not lc.hard:
                    raise MissingSemanticBinding(f"missing membership scores for {_lc_name(lc)!r}")
                candidates = self.positions(lc)
                scores = {i: float(i in candidates) for i in self.domain}
            missing = self.domain - scores.keys()
            if missing:
                raise MissingSemanticBinding(f"missing membership scores at positions {sorted(missing)}")
            selected = frozenset(i for i in self.domain if scores[i] >= lc.threshold)
            return SemanticValue("selection_set", selected)
        if op == "queryL":
            selectors = [item for item in lc.e if hasattr(item, "e") and kind(item) in {"iotaL", "miotaL"}]
            if len(selectors) != 1:
                raise UnsupportedSemanticShape("queryL requires exactly one direct iotaL or miotaL selector")
            selected = self.evaluate(selectors[0])
            if not selected.valid:
                return SemanticValue("answer", None, valid=False)
            if selected.kind == "selection":
                return SemanticValue("answer", self._category(lc.concept, selected.value))
            values = tuple(
                self._category(lc.concept, index) if index in selected.value else None
                for index in self.domain
            )
            return SemanticValue("answer", values)
        raise UnsupportedSemanticShape(f"{op} is not a value-returning generation LC")

    def _boolean(self, lc):
        op = kind(lc)
        children = [item for item in lc.e if hasattr(item, "e")]
        if op in {"andL", "orL", "nandL", "norL", "xorL", "iffL", "equivalenceL", "ifL", "notL"}:
            if op == "notL" and len(children) == 1:
                return not self._numeric_or_boolean(children[0])
            if op in {"ifL", "xorL", "iffL", "equivalenceL"} and len(children) != 2:
                raise UnsupportedSemanticShape(f"{op} requires two Boolean children")
            if not children:
                raise UnsupportedSemanticShape(f"{op} requires Boolean children")
            values = [self._numeric_or_boolean(child) for child in children]
            if op in {"andL", "nandL"}:
                result = all(values)
                return not result if op == "nandL" else result
            if op in {"orL", "norL"}:
                result = any(values)
                return not result if op == "norL" else result
            if op == "ifL":
                return not values[0] or values[1]
            if op == "xorL":
                return values[0] != values[1]
            return values[0] == values[1]
        if op == "fixedL":
            observed = self.context.fixed_truth_by_lc.get(_lc_name(lc), self.context.fixed_truth)
            if observed is None:
                raise MissingSemanticBinding("fixedL requires fixed_truth for observed positions")
            if any(index not in self.domain for index in observed):
                return False
            selected = self.positions(lc)
            return all((index in selected) == bool(truth)
                       for index, truth in observed.items() if index in self.domain)
        if op in {"sameL", "differentL"}:
            variables = [item.name for item in lc.e if _is_v(item)]
            if len(variables) < 2:
                raise UnsupportedSemanticShape(f"{op} requires at least two bound variables")
            values = []
            for variable in variables:
                if variable not in self.context.bindings:
                    raise MissingSemanticBinding(f"missing position binding for {variable!r}")
                position = self.context.bindings[variable]
                if position not in self.domain:
                    return False
                values.append(self._category(lc.concept, position))
            same = len(set(values)) == 1
            return same if op == "sameL" else not same
        if op in {"atMostL", "atLeastL", "exactL", "existsL", "atMostAL", "atLeastAL", "exactAL", "existsAL"}:
            child_values = [item for item in lc.e if hasattr(item, "e") and kind(item) == "sumL"]
            count = self._numeric(child_values[0]) if len(child_values) == 1 else len(self.positions(lc))
            limit = _count_limit(lc)
            if op.startswith("atMost"):
                return count <= limit
            if op.startswith("atLeast") or op.startswith("exists"):
                return count >= limit
            return count == limit
        if op in _COMPARATIVE_TYPES:
            segments = self._segments(lc.e)
            if len(segments) != 2:
                raise UnsupportedSemanticShape(f"{op} requires two count operands")
            left, right = (
                self._numeric(segment) if kind(segment) == "sumL"
                else len(self.positions(segment))
                for segment in segments
            )
            difference = left - right
            offset = _last_int(lc.e) or 0
            return {
                ">": difference > offset, ">=": difference >= offset,
                "<": difference < offset, "<=": difference <= offset,
                "==": difference == offset, "!=": difference != offset,
            }[_COMPARATIVE_TYPES[op]]
        if op == "forAllL":
            segments = self._segments(lc.e)
            if len(segments) != 2:
                raise UnsupportedSemanticShape("forAllL requires two unary predicates")
            return self.positions(segments[0]) <= self.positions(segments[1])
        raise UnsupportedSemanticShape(f"{op} has no request-bound Boolean interpretation")

    def _numeric_or_boolean(self, lc):
        result = self.evaluate(lc)
        if result.kind == "selection":
            return result.valid
        if result.kind != "boolean":
            raise UnsupportedSemanticShape(f"{kind(lc)} returns {result.kind}, not a Boolean")
        return result.valid and result.value

    def evaluate(self, lc):
        op = kind(lc)
        if op in _VALUE_TYPES:
            return self._value(lc)
        if op in _BOOLEAN_TYPES:
            return SemanticValue("boolean", self._boolean(lc))
        if op in _FILTER_TYPES:
            return SemanticValue("filter", self.positions(lc))
        raise UnsupportedSemanticShape(f"unsupported generation LC type {op}")


class _FlatExpr:
    def __init__(self, elements):
        self.e = tuple(elements)
        self._kind = "flat_predicate"


@dataclass(frozen=True)
class GenerationConstraintPlan:
    graph: Any
    bundle: Any
    base_dfa: Any
    deferred: tuple[tuple[str, Any], ...]

    def evaluate(self, labels, context: GenerationSemanticContext):
        evaluator = _Evaluator(self.bundle, labels, context)
        return {name: evaluator.evaluate(lc) for name, lc in self.deferred}

    def context_from_data_nodes(self, nodes, *, expected=None, request_values=None,
                                path_resolver=None):
        """Build request facts from ordered candidate DataNodes.

        ``<generated_token>/label`` supplies observed compact labels for
        ``fixedL``.  ``<selector-name>/score`` supplies soft ``miotaL`` scores;
        a unary generated-token selector may instead use
        ``<generated_token>/local/softmax``.  ``generation_variables`` on a
        node names the LC variables bound to its position.
        """
        nodes = tuple(nodes)
        context = GenerationSemanticContext.from_data_nodes(
            nodes, expected=dict(expected or {}),
            request_values=dict(request_values or {}), path_resolver=path_resolver,
        )
        attributes = context.attributes
        bindings = {}
        for index, attrs in attributes.items():
            names = attrs.get("generation_variables", ())
            names = (names,) if isinstance(names, str) else tuple(names)
            for name in names:
                if not isinstance(name, str) or not name:
                    raise MissingSemanticBinding("generation_variables must contain nonempty names")
                if name in bindings:
                    raise MissingSemanticBinding(f"variable {name!r} is bound at multiple positions")
                bindings[name] = index

        fixed_truth = {}
        scores = {}
        visited = set()
        for _, head in self.deferred:
            for lc in _walk_lc(head):
                if not hasattr(lc, "e") or id(lc) in visited:
                    continue
                visited.add(id(lc))
                op = kind(lc)
                name = _lc_name(lc)
                if op == "eqL":
                    concept, attribute, _ = lc.e
                    concept = _concept_tuple(concept)[0] if _concept_tuple(concept) else concept
                    candidates = (range(len(nodes)) if concept is self.bundle.generated_token
                                  else context.memberships.get(getattr(concept, "name", None)))
                    if candidates is None:
                        raise MissingSemanticBinding(f"missing candidate membership for eqL {name!r}")
                    missing = [i for i in candidates if attribute not in attributes[i]]
                    if missing:
                        raise MissingSemanticBinding(
                            f"eqL {name!r} needs attribute {attribute!r} at positions {missing}"
                        )
                elif op == "fixedL":
                    predicate = _token_predicate_from_expr(lc, self.bundle)
                    if predicate is None:
                        raise UnsupportedSemanticShape(
                            f"automatic fixedL binding needs a pathless generated-token predicate: {name}"
                        )
                    allowed = {
                        self.bundle.vocabulary.label_for_token(token)
                        for token in _predicate_token_set(predicate, self.bundle)
                    }
                    label_key = f"<{self.bundle.generated_token.name}>/label"
                    observed = {
                        i: _observed_label(attrs[label_key], self.bundle.vocabulary, label_key) in allowed
                        for i, attrs in attributes.items() if label_key in attrs
                    }
                    if not observed:
                        raise MissingSemanticBinding(f"fixedL {name!r} needs observed {label_key!r}")
                    fixed_truth[name] = observed
                elif op == "miotaL" and not lc.hard:
                    score_key = f"{name}/score"
                    predicate = _token_predicate_from_expr(lc, self.bundle)
                    token_set = (_predicate_token_set(predicate, self.bundle)
                                 if predicate is not None else set())
                    token_label = (self.bundle.vocabulary.label_for_token(next(iter(token_set)))
                                   if len(token_set) == 1 else None)
                    probability_key = f"<{self.bundle.generated_token.name}>/local/softmax"
                    per_position = {}
                    for index, attrs in attributes.items():
                        if score_key in attrs:
                            value = attrs[score_key]
                        elif token_label is not None and probability_key in attrs:
                            distribution = attrs[probability_key]
                            if hasattr(distribution, "tolist"):
                                distribution = distribution.tolist()
                            if len(distribution) != self.bundle.vocabulary.label_count:
                                raise MissingSemanticBinding(
                                    f"{probability_key!r} has wrong size at position {index}"
                                )
                            value = distribution[token_label]
                        else:
                            raise MissingSemanticBinding(
                                f"miotaL {name!r} needs {score_key!r} at position {index}"
                            )
                        per_position[index] = _score(value, score_key)
                    scores[name] = per_position
                elif op in {"sameL", "differentL"}:
                    for variable in (item.name for item in lc.e if _is_v(item)):
                        if variable not in bindings:
                            raise MissingSemanticBinding(
                                f"{op} {name!r} needs generation_variables binding for {variable!r}"
                            )
        context = replace(context, bindings=bindings, fixed_truth_by_lc=fixed_truth,
                          membership_scores=scores)
        validator = _Evaluator(self.bundle, (0,) * len(nodes), context)
        for _, head in self.deferred:
            for lc in _walk_lc(head):
                if kind(lc) in {"sameL", "differentL"}:
                    for variable in (item.name for item in lc.e if _is_v(item)):
                        validator._category(lc.concept, bindings[variable])
        return context

    def bind_data_nodes(self, nodes, *, max_sequence_length: int, expected=None,
                        request_values=None, path_resolver=None, search_budget=100_000):
        """Build request facts and bind the resulting bounded decoder."""
        context = self.context_from_data_nodes(
            nodes, expected=expected, request_values=request_values,
            path_resolver=path_resolver,
        )
        return self.bind(context, max_sequence_length=max_sequence_length,
                         search_budget=search_budget)

    def bind(self, context: GenerationSemanticContext, *, max_sequence_length: int,
             search_budget: int = 100_000):
        if max_sequence_length < 0:
            raise ValueError("max_sequence_length must be non-negative")
        if not isinstance(context, GenerationSemanticContext):
            raise TypeError("context must be a GenerationSemanticContext")
        if context.position_count is not None and context.position_count > max_sequence_length:
            raise ValueError("max_sequence_length is shorter than the candidate DataNode snapshot")
        unknown = set(context.expected) - {name for name, _ in self.deferred}
        if unknown:
            raise MissingSemanticBinding(f"expected values name unknown deferred LCs: {sorted(unknown)}")
        base_dfa = bind_contextual_dfa(self.base_dfa, self.graph, context.request_values)
        return BoundSemanticDFA(self, base_dfa, context, max_sequence_length, search_budget)


def compile_generation_constraint_plan(graph, bundle, *, max_sequence_length=None):
    """Compile regular LCs and retain typed LCs for request-time binding."""
    analyses = analyze_generation_constraints(
        graph, bundle, on_unsupported="ignore", max_sequence_length=max_sequence_length,
    )
    base_dfa = constraints_to_dfa_from_graph(
        graph, bundle, on_unsupported="ignore", max_sequence_length=max_sequence_length,
    )
    deferred = []
    for analysis in analyses:
        lc = graph.logicalConstrains[analysis.lc_name]
        if analysis.supported and not normalize_lc(lc, bundle=bundle).irregular_children:
            continue
        if getattr(lc, "_generation_latent_specs", ()) and not hasattr(lc, "_generation_dfa_constraint"):
            continue
        deferred.append((analysis.lc_name, lc))
    return GenerationConstraintPlan(graph, bundle, base_dfa, tuple(deferred))


class BoundSemanticDFA:
    """Lazy bounded DFA-like product used by the existing decoders."""

    def __init__(self, plan, base_dfa, context, max_sequence_length, search_budget):
        self.plan = plan
        self.base_dfa = base_dfa
        self.context = context
        self.max_sequence_length = max_sequence_length
        self.search_budget = search_budget
        self.alphabet = base_dfa.alphabet
        self.start_state = (base_dfa.start_state, ())
        self._reach_cache = {}

    def step(self, state, symbol):
        base_state, prefix = state
        if len(prefix) >= self.max_sequence_length:
            return None
        next_base = self.base_dfa.step(base_state, symbol)
        if next_base is None:
            return None
        return next_base, prefix + (int(symbol),)

    def values(self, state):
        return self.plan.evaluate(state[1], self.context)

    def is_accepting(self, state):
        base_state, prefix = state
        if self.context.position_count is not None and len(prefix) != self.context.position_count:
            return False
        if not self.base_dfa.is_accepting(base_state):
            return False
        for name, result in self.values(state).items():
            if not result.valid:
                return False
            if result.kind == "boolean" and not result.value:
                return False
            if name in self.context.expected and result.value != self.context.expected[name]:
                return False
        return True

    def accepts(self, sequence):
        state = self.start_state
        for symbol in sequence:
            state = self.step(state, symbol)
            if state is None:
                return False
        return self.is_accepting(state)

    def can_reach_accepting(self, state, max_steps=None):
        remaining = self.max_sequence_length - len(state[1])
        if max_steps is not None:
            remaining = min(remaining, int(max_steps))
        if remaining < 0:
            return False
        key = (state, remaining)
        if key in self._reach_cache:
            return self._reach_cache[key]
        queue = deque([(state, 0)])
        seen = {state}
        work = 0
        while queue:
            current, depth = queue.popleft()
            if self.is_accepting(current):
                self._reach_cache[key] = True
                return True
            if depth >= remaining:
                continue
            for symbol in self.base_dfa.allowed_tokens(current[0]):
                work += 1
                if work > self.search_budget:
                    raise RuntimeError("semantic reachability exceeded search_budget")
                nxt = self.step(current, symbol)
                if nxt is not None and nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, depth + 1))
        self._reach_cache[key] = False
        return False

    def allowed_tokens(self, state, remaining_steps=None):
        if remaining_steps is not None and remaining_steps <= 0:
            return set()
        return {
            symbol for symbol in self.base_dfa.allowed_tokens(state[0], remaining_steps=remaining_steps)
            if (nxt := self.step(state, symbol)) is not None
            and self.can_reach_accepting(
                nxt, None if remaining_steps is None else remaining_steps - 1,
            )
        }
