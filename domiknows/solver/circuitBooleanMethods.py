"""Exact logical-constraint processor backed by decision diagrams.

Every operator builds a symbolic circuit.  Probability tensors are attached
only to grounded leaves and are consumed later by differentiable weighted
model counting, so repeated occurrences of a leaf retain logical identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

import torch

from .bdd import BDDManager, BDDNode
from .constraintsProcessorInterface import constraintsProcessor
from .ilpConfig import ilpConfig


@dataclass(frozen=True, slots=True)
class CircuitLeaf:
    """Stable grounded leaf handle produced by the constraint constructor."""

    key: tuple[str, object, int]
    probability: torch.Tensor
    variable_key: Hashable
    value_index: int
    probabilities: tuple[torch.Tensor, ...]
    categorical: bool = False
    fixed_value: int | None = None


class PySDDCircuitManager:
    """Optional pysdd-backed manager with torch-native WMC evaluation.

    Categorical concepts are encoded with an exactly-one SDD and literal
    weights ``w(class)=p(class)``, ``w(not class)=1``.  Thus a categorical
    assignment has exactly its softmax probability rather than an independent
    Bernoulli product.
    """

    backend_name = "pysdd"

    def __init__(self, max_nodes=100_000, size_limit_action="raise"):
        from pysdd.sdd import SddManager

        self.max_nodes = max_nodes
        self.size_limit_action = size_limit_action
        self._manager = SddManager(var_count=1, auto_gc_and_minimize=False)
        self.false = self._manager.false()
        self.true = self._manager.true()
        self._next_literal = 1
        self._literal_by_key = {}
        self._weights = {}
        self._categorical_groups = {}
        self._active_categorical_groups = set()
        self._binary_literals = set()
        self._registered_leaf_keys = set()
        self._warned_size = False

    def begin_evaluation(self):
        self._weights.clear()
        self._registered_leaf_keys.clear()
        self._active_categorical_groups.clear()

    @property
    def variable_keys(self):
        return tuple(self._literal_by_key)

    @property
    def node_count(self):
        # pysdd exposes size() on nodes and live_size() on managers depending
        # on version.  This count is used only for diagnostics/budget checks.
        live_size = getattr(self._manager, "live_size", None)
        return int(live_size()) if callable(live_size) else len(self._literal_by_key) + 2

    def _literal(self, key):
        if key not in self._literal_by_key:
            literal = self._next_literal
            if literal > 1:
                self._manager.add_var_after_last()
            self._next_literal += 1
            self._literal_by_key[key] = literal
        return self._literal_by_key[key]

    def _check_size(self, node):
        if self.max_nodes is None or int(node.size()) <= self.max_nodes:
            return node
        from .bdd import CircuitSizeLimitExceeded

        message = (
            f"Exact semantic-loss circuit exceeded max_nodes={self.max_nodes} "
            f"({int(node.size())} SDD nodes compiled)."
        )
        if self.size_limit_action == "raise":
            raise CircuitSizeLimitExceeded(message)
        if not self._warned_size:
            import warnings

            warnings.warn(message, RuntimeWarning, stacklevel=3)
            self._warned_size = True
        return node

    @staticmethod
    def _scalar(value):
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError("A circuit branch probability must be scalar")
            return value.reshape(())
        return torch.as_tensor(value, dtype=torch.get_default_dtype()).reshape(())

    def literal(self, variable_key, value_index, probabilities):
        probabilities = tuple(self._scalar(p) for p in probabilities)
        if len(probabilities) == 2 and value_index == 1 and variable_key[0] == "binary":
            leaf_key = variable_key[1]
            literal = self._literal(leaf_key)
            self._binary_literals.add(literal)
            self._weights[literal] = probabilities[1]
            self._registered_leaf_keys.add(leaf_key)
            return self._check_size(self._manager.literal(literal))

        group_literals = []
        for index, probability in enumerate(probabilities):
            class_key = (variable_key, index)
            literal = self._literal(class_key)
            group_literals.append(literal)
            self._weights[literal] = probability
            self._registered_leaf_keys.add(class_key)
        self._categorical_groups[variable_key] = tuple(group_literals)
        self._active_categorical_groups.add(variable_key)
        return self._check_size(self._manager.literal(group_literals[value_index]))

    def negate(self, node):
        return self._check_size(~node)

    def conjunction(self, left, right):
        return self._check_size(left & right)

    def disjunction(self, left, right):
        return self._check_size(left | right)

    def and_all(self, nodes):
        result = self.true
        for node in nodes:
            result = self.conjunction(result, node)
        return result

    def or_all(self, nodes):
        result = self.false
        for node in nodes:
            result = self.disjunction(result, node)
        return result

    def _with_categorical_axioms(self, root):
        for group_key in self._active_categorical_groups:
            literals = self._categorical_groups[group_key]
            at_least_one = self.or_all(self._manager.literal(lit) for lit in literals)
            at_most_one = self.true
            for i, left in enumerate(literals):
                for right in literals[i + 1 :]:
                    at_most_one &= ~(self._manager.literal(left) & self._manager.literal(right))
            root &= at_least_one & at_most_one
        return self._check_size(root)

    def wmc(self, root):
        root = self._with_categorical_axioms(root)
        reference = next(iter(self._weights.values()), torch.tensor(1.0))
        zero, one = reference.new_zeros(()), reference.new_ones(())
        memo = {}
        weights = dict(self._weights)
        for group_key in self._active_categorical_groups:
            literals = self._categorical_groups[group_key]
            total = sum((weights[literal] for literal in literals), zero)
            if not torch.allclose(total.detach(), one.detach(), rtol=1e-5, atol=1e-7):
                safe_total = total.clamp_min(torch.finfo(total.dtype).tiny)
                for literal in literals:
                    weights[literal] = weights[literal] / safe_total

        def evaluate(node):
            node_id = int(node.id)
            if node_id in memo:
                return memo[node_id]
            if node.is_false():
                return zero
            if node.is_true():
                return one
            if node.is_literal():
                literal = int(node.literal)
                positive = abs(literal)
                probability = weights[positive]
                if literal > 0:
                    value = probability
                elif positive in self._binary_literals:
                    value = one - probability
                else:
                    value = one
            else:
                value = sum(
                    (evaluate(prime) * evaluate(sub) for prime, sub in node.elements()),
                    zero,
                )
            memo[node_id] = value
            return value

        return evaluate(root)


def make_circuit_manager(backend="auto", max_nodes=100_000, size_limit_action="raise"):
    """Create the requested manager, preferring pysdd when it is importable."""

    if backend not in {"auto", "bdd", "pysdd"}:
        raise ValueError("backend must be 'auto', 'bdd', or 'pysdd'")
    if backend in {"auto", "pysdd"}:
        try:
            import pysdd  # noqa: F401

            return PySDDCircuitManager(max_nodes=max_nodes, size_limit_action=size_limit_action)
        except (ImportError, OSError):
            if backend == "pysdd":
                raise
    return BDDManager(max_nodes=max_nodes, size_limit_action=size_limit_action)


class circuitBooleanMethods(constraintsProcessor):
    """Constraint processor that returns exact circuit nodes."""

    grad = True
    is_exact_circuit = True

    def __init__(
        self,
        _ildConfig=ilpConfig,
        *,
        backend="auto",
        max_nodes=100_000,
        size_limit_action="raise",
    ):
        self.config = _ildConfig
        self.requested_backend = backend
        self.max_nodes = max_nodes
        self.size_limit_action = size_limit_action
        self.manager = make_circuit_manager(
            backend=backend,
            max_nodes=max_nodes,
            size_limit_action=size_limit_action,
        )
        self.current_device = None
        self.current_dtype = None
        self._leaf_keys = set()

    @property
    def backend_name(self):
        return self.manager.backend_name

    @property
    def node_count(self):
        return self.manager.node_count

    def begin_evaluation(self):
        self.manager.begin_evaluation()
        self._leaf_keys.clear()

    def _node(self, value):
        if isinstance(value, CircuitLeaf):
            if value.fixed_value is not None:
                return self.manager.true if value.fixed_value else self.manager.false
            self._leaf_keys.add(value.key)
            return self.manager.literal(
                value.variable_key,
                value.value_index,
                value.probabilities,
            )
        if isinstance(value, BDDNode):
            return value
        # pysdd nodes intentionally have no common public base class.
        if hasattr(value, "is_literal") and hasattr(value, "elements"):
            return value
        if value is None:
            return self.manager.true
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise TypeError("A raw tensor cannot be used as a circuit node")
            value = bool(value.detach().item())
        if isinstance(value, (bool, int, float)):
            return self.manager.true if bool(value) else self.manager.false
        raise TypeError(f"Unsupported circuit operand {type(value).__name__}")

    def _nodes(self, values):
        return [self._node(value) for value in values if value is not None]

    def notVar(self, _, var, onlyConstrains=False):
        return self.manager.negate(self._node(var))

    def andVar(self, _, *var, onlyConstrains=False):
        return self.manager.and_all(self._nodes(var))

    def orVar(self, _, *var, onlyConstrains=False):
        return self.manager.or_all(self._nodes(var))

    def nandVar(self, _, *var, onlyConstrains=False):
        return self.notVar(None, self.andVar(None, *var))

    def norVar(self, _, *var, onlyConstrains=False):
        return self.notVar(None, self.orVar(None, *var))

    def ifVar(self, _, var1, var2, onlyConstrains=False):
        return self.orVar(None, self.notVar(None, var1), var2)

    def xorVar(self, _, *var, onlyConstrains=False):
        # DomiKnowS' n-ary xorL means exactly one operand is true.
        return self.countVar(None, *var, limitOp="==", limit=1)

    def equivalenceVar(self, _, *var, onlyConstrains=False):
        nodes = self._nodes(var)
        if len(nodes) <= 1:
            return self.manager.true
        all_true = self.manager.and_all(nodes)
        all_false = self.manager.and_all(self.manager.negate(node) for node in nodes)
        return self.manager.disjunction(all_true, all_false)

    def _count_states(self, nodes: Sequence):
        states = [self.manager.true]
        for node in nodes:
            negated = self.manager.negate(node)
            next_states = [self.manager.false] * (len(states) + 1)
            for count, state in enumerate(states):
                next_states[count] = self.manager.disjunction(
                    next_states[count], self.manager.conjunction(state, negated)
                )
                next_states[count + 1] = self.manager.disjunction(
                    next_states[count + 1], self.manager.conjunction(state, node)
                )
            states = next_states
        return states

    @staticmethod
    def _comparison(left, right, operation):
        if operation == ">":
            return left > right
        if operation == ">=":
            return left >= right
        if operation == "<":
            return left < right
        if operation == "<=":
            return left <= right
        if operation == "==":
            return left == right
        if operation == "!=":
            return left != right
        raise ValueError(f"Unsupported count comparison {operation!r}")

    def countVar(
        self,
        _,
        *var,
        onlyConstrains=False,
        limitOp="None",
        limit=1,
        logicMethodName="COUNT",
    ):
        states = self._count_states(self._nodes(var))
        selected = [
            state for count, state in enumerate(states)
            if self._comparison(count, int(limit), limitOp)
        ]
        return self.manager.or_all(selected)

    def compareCountsVar(
        self,
        _,
        varsA,
        varsB,
        *,
        compareOp=">",
        diff=0,
        onlyConstrains=False,
        logicMethodName="COUNT_CMP",
    ):
        states_a = self._count_states(self._nodes(varsA))
        states_b = self._count_states(self._nodes(varsB))
        matches = []
        for count_a, state_a in enumerate(states_a):
            for count_b, state_b in enumerate(states_b):
                if self._comparison(count_a - count_b, int(diff), compareOp):
                    matches.append(self.manager.conjunction(state_a, state_b))
        return self.manager.or_all(matches)

    def fixedVar(self, _, var, onlyConstrains=False):
        return self._node(var)

    def summationVar(
        self,
        _,
        *var,
        onlyConstrains=False,
        label=None,
        logicMethodName="SUMMATION",
    ):
        if label is None:
            raise ValueError("Exact sumL requires its integer label")
        if torch.is_tensor(label):
            label = int(label.detach().reshape(-1)[0].item())
        return self.countVar(None, *var, limitOp="==", limit=int(label))

    def iotaVar(
        self,
        _,
        *var,
        onlyConstrains=False,
        temperature=1.0,
        logicMethodName="IOTA",
    ):
        nodes = self._nodes(var)
        selections = []
        for index, node in enumerate(nodes):
            selections.append(
                self.manager.and_all(
                    [node]
                    + [
                        self.manager.negate(other)
                        for other_index, other in enumerate(nodes)
                        if other_index != index
                    ]
                )
            )
        if onlyConstrains:
            return self.manager.or_all(selections)
        return selections

    def queryVar(
        self,
        _,
        concept,
        subclasses,
        selection_vars,
        *,
        subclass_data=None,
        onlyConstrains=False,
        temperature=1.0,
        logicMethodName="QUERY",
    ):
        selections = self._nodes(selection_vars)
        if subclass_data is None:
            raise ValueError("Exact queryL requires per-entity subclass data")
        class_nodes = []
        for class_index in range(len(subclasses)):
            alternatives = []
            for entity_index, selection in enumerate(selections):
                if entity_index >= len(subclass_data):
                    continue
                entity_values = subclass_data[entity_index]
                if class_index >= len(entity_values):
                    continue
                alternatives.append(
                    self.manager.conjunction(
                        selection, self._node(entity_values[class_index])
                    )
                )
            class_nodes.append(self.manager.or_all(alternatives))
        return class_nodes

    def sameVar(
        self,
        _,
        concept,
        subclasses,
        *entity_var_groups,
        onlyConstrains=False,
        logicMethodName="SAME",
    ):
        groups = [self._nodes(group) for group in entity_var_groups]
        if not groups:
            return self.manager.false
        matches = []
        for class_index in range(len(subclasses)):
            class_literals = [
                group[class_index]
                for group in groups
                if class_index < len(group)
            ]
            if len(class_literals) == len(groups):
                matches.append(self.manager.and_all(class_literals))
        return self.manager.or_all(matches)

    def wmc(self, node):
        return self.manager.wmc(self._node(node))

    def grounding_signature(self):
        by_concept = {}
        for concept_name, instance_id, _ in self._leaf_keys:
            by_concept.setdefault(concept_name, set()).add(instance_id)
        return tuple(sorted((name, len(ids)) for name, ids in by_concept.items()))

    @property
    def leaf_key_signature(self):
        return tuple(sorted(self._leaf_keys, key=repr))


# Conventional class-name alias while retaining the repository's existing
# lower-camel backend naming style.
CircuitBooleanMethods = circuitBooleanMethods
