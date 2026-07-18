"""Reduced ordered multi-valued BDDs used by exact semantic loss.

The public name says BDD because binary variables are the common case.  The
implementation is deliberately multi-valued: an ``EnumConcept`` instance is
one categorical random variable with K branches, not K independent Bernoulli
variables.  This is essential for weighted model counting with softmax output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Iterable, Tuple

import torch


class CircuitSizeLimitExceeded(RuntimeError):
    """Raised when the configured circuit node budget is exceeded."""


@dataclass(frozen=True, slots=True)
class BDDNode:
    """A terminal or a reduced decision node."""

    node_id: int
    variable: int | None = None
    children: Tuple["BDDNode", ...] = ()

    @property
    def is_terminal(self) -> bool:
        return self.variable is None

    @property
    def terminal_value(self) -> bool:
        if not self.is_terminal:
            raise ValueError("Only terminal nodes have a truth value")
        return self.node_id == 1


class BDDManager:
    """Hash-consed reduced ordered decision diagram manager.

    Structure is retained between evaluations, while probability tensors are
    refreshed for every batch.  Reconstructing the same grounded constraint
    therefore reuses the compiled nodes and apply-cache entries.
    """

    backend_name = "bdd"

    def __init__(self, max_nodes: int | None = 100_000, size_limit_action: str = "raise"):
        if size_limit_action not in {"raise", "warn"}:
            raise ValueError("size_limit_action must be 'raise' or 'warn'")
        self.max_nodes = max_nodes
        self.size_limit_action = size_limit_action
        self.false = BDDNode(0)
        self.true = BDDNode(1)
        self._next_node_id = 2
        self._variables: Dict[Hashable, int] = {}
        self._variable_keys: list[Hashable] = []
        self._domain_sizes: list[int] = []
        self._weights: Dict[int, Tuple[torch.Tensor, ...]] = {}
        self._unique: Dict[tuple[int, tuple[int, ...]], BDDNode] = {}
        self._apply_cache: Dict[tuple[str, int, int], BDDNode] = {}
        self._not_cache: Dict[int, BDDNode] = {0: self.true, 1: self.false}
        self._warned_size = False

    def begin_evaluation(self) -> None:
        """Forget stale tensor weights but retain all symbolic structure."""

        self._weights.clear()

    @property
    def node_count(self) -> int:
        return self._next_node_id

    @property
    def variable_count(self) -> int:
        return len(self._variable_keys)

    @property
    def variable_keys(self) -> tuple[Hashable, ...]:
        return tuple(self._variable_keys)

    def _check_size(self) -> None:
        if self.max_nodes is None or self.node_count <= self.max_nodes:
            return
        message = (
            f"Exact semantic-loss circuit exceeded max_nodes={self.max_nodes} "
            f"({self.node_count} nodes compiled)."
        )
        if self.size_limit_action == "raise":
            raise CircuitSizeLimitExceeded(message)
        if not self._warned_size:
            import warnings

            warnings.warn(message, RuntimeWarning, stacklevel=3)
            self._warned_size = True

    def _declare_variable(self, key: Hashable, domain_size: int) -> int:
        if domain_size < 2:
            raise ValueError("Circuit variables need at least two values")
        if key in self._variables:
            variable = self._variables[key]
            if self._domain_sizes[variable] != domain_size:
                raise ValueError(f"Variable {key!r} changed domain size")
            return variable
        variable = len(self._variable_keys)
        self._variables[key] = variable
        self._variable_keys.append(key)
        self._domain_sizes.append(domain_size)
        return variable

    @staticmethod
    def _scalar_probability(value) -> torch.Tensor:
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError("A circuit branch probability must be scalar")
            return value.reshape(())
        return torch.as_tensor(value, dtype=torch.get_default_dtype()).reshape(())

    def literal(
        self,
        variable_key: Hashable,
        value_index: int,
        probabilities: Iterable,
    ) -> BDDNode:
        probabilities = tuple(self._scalar_probability(p) for p in probabilities)
        variable = self._declare_variable(variable_key, len(probabilities))
        if not 0 <= value_index < len(probabilities):
            raise IndexError("Literal value is outside the variable domain")
        self._weights[variable] = probabilities
        children = tuple(
            self.true if index == value_index else self.false
            for index in range(len(probabilities))
        )
        return self._mk(variable, children)

    def _mk(self, variable: int, children: tuple[BDDNode, ...]) -> BDDNode:
        if len(children) != self._domain_sizes[variable]:
            raise ValueError("Decision-node arity does not match its variable domain")
        if all(child is children[0] for child in children[1:]):
            return children[0]
        key = (variable, tuple(child.node_id for child in children))
        node = self._unique.get(key)
        if node is None:
            node = BDDNode(self._next_node_id, variable, children)
            self._next_node_id += 1
            self._unique[key] = node
            self._check_size()
        return node

    def negate(self, node: BDDNode) -> BDDNode:
        cached = self._not_cache.get(node.node_id)
        if cached is not None:
            return cached
        result = self._mk(node.variable, tuple(self.negate(c) for c in node.children))
        self._not_cache[node.node_id] = result
        self._not_cache[result.node_id] = node
        return result

    def _apply(
        self,
        name: str,
        operation: Callable[[bool, bool], bool],
        left: BDDNode,
        right: BDDNode,
    ) -> BDDNode:
        cache_key = (name, left.node_id, right.node_id)
        cached = self._apply_cache.get(cache_key)
        if cached is not None:
            return cached
        if left.is_terminal and right.is_terminal:
            result = self.true if operation(left.terminal_value, right.terminal_value) else self.false
        else:
            left_var = left.variable if not left.is_terminal else float("inf")
            right_var = right.variable if not right.is_terminal else float("inf")
            top = int(min(left_var, right_var))
            children = []
            for value in range(self._domain_sizes[top]):
                left_child = left.children[value] if left.variable == top else left
                right_child = right.children[value] if right.variable == top else right
                children.append(self._apply(name, operation, left_child, right_child))
            result = self._mk(top, tuple(children))
        self._apply_cache[cache_key] = result
        return result

    def conjunction(self, left: BDDNode, right: BDDNode) -> BDDNode:
        return self._apply("and", lambda a, b: a and b, left, right)

    def disjunction(self, left: BDDNode, right: BDDNode) -> BDDNode:
        return self._apply("or", lambda a, b: a or b, left, right)

    def and_all(self, nodes: Iterable[BDDNode]) -> BDDNode:
        result = self.true
        for node in nodes:
            result = self.conjunction(result, node)
        return result

    def or_all(self, nodes: Iterable[BDDNode]) -> BDDNode:
        result = self.false
        for node in nodes:
            result = self.disjunction(result, node)
        return result

    def wmc(self, root: BDDNode) -> torch.Tensor:
        """Evaluate differentiable weighted model count for *root*."""

        reference = next(
            (weight for weights in self._weights.values() for weight in weights),
            None,
        )
        if reference is None:
            reference = torch.tensor(1.0, dtype=torch.get_default_dtype())
        zero = reference.new_zeros(())
        one = reference.new_ones(())
        memo: Dict[int, torch.Tensor] = {0: zero, 1: one}

        def evaluate(node: BDDNode) -> torch.Tensor:
            cached = memo.get(node.node_id)
            if cached is not None:
                return cached
            if node.variable not in self._weights:
                key = self._variable_keys[node.variable]
                raise RuntimeError(f"No current probability tensor registered for {key!r}")
            weights = self._weights[node.variable]
            total = sum(weights, zero)
            if len(weights) > 2 or not torch.allclose(
                total.detach(), one.detach(), rtol=1e-5, atol=1e-7
            ):
                # Enum softmaxes already sum to one.  Normalising also gives
                # is_a sibling groups categorical semantics when their binary
                # heads are not perfectly calibrated to a simplex.
                safe_total = total.clamp_min(torch.finfo(total.dtype).tiny)
                weights = tuple(weight / safe_total for weight in weights)
            value = sum(
                (weight * evaluate(child) for weight, child in zip(weights, node.children)),
                zero,
            )
            memo[node.node_id] = value
            return value

        return evaluate(root)

