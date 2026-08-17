"""Reduced ordered multi-valued BDDs used by exact semantic loss.

BDD means binary decision diagram: a graph that represents a Boolean formula
by branching on variable values until it reaches a true or false terminal.  It
is reduced by sharing equivalent nodes and ordered by testing variables in a
fixed sequence, which makes repeated logical operations compact and efficient.

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

    #: The sum-product recursion is pure broadcasting arithmetic, so a branch
    #: weight may carry a leading batch axis: one compiled circuit then scores
    #: every grounding row at once instead of once per row.
    supports_batched_weights = True

    @staticmethod
    def _scalar_probability(value) -> torch.Tensor:
        """Normalise one branch weight to a scalar or a ``[R]`` batch vector."""
        if torch.is_tensor(value):
            if value.dim() == 0 or value.numel() == 1:
                return value.reshape(())
            if value.dim() == 1:
                return value          # batched: one weight per grounding row
            raise ValueError(
                "A circuit branch probability must be scalar or 1-D (batched)")
        return torch.as_tensor(value, dtype=torch.get_default_dtype()).reshape(())

    def _batch_size(self, variables) -> int:
        """Rows the registered weights carry, or 0 when every weight is scalar."""
        size = 0
        for variable in variables:
            for weight in self._weights.get(variable, ()):
                if weight.dim() == 1:
                    if size and weight.shape[0] != size:
                        raise ValueError(
                            "Batched circuit weights disagree on the row count "
                            f"({weight.shape[0]} vs {size})")
                    size = weight.shape[0]
        return size

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

    def _reference_scalars(self):
        reference = next(
            (weight for weights in self._weights.values() for weight in weights),
            None,
        )
        if reference is None:
            reference = torch.tensor(1.0, dtype=torch.get_default_dtype())
        return reference.new_zeros(()), reference.new_ones(())

    def _normalized_weights(self, variable: int, zero, one):
        """Branch weights for *variable*, renormalised to a simplex.

        Enum softmaxes already sum to one.  Normalising also gives ``is_a``
        sibling groups categorical semantics when their binary heads are not
        perfectly calibrated.
        """
        if variable not in self._weights:
            key = self._variable_keys[variable]
            raise RuntimeError(f"No current probability tensor registered for {key!r}")
        weights = self._weights[variable]
        total = sum(weights, zero)
        # A plain scalar comparison rather than torch.allclose: this runs once
        # per variable per evaluation and allclose dominated the profile (17% of
        # a per-grounding factor-graph pass) for a check on a single 0-dim value.
        detached = total.detach()
        drift = (abs(float(detached) - 1.0) if detached.numel() == 1
                 else float((detached - 1.0).abs().max()))
        if len(weights) > 2 or drift > 1e-5:
            safe_total = total.clamp_min(torch.finfo(total.dtype).tiny)
            weights = tuple(weight / safe_total for weight in weights)
        return weights

    def wmc(self, root: BDDNode) -> torch.Tensor:
        """Evaluate differentiable weighted model count for *root*."""

        zero, one = self._reference_scalars()
        memo: Dict[int, torch.Tensor] = {0: zero, 1: one}

        def evaluate(node: BDDNode) -> torch.Tensor:
            cached = memo.get(node.node_id)
            if cached is not None:
                return cached
            weights = self._normalized_weights(node.variable, zero, one)
            value = sum(
                (weight * evaluate(child) for weight, child in zip(weights, node.children)),
                zero,
            )
            memo[node.node_id] = value
            return value

        return evaluate(root)

    def support(self, root: BDDNode) -> set:
        """Variables reachable from *root* (those the formula actually constrains)."""
        seen, stack, variables = set(), [root], set()
        while stack:
            node = stack.pop()
            if node.node_id in seen or node.is_terminal:
                continue
            seen.add(node.node_id)
            variables.add(node.variable)
            stack.extend(node.children)
        return variables

    def _scope(self, root: BDDNode, variables=None):
        """Ordered variable indices an evaluation must charge.

        The reduced support of *root*, plus any caller-declared variable the
        reduction eliminated (``A and (B or not B)`` drops ``B`` entirely).
        """
        scope = set(self.support(root))
        if variables is not None:
            for key in variables:
                if key in self._variables:
                    scope.add(self._variables[key])
        return sorted(scope)

    def _smooth_evaluate(self, root: BDDNode, order, weights_by_variable,
                         zero, one, combine):
        """Semiring evaluation that charges **every** variable in ``order``.

        ``wmc``'s plain recursion may skip a variable a node does not test,
        because a normalised group sums to one.  That identity is specific to
        the sum-product semiring and to the *value*: it does not hold for
        max-product (``max_k w_k < 1``), and it does not hold for *derivatives*
        in any semiring — a skipped variable contributes no gradient even though
        assignments over it are in the model set, which is exactly what makes a
        reduced diagram non-smooth.

        Indexing the recursion by position in ``order`` restores smoothness
        without rebuilding the diagram: at every level the variable is charged
        whether or not the current node happens to test it.

        ``combine`` picks the semiring: ``sum`` for weighted model counting,
        ``max`` for MAP.  Returns ``(value, choices)``; ``choices`` is ``None``
        for an unsatisfiable branch and a ``{variable: value_index}`` traceback
        under ``max``.
        """
        memo = {}
        track = combine is max

        def best(node: BDDNode, level: int):
            if node.is_terminal and not node.terminal_value:
                return zero, None
            if level == len(order):
                return (one, {}) if node.terminal_value else (zero, None)

            key = (node.node_id, level)
            cached = memo.get(key)
            if cached is not None:
                return cached

            variable = order[level]
            weights = weights_by_variable[variable]
            tests_here = (not node.is_terminal) and node.variable == variable

            if track:
                best_value, best_choices = zero, None
                for index, weight in enumerate(weights):
                    child = node.children[index] if tests_here else node
                    child_value, child_choices = best(child, level + 1)
                    if child_choices is None:
                        continue
                    candidate = weight * child_value
                    if (best_choices is None
                            or float(candidate.detach()) > float(best_value.detach())):
                        best_value, best_choices = candidate, {
                            variable: index, **child_choices}
                result = (best_value, best_choices)
            else:
                total, feasible = zero, False
                for index, weight in enumerate(weights):
                    child = node.children[index] if tests_here else node
                    child_value, child_choices = best(child, level + 1)
                    if child_choices is None:
                        continue
                    feasible = True
                    total = total + weight * child_value
                result = (total, {} if feasible else None)

            memo[key] = result
            return result

        return best(root, 0)

    def marginals(self, root: BDDNode, variables=None):
        """Exact constrained marginals ``P(v = k | root)`` by the derivative identity.

        ``P(v=k | phi) = w_{v,k} * dZ/dw_{v,k} / Z`` holds once two defects are
        removed:

        * **non-smoothness** — fixed by :meth:`_smooth_evaluate`, which charges
          every variable in scope so ``Z`` is multilinear in all of them; and
        * **tied weights** — the derivative is taken w.r.t. the *registered
          branch weights*, which are independent nodes of the autograd graph,
          rather than w.r.t. a source probability ``p`` from which a binary
          leaf's ``(1-p, p)`` are both derived (differentiating that mixes the
          two literals and can leave ``[0, 1]``).

        One backward pass yields every class of every variable, where
        conditioning needs one weighted model count per class.  The result stays
        differentiable in the model's probabilities (``create_graph=True``), so
        it can be used inside a training forward pass.

        Weights may carry a batch axis: with ``[R]`` weights the whole recursion
        broadcasts, the partition comes out ``[R]``, and because row ``r``'s
        partition depends only on row ``r``'s weights, ``grad(Z.sum(), w)`` is
        exactly the per-row derivative. One circuit walk then yields every
        grounding's marginals instead of one walk per grounding.

        Returns ``{variable_key: tensor[K]}``, or ``{variable_key: tensor[R, K]}``
        when the weights are batched.
        """
        zero, one = self._reference_scalars()
        order = self._scope(root, variables)
        if not order:
            return {}

        raw = {variable: self._normalized_weights(variable, zero, one)
               for variable in order}
        # Differentiate w.r.t. the *branch weights*, which are independent nodes,
        # never w.r.t. a source probability that a binary leaf's (1-p, p) share.
        # When the caller's tensors carry no graph (inference, or plain inputs)
        # stand-alone leaves are used instead and the answer is detached.
        differentiable = torch.is_grad_enabled() and any(
            w.requires_grad for weights in raw.values() for w in weights)

        with torch.enable_grad():
            if differentiable:
                weights_by_variable = {v: tuple(w * one for w in weights)
                                       for v, weights in raw.items()}
            else:
                weights_by_variable = {
                    v: tuple(w.detach().clone().requires_grad_(True) for w in weights)
                    for v, weights in raw.items()}
            flat = [w for variable in order for w in weights_by_variable[variable]]

            partition, feasible = self._smooth_evaluate(
                root, order, weights_by_variable, zero, one, sum)
            if feasible is None or float(partition.detach().min()) <= 0.0:
                raise ValueError("Cannot condition on an unsatisfiable constraint")

            # ``partition`` is [R] when batched; summing makes a scalar output for
            # autograd without mixing rows, since Z_r involves only row r.
            grads = torch.autograd.grad(
                partition.sum(), flat,
                create_graph=differentiable, allow_unused=True)

        batched = partition.dim() > 0
        out, cursor = {}, 0
        for variable in order:
            weights = weights_by_variable[variable]
            values = []
            for weight in weights:
                grad = grads[cursor]
                cursor += 1
                if grad is None:
                    values.append(torch.zeros_like(partition) if batched else zero)
                else:
                    values.append(weight * grad / partition)
            # Stack gives [K] unbatched and [K, R] batched; expose [R, K] so the
            # result lines up with the caller's belief matrices.
            row = torch.stack(values)
            if batched:
                row = row.transpose(0, 1)
            out[self._variable_keys[variable]] = row if differentiable else row.detach()
        return out

    def map_assignment(self, root: BDDNode, variables=None):
        """Most probable assignment satisfying *root* (max-product semiring).

        :param variables: variable *keys* the answer must cover. Defaults to the
            reduced support of *root*. Pass the caller's declared leaves when a
            variable may be simplified away — ``A and (B or not B)`` reduces to
            ``A``, so ``B`` leaves the support entirely; it is then unconstrained
            and takes its own argmax, but the caller usually still wants it
            scored and reported.

        Same recursion as ``wmc`` with ``sum`` replaced by ``max`` and the
        maximising branch traced back.

        **Reduction must be undone explicitly here.**  ``wmc`` may skip a
        variable a node does not test, because summing a normalised group gives
        exactly one.  Max-product has no such identity: ``max_k w_k < 1``, so
        skipping a variable silently *overstates* the score and can elect the
        wrong assignment (``A -> B`` with ``p_A=0.8, p_B=0.2`` scored ``A=False``
        at 0.2 against the true optimum ``A=B=True`` at 0.16).  The recursion is
        therefore indexed by position in the ordered support: at every level the
        variable is charged, whether or not the current node happens to test it.

        Returns ``(value, {variable_key: value_index})``; ``value`` is 0 and the
        assignment empty when *root* is unsatisfiable.
        """
        zero, one = self._reference_scalars()
        order = self._scope(root, variables)
        weights_by_variable = {
            variable: self._normalized_weights(variable, zero, one)
            for variable in order
        }
        value, choices = self._smooth_evaluate(
            root, order, weights_by_variable, zero, one, max)
        if choices is None:
            return zero, {}
        return value, {self._variable_keys[v]: k for v, k in choices.items()}

