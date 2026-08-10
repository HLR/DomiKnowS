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
    #: Weights here are per-literal scalars keyed into a shared dict, so there is
    #: no row axis to broadcast over; batched evaluation is a BDD-only fast path.
    supports_batched_weights = False

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
        self._vtree_scope_cache = {}

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

    def _evaluation_weights(self, zero, one):
        """Literal weights with each categorical group renormalised to a simplex."""
        weights = dict(self._weights)
        for group_key in self._active_categorical_groups:
            literals = self._categorical_groups[group_key]
            total = sum((weights[literal] for literal in literals), zero)
            if not torch.allclose(total.detach(), one.detach(), rtol=1e-5, atol=1e-7):
                safe_total = total.clamp_min(torch.finfo(total.dtype).tiny)
                for literal in literals:
                    weights[literal] = weights[literal] / safe_total
        return weights

    def _literal_value(self, node, weights, zero, one):
        literal = int(node.literal)
        positive = abs(literal)
        probability = weights[positive]
        if literal > 0:
            return probability
        if positive in self._binary_literals:
            return one - probability
        # A negated categorical class carries no weight of its own; the
        # exactly-one axioms already account for the group.
        return one

    def wmc(self, root):
        root = self._with_categorical_axioms(root)
        reference = next(iter(self._weights.values()), torch.tensor(1.0))
        zero, one = reference.new_zeros(()), reference.new_ones(())
        memo = {}
        weights = self._evaluation_weights(zero, one)

        def evaluate(node):
            node_id = int(node.id)
            if node_id in memo:
                return memo[node_id]
            if node.is_false():
                return zero
            if node.is_true():
                return one
            if node.is_literal():
                value = self._literal_value(node, weights, zero, one)
            else:
                value = sum(
                    (evaluate(prime) * evaluate(sub) for prime, sub in node.elements()),
                    zero,
                )
            memo[node_id] = value
            return value

        return evaluate(root)

    def _vtree_scope(self, vtree):
        """Variables under *vtree* (cached) — the scope a node there must cover."""
        if vtree is None:
            return frozenset()
        key = int(vtree.position())
        cached = self._vtree_scope_cache.get(key)
        if cached is None:
            if vtree.is_leaf():
                cached = frozenset({int(vtree.var())})
            else:
                cached = (self._vtree_scope(vtree.left())
                          | self._vtree_scope(vtree.right()))
            self._vtree_scope_cache[key] = cached
        return cached

    def map_assignment(self, root, variables=None):
        """Most probable satisfying assignment (max-product with vtree smoothing).

        An SDD is decomposable and deterministic but **not smooth**, and unlike
        weighted model counting max-product has no identity that makes an
        omitted variable free: a normalised group sums to one, but
        ``max_k w_k < 1``.  Scoring a path without charging the variables it
        omits overstates it and elects the wrong assignment.

        Smoothing is therefore explicit: every node is evaluated against the
        scope it is *expected* to cover (its vtree), and any variable the node
        does not mention is charged its own best branch.  Categorical classes
        are handled by the exactly-one axioms conjoined into the root, so an
        unmentioned class literal is correctly charged weight one for being
        false.

        Returns ``(value, {variable_key: value_index})``.
        """
        root = self._with_categorical_axioms(root)
        reference = next(iter(self._weights.values()), torch.tensor(1.0))
        zero, one = reference.new_zeros(()), reference.new_ones(())
        weights = self._evaluation_weights(zero, one)

        # literal -> (variable_key, true_index); binary literals also encode False.
        meaning, is_categorical = {}, {}
        for group_key, literals in self._categorical_groups.items():
            if group_key in self._active_categorical_groups:
                for index, literal in enumerate(literals):
                    meaning[literal] = (group_key, index)
                    is_categorical[literal] = True
        for key, literal in self._literal_by_key.items():
            if literal in self._binary_literals:
                meaning[literal] = (("binary", key), 1)
                is_categorical[literal] = False

        def branch_weights(literal):
            """``(weight_false, weight_true)`` for one SDD variable."""
            positive = weights[literal]
            if is_categorical.get(literal, False):
                # A class literal's "false" carries no weight of its own; the
                # exactly-one axioms account for the group.
                return one, positive
            return one - positive, positive

        def charge_missing(missing):
            """Best completion over variables a node does not mention."""
            value, choices = one, {}
            for literal in missing:
                if literal not in meaning:
                    continue
                w_false, w_true = branch_weights(literal)
                variable_key, true_index = meaning[literal]
                if float(w_true.detach()) >= float(w_false.detach()):
                    value = value * w_true
                    choices[variable_key] = true_index
                else:
                    value = value * w_false
                    if not is_categorical.get(literal, False):
                        choices[variable_key] = 0
            return value, choices

        memo = {}

        def evaluate(node, scope):
            key = (int(node.id), scope)
            cached = memo.get(key)
            if cached is not None:
                return cached

            if node.is_false():
                return zero, None
            if node.is_true():
                base, covered, choices = one, frozenset(), {}
            elif node.is_literal():
                literal = int(node.literal)
                positive = abs(literal)
                w_false, w_true = branch_weights(positive)
                base = w_true if literal > 0 else w_false
                covered = frozenset({positive})
                choices = {}
                if positive in meaning:
                    variable_key, true_index = meaning[positive]
                    if literal > 0:
                        choices = {variable_key: true_index}
                    elif not is_categorical.get(positive, False):
                        choices = {variable_key: 0}
            else:
                node_vtree = node.vtree()
                left = self._vtree_scope(node_vtree.left())
                right = self._vtree_scope(node_vtree.right())
                base, choices = zero, None
                for prime, sub in node.elements():
                    prime_value, prime_choices = evaluate(prime, left)
                    if prime_choices is None:
                        continue
                    sub_value, sub_choices = evaluate(sub, right)
                    if sub_choices is None:
                        continue
                    candidate = prime_value * sub_value
                    if (choices is None
                            or float(candidate.detach()) > float(base.detach())):
                        base, choices = candidate, {**prime_choices, **sub_choices}
                if choices is None:
                    memo[key] = (zero, None)
                    return zero, None
                covered = self._vtree_scope(node_vtree)

            extra_value, extra_choices = charge_missing(scope - covered)
            result = (base * extra_value, {**choices, **extra_choices})
            memo[key] = result
            return result

        if root.is_true() or root.is_false():
            scope = frozenset()
        else:
            scope = self._vtree_scope(root.vtree())

        # Variables the compilation eliminated must still be scored and
        # reported (``A and (B or not B)`` drops ``B`` from the diagram).
        if variables is not None:
            declared = set()
            for variable_key in variables:
                group = self._categorical_groups.get(variable_key)
                if group is not None:
                    declared.update(group)
                elif (isinstance(variable_key, tuple) and len(variable_key) == 2
                      and variable_key[0] == "binary"):
                    literal = self._literal_by_key.get(variable_key[1])
                    if literal is not None:
                        declared.add(literal)
            scope = scope | frozenset(declared)

        value, choices = evaluate(root, scope)
        if choices is None:
            return zero, {}
        return value, choices


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

    def miotaVar(
        self,
        _,
        *var,
        onlyConstrains=False,
        threshold=0.5,
        hard=False,
        logicMethodName="MIOTA",
    ):
        return self._nodes(var)

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
        multi_answer=False,
        threshold=None,
        logicMethodName="QUERY",
    ):
        selections = self._nodes(selection_vars)
        if subclass_data is None:
            raise ValueError("Exact queryL requires per-entity subclass data")
        if multi_answer:
            return [
                [
                    self.manager.conjunction(
                        selection,
                        self._node(subclass_data[entity_index][class_index]),
                    )
                    for class_index in range(len(subclasses))
                ]
                for entity_index, selection in enumerate(selections)
                if entity_index < len(subclass_data)
            ]
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

    # ------------------------------------------------------------------ #
    # R3 — constrained inference on the compiled circuit
    # ------------------------------------------------------------------ #

    def marginals(self, node, leaves, method="auto"):
        """Exact constrained marginals ``P(leaf | phi)``, differentiable.

        Two exact implementations, both verified against brute-force
        enumeration:

        ``'gradient'``
            The arithmetic-circuit identity ``P(l|phi) = w_l * dZ/dw_l / Z``.
            **One** backward pass yields every class of every variable. It was
            unusable until two defects were fixed: the diagrams are *reduced*
            and so not smooth (a variable skipped on a path got no derivative,
            yielding ``None``/NaN), and the derivative must be taken w.r.t. the
            *registered branch weights* — differentiating w.r.t. a source ``p``
            from which a binary leaf's ``(1-p, p)`` are both built mixes the two
            literals and can leave ``[0, 1]``. See
            :meth:`BDDManager.marginals`.

        ``'conditioning'``
            ``WMC(phi and l) / WMC(phi)``. Assumes nothing about the circuit
            beyond a correct ``wmc``, so it is the reference implementation, but
            costs one weighted model count *per queried leaf*.

        ``'auto'`` (default) uses the gradient path where the backend provides
        it and falls back to conditioning otherwise. The two agree to numerical
        precision — a parity test pins that.

        Returns a list of scalar tensors aligned with ``leaves``.
        """
        if method not in ("auto", "gradient", "conditioning"):
            raise ValueError("method must be 'auto', 'gradient' or 'conditioning'")

        root = self._node(node)
        manager_marginals = getattr(self.manager, "marginals", None)
        if method != "conditioning" and manager_marginals is not None:
            for leaf in leaves:  # register leaves the reduction may have dropped
                self._node(leaf)
            table = manager_marginals(
                root, variables=[leaf.variable_key for leaf in leaves])
            resolved = []
            for leaf in leaves:
                row = table.get(leaf.variable_key)
                if row is None:  # variable absent from the circuit entirely
                    resolved = None
                    break
                # ``row`` is [K] unbatched and [R, K] batched — index the class
                # axis from the end so both layouts work.
                resolved.append(row[..., leaf.value_index])
            if resolved is not None:
                return resolved
        if method == "gradient":
            raise NotImplementedError(
                f"{self.backend_name} backend has no gradient marginals; "
                "use method='conditioning'")

        partition = self.manager.wmc(root)
        if float(partition.detach()) <= 0.0:
            # Match the gradient path, which raises: conditioning on an
            # unsatisfiable constraint is undefined, and dividing would hand the
            # caller a silent inf/nan instead of a reportable failure.
            raise ValueError("Cannot condition on an unsatisfiable constraint")
        return [self.manager.wmc(self.manager.conjunction(root, self._node(leaf)))
                / partition for leaf in leaves]

    #: Backends whose MAP is exact. Both are smoothed explicitly — the BDD by
    #: charging every variable in scope, the SDD by charging what each node's
    #: vtree scope omits.
    MAP_BACKENDS = ("bdd", "pysdd")

    @property
    def supports_map(self):
        return hasattr(self.manager, "map_assignment")

    def map_assignment(self, node, leaves=None):
        """Most probable assignment satisfying the constraint (max-product).

        Returns ``(value, {variable_key: value_index})``. Constraint-respecting
        by construction, so it replaces ILP for anything that compiles.

        :param leaves: optional ``CircuitLeaf`` iterable declaring the variables
            the answer must cover, for leaves the reduction may eliminate.

        Exact on the BDD backend only. The SDD backend is not smooth, and
        max-product — unlike weighted model counting — cannot treat a skipped
        variable as free, so MAP there would be quietly wrong; it raises instead.
        """
        manager_map = getattr(self.manager, "map_assignment", None)
        if manager_map is None:
            raise NotImplementedError(
                f"MAP inference is exact only on the {'/'.join(self.MAP_BACKENDS)} "
                f"backend; this processor uses {self.backend_name!r}. Construct "
                f"circuitBooleanMethods(backend='bdd') for MAP. (Weighted model "
                f"counting and marginals are exact on every backend.)")
        variables = None
        if leaves is not None:
            variables = []
            for leaf in leaves:
                # Register the leaf so a variable simplified out of the diagram
                # still has weights available to score and report.
                self._node(leaf)
                variables.append(leaf.variable_key)
        return manager_map(self._node(node), variables=variables)

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
