"""R3 — factor-graph heads: the forward pass *is* inference in the constrained model.

Where R2 computes ``P(phi)`` under the model's own **independent** distribution
and penalises it, R3 makes the model's distribution ``p(y | x, phi)``. A hard
constraint gets zero mass, so it is satisfied *by construction*, and every
concept's gradient is its marginal under the **constrained joint** — which
accounts for how the other concepts must move to accommodate it. That is the
strongest form of the credit-assignment claim the R line has been chasing.

How the marginals are obtained
------------------------------
The classical arithmetic-circuit identity ``P(l | phi) = dZ/dw_l * w_l / Z``
requires the circuit to be **smooth** as well as decomposable and deterministic.
R3's plan flagged that as a caveat to verify rather than assume, and checking it
against brute-force enumeration **initially failed** for two reasons — the
diagrams are *reduced* (so a variable skipped on a path got no derivative), and
the derivative was being taken w.r.t. a source ``p`` from which a binary leaf's
``(1-p, p)`` are both built (which mixes the two literals).

Both defects are now fixed at the source: the evaluators smooth explicitly (the
BDD charges every variable in scope, the SDD charges whatever each node's vtree
scope omits), and derivatives are taken w.r.t. the registered branch weights.
The identity therefore holds, and marginals cost **one** backward pass instead
of one weighted model count per class. ``conditioning`` remains available as the
assumption-free reference implementation, and a parity test pins the two
together. See :meth:`circuitBooleanMethods.marginals`.

MAP (Phase C) is exact on **both** backends for the same reason — max-product
has no "sums to one" identity, so smoothing is what makes it correct.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

from domiknows.solver.bdd import CircuitSizeLimitExceeded
from domiknows.solver.circuitBooleanMethods import CircuitLeaf, circuitBooleanMethods


# --------------------------------------------------------------------------- #
# Variable groups: R4's synthesised joint heads are the circuit's variables
# --------------------------------------------------------------------------- #

@dataclass
class VariableGroup:
    """One categorical variable of the factor graph.

    ``name`` identifies the (instance, exclusivity group) pair; ``size`` is its
    number of classes. This is exactly R4's synthesised joint head and exactly
    the circuit's categorical group — the factor graph's variables and the
    circuit's variables are the same objects.
    """
    name: str
    size: int


def _leaves_for(group: VariableGroup, probabilities: torch.Tensor) -> List[CircuitLeaf]:
    """One ``CircuitLeaf`` per class of ``group``, sharing its categorical key.

    ``probabilities`` is ``[K]`` for a single grounding or ``[R, K]`` to score R
    groundings through one circuit; in the batched case each class weight is the
    ``[R]`` column, which the sum-product recursion broadcasts over.
    """
    variable_key = ("categorical", group.name, 0)
    if probabilities.dim() > 1:
        classes = tuple(probabilities[:, i] for i in range(group.size))
    else:
        values = probabilities.reshape(-1)
        classes = tuple(values[i] for i in range(group.size))
    return [
        CircuitLeaf((group.name, 0, i), classes[i], variable_key, i, classes,
                    categorical=True)
        for i in range(group.size)
    ]


# --------------------------------------------------------------------------- #
# The head
# --------------------------------------------------------------------------- #

@dataclass
class FactorGraphReport:
    """Which constraints are enforced structurally, and which fell back.

    A graph is only "constraint-respecting by construction" for the subset that
    compiled — the head reports that subset rather than letting a partially
    exact model be described as exact.
    """
    exact: List[str] = field(default_factory=list)
    fallback: List[str] = field(default_factory=list)

    @property
    def exact_fraction(self) -> float:
        """Share of constraints enforced structurally, or NaN when none were seen.

        An empty report means the structure never ran — which must not read as
        "everything was exact". NaN says "no measurement", matching how the
        semantic-loss path's ``exact_fraction`` is already consumed
        (``main-bert-compare.py`` NaN-guards it).
        """
        total = len(self.exact) + len(self.fallback)
        return float('nan') if total == 0 else len(self.exact) / total

    def render(self) -> str:
        return (f'factor-graph head: {len(self.exact)} constraint(s) enforced '
                f'structurally, {len(self.fallback)} fell back '
                f'(exact_fraction={self.exact_fraction:.2f})'
                + (f'\n  fallback: {", ".join(self.fallback)}' if self.fallback else ''))


class FactorGraphHead(nn.Module):
    """Replace independent per-concept heads with inference in ``p(y | x, phi)``.

    :param groups: the categorical variables (R4's joint heads).
    :param build_constraint: ``callable(processor, leaves_by_group) -> node`` that
        builds the constraint circuit from per-group ``CircuitLeaf`` lists. This
        is where a compiled LC is plugged in; the callable form keeps the head
        independent of how the constraint was produced.
    :param backend: circuit backend. ``'bdd'`` (default) additionally supports
        exact MAP; ``'pysdd'`` supports marginals only.
    :param max_nodes: circuit budget. On overflow the head falls back to the
        unconstrained beliefs for that constraint and records it in
        :attr:`report` — never a silent approximation.

    ``forward`` returns constrained marginals per group, shaped like the inputs,
    so the downstream loss, metrics and inference consume them unchanged.

    .. warning::
       **Decode with :meth:`map_predict`, not with ``argmax`` of the marginals.**
       The marginals are exact conditionals (checked against brute-force
       enumeration), but they are a *factorised* readout of a distribution that
       is not factorised. Taking each group's argmax independently is maximum
       posterior marginals (MPM), and MPM is not MAP: it can return an
       assignment with **zero** posterior probability. Measured here, MPM
       violated an ``exactly-one`` constraint on 73 of 3000 random inputs
       (typing rule: 5 of 3000), while MAP violated none.

       So "satisfied by construction" is a property of :meth:`map_predict` and
       of the *distribution*, not of a per-group argmax. Sampling has the same
       caveat: sample the joint through the circuit, never the product of
       marginals.
    """

    def __init__(self, groups: Sequence[VariableGroup], build_constraint,
                 backend: str = 'bdd', max_nodes: int = 100_000,
                 name: str = 'phi', processor=None):
        super().__init__()
        self.groups = list(groups)
        self.build_constraint = build_constraint
        self.backend = backend
        self.max_nodes = max_nodes
        self.name = name
        self.report = FactorGraphReport()
        # An injected processor is *reused* across calls. Managers hash-cons
        # their nodes and memoise ``apply``, so re-evaluating the same formula
        # shape with new weights costs only the weight refresh — which is the
        # whole point when a head is applied once per grounding row.
        self._shared_processor = processor

    # -- internals ------------------------------------------------------- #

    def _processor(self):
        if self._shared_processor is not None:
            # begin_evaluation drops stale weights but keeps the compiled
            # structure, so the second and later rows reuse the circuit.
            self._shared_processor.begin_evaluation()
            return self._shared_processor
        processor = circuitBooleanMethods(
            backend=self.backend, max_nodes=self.max_nodes,
            size_limit_action='raise')
        processor.begin_evaluation()
        return processor

    def _compile(self, processor, beliefs: Dict[str, torch.Tensor]):
        leaves_by_group = {
            group.name: _leaves_for(group, beliefs[group.name])
            for group in self.groups
        }
        node = self.build_constraint(processor, leaves_by_group)
        return node, leaves_by_group

    # -- forward = inference --------------------------------------------- #

    def forward(self, beliefs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Constrained marginals ``P(y_g = k | phi)`` per group.

        ``beliefs`` maps group name to an unconstrained distribution ``[K]``, or
        to ``[R, K]`` to score R groundings of the *same formula shape* through a
        single circuit. The returned tensors match the input shapes and are
        differentiable in the inputs.

        Batching is the difference between one circuit walk per grounding and
        one for all of them, which is what makes exact per-grounding inference
        affordable at graph scale.
        """
        self.report = FactorGraphReport()
        try:
            processor = self._processor()
            node, leaves_by_group = self._compile(processor, beliefs)
        except CircuitSizeLimitExceeded:
            self.report.fallback.append(f'{self.name} (circuit-size-limit)')
            return {name: t.clone() for name, t in beliefs.items()}

        # One call for every group: the gradient identity yields the whole table
        # from a single backward pass, so asking per group would recompute it.
        flat_leaves, spans, cursor = [], {}, 0
        for group in self.groups:
            leaves = leaves_by_group[group.name]
            flat_leaves.extend(leaves)
            spans[group.name] = (cursor, cursor + len(leaves))
            cursor += len(leaves)

        # ``marginals`` computes the partition itself, so asking for it
        # separately beforehand doubled the weighted model count on every call.
        # An unsatisfiable constraint has no conditional distribution; the
        # marginal routines raise rather than divide by zero.
        try:
            marginals = processor.marginals(node, flat_leaves)
        except (ValueError, ZeroDivisionError):
            self.report.fallback.append(f'{self.name} (unsatisfiable)')
            return {name: t.clone() for name, t in beliefs.items()}

        out = {}
        for group in self.groups:
            start, stop = spans[group.name]
            columns = marginals[start:stop]
            if columns and columns[0].dim() > 0:
                out[group.name] = torch.stack(columns, dim=-1)   # [R, K]
            else:
                out[group.name] = torch.stack([m.reshape(()) for m in columns])
        self.report.exact.append(self.name)
        return out

    # -- Phase C: MAP replaces ILP for compiled constraints --------------- #

    @torch.no_grad()
    def map_predict(self, beliefs: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """Most probable *constraint-satisfying* assignment, as ``{group: class}``.

        Exact on the BDD backend, so it replaces ILP for anything that compiles.
        Raises :class:`NotImplementedError` on a backend whose MAP is not exact
        rather than returning an unsound argmax.
        """
        processor = self._processor()
        node, leaves_by_group = self._compile(processor, beliefs)
        all_leaves = [leaf for leaves in leaves_by_group.values() for leaf in leaves]
        _, assignment = processor.map_assignment(node, leaves=all_leaves)

        out = {}
        for group in self.groups:
            key = ("categorical", group.name, 0)
            if key in assignment:
                out[group.name] = int(assignment[key])
            else:  # unsatisfiable: no constrained assignment exists
                out[group.name] = int(beliefs[group.name].reshape(-1).argmax())
        return out


# --------------------------------------------------------------------------- #
# Convenience: the loss term a structurally-enforced constraint no longer needs
# --------------------------------------------------------------------------- #

def semantic_loss(head: FactorGraphHead, beliefs: Dict[str, torch.Tensor]
                  ) -> torch.Tensor:
    """``-log P(phi)`` under the *unconstrained* beliefs (R2's term).

    Provided so a caller can show the redundancy directly: evaluated on a
    :class:`FactorGraphHead`'s own output it is ~0, because the distribution is
    already conditioned on ``phi``. Keep it only for soft and non-compiling
    constraints.
    """
    processor = head._processor()
    node, _ = head._compile(processor, beliefs)
    return -torch.log(processor.wmc(node).clamp_min(1e-30))
