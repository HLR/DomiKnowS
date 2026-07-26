"""R4 Phase B — constraint refinement layer.

The refinement layer sits *between* the encoder/heads and the loss: it reads
every concept's current beliefs, exchanges messages along the constraint
structure, and returns corrected beliefs. Constraints stop being only an
output-side score and start shaping the representation that feeds the heads.

Design — messages are the constraint's own violation gradient
-------------------------------------------------------------
The message a rule sends is derived from its (differentiable) violation, exactly
as the plan specifies. Concretely, one refinement step is a *step of
constraint-descent*::

    logits' = logits - step * W_c * d(sum_g violation_g)/d(logits_c)

so the update provably moves every participating concept toward satisfying the
rule, *by construction* — no training is needed for the direction to be correct,
which is what makes the verification exact rather than statistical. ``W_c`` is a
learned per-concept gate (the only free parameter); zeroing it, or zeroing the
messages, must undo the correction, which is the "not a free-parameter win"
ablation.

Crucially the internal violation uses **Product** semantics, not Gödel: under
Gödel the antecedent of an implication gets exactly zero gradient (the defect the
whole R-line is about), so a Gödel message could never correct a relation that is
over-confident about a mistyped argument. Product gives every literal a non-zero,
sign-correct gradient.

Scope, stated honestly
----------------------
Factors are built for the constraint families the grounding join already
covers: the **typing implication** ``ifL(rel('x','y'), andL(A('x'), B('y')))``
(conll04's rules) and **mutual exclusion** over a set of literals on one node.
Constraints outside that (nested LC results, ``eqL``-as-element) have no edges
yet and are skipped explicitly by the factor builder rather than contributing a
silently-wrong message — see :func:`build_typing_factors`.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import torch
from torch import nn


# --------------------------------------------------------------------------- #
# Factors: the edges of the constraint graph
# --------------------------------------------------------------------------- #

@dataclass
class Literal:
    """One participating (concept, class) reference and its per-row node map.

    ``node_index[r]`` is the row of ``concept``'s belief matrix that factor row
    ``r`` reads/writes. For a typing rule this is exactly the grounding binding:
    the relation literal reads node ``r``; the ``A('x')`` consequent reads node
    ``r // n_dest``; the ``B('y')`` consequent reads node ``r % n_dest``.
    """
    concept: str
    class_index: int
    node_index: torch.Tensor  # long [R]


@dataclass
class Factor:
    """A grounded constraint template and its differentiable violation.

    ``kind`` selects the violation form; ``literals`` are its operands in a
    fixed order (``violation`` interprets that order per kind).
    """
    kind: str
    literals: List[Literal]
    name: str = ''

    def gather(self, beliefs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Per-literal probability vector ``[R]`` for the referenced class."""
        cols = []
        for lit in self.literals:
            probs = beliefs[lit.concept]                      # [N x K]
            cols.append(probs[lit.node_index, lit.class_index])
        return cols

    def violation(self, beliefs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return _VIOLATIONS[self.kind](self.gather(beliefs))


def _violation_implication(cols: Sequence[torch.Tensor]) -> torch.Tensor:
    """``ifL(rel, andL(A, B))`` under Product: ``p_rel * (1 - p_A * p_B)``.

    ``cols == [p_rel, p_A, p_B, ...]``; extra consequents multiply into the
    conjunction. This is precisely the interpreter's product ``ifVar`` loss, so
    the message is the compiled violation's gradient, not a bespoke surrogate.
    """
    p_rel = cols[0]
    cons = cols[1]
    for extra in cols[2:]:
        cons = cons * extra
    return p_rel * (1.0 - cons)


def _violation_exclusion(cols: Sequence[torch.Tensor]) -> torch.Tensor:
    """At-most-one over a literal set: ``sum_{i<j} p_i * p_j`` (pairwise Product nandL).

    This is exactly ``nandL``/``atMostL(...,1)``'s violation: it penalises — and
    its gradient separates — every jointly-high pair, and nothing else.
    """
    total = torch.zeros_like(cols[0])
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            total = total + cols[i] * cols[j]
    return total


def _violation_at_least_one(cols: Sequence[torch.Tensor]) -> torch.Tensor:
    """At-least-one (``orL`` / ``atLeastL(...,1)``): ``prod_i (1 - p_i)``.

    High (violated) only when every literal is off; the gradient raises whichever
    is closest to on — the Product ``orL`` loss.
    """
    prod = torch.ones_like(cols[0])
    for c in cols:
        prod = prod * (1.0 - c)
    return prod


def _violation_exactly_one(cols: Sequence[torch.Tensor]) -> torch.Tensor:
    """Exactly-one (``exactL(...,1)``): at-most-one *and* at-least-one.

    The sum of the two one-sided penalties, so the message both separates
    jointly-high literals and lifts an all-off group — the full ``exactL(...,1)``
    correction, not just the ``<=1`` half.
    """
    return _violation_exclusion(cols) + _violation_at_least_one(cols)


#: kind -> (list of per-row prob vectors) -> per-row violation [R]
_VIOLATIONS: Dict[str, Callable[[Sequence[torch.Tensor]], torch.Tensor]] = {
    'implication': _violation_implication,
    'exclusion': _violation_exclusion,
    'at_least_one': _violation_at_least_one,
    'exactly_one': _violation_exactly_one,
}


# --------------------------------------------------------------------------- #
# The refinement module
# --------------------------------------------------------------------------- #

class ConstraintRefinement(nn.Module):
    """Iterative belief refinement by constraint-violation descent.

    :param concepts: ``{concept_name: K}`` — the number of classes each concept
        head emits (2 for a binary concept, K for an ``EnumConcept``).
    :param steps: number of unrolled refinement steps (k ≈ 2–3). Backprop flows
        through the unrolling when the module is in ``train()`` mode.
    :param step_size: base learning rate of the inner constraint-descent step.
    :param learn_gate: if True, a per-concept gate ``W_c`` (init 1.0) is learned;
        this is the layer's only free parameter, so the ablation that zeros it
        recovers the un-refined beliefs exactly.
    """

    def __init__(self, concepts: Dict[str, int], steps: int = 2,
                 step_size: float = 1.0, learn_gate: bool = True):
        super().__init__()
        self.concept_k = dict(concepts)
        self.steps = steps
        self.step_size = step_size
        self.learn_gate = learn_gate
        if learn_gate:
            self.gate = nn.ParameterDict({
                name: nn.Parameter(torch.ones(())) for name in concepts})
        else:
            self.gate = None

    def _gate(self, name: str, ref: torch.Tensor) -> torch.Tensor:
        # An unknown concept keeps an ungated (weight-1) message rather than
        # raising: the gate set is derived from the constraint declarations, and
        # a runtime concept outside that set should still receive its message.
        if self.gate is None or name not in self.gate:
            return torch.ones((), dtype=ref.dtype, device=ref.device)
        return self.gate[name].to(ref.device)

    def forward(self, logits: Dict[str, torch.Tensor], factors: List[Factor],
                zero_messages: bool = False) -> Dict[str, torch.Tensor]:
        """Refine ``logits`` (per concept ``[N x K]``) under ``factors``.

        Returns refined logits with the same shapes/keys. ``zero_messages``
        disables the message pass (the ablation), so the output equals the input
        — a direct check that any improvement came from the messages.
        """
        cur = {name: t for name, t in logits.items()}
        if zero_messages or not factors:
            return {name: t.clone() for name, t in cur.items()}

        # Only concepts that some factor actually touches can receive a message;
        # leave the rest untouched (and out of the autograd.grad input list, so
        # allow_unused stays a guard rather than the norm).
        touched = []
        for factor in factors:
            for lit in factor.literals:
                if lit.concept not in touched:
                    touched.append(lit.concept)

        for _ in range(self.steps):
            # Re-root the graph each step: the messages are gradients of *this
            # step's* beliefs, and unrolling backprops through the chain.
            work = {name: cur[name].requires_grad_(True) if not cur[name].requires_grad
                    else cur[name] for name in cur}
            probs = {name: torch.softmax(work[name], dim=-1) for name in work}

            total = None
            for factor in factors:
                v = factor.violation(probs).sum()
                total = v if total is None else total + v
            if total is None:
                break

            inputs = [work[name] for name in touched]
            grads = torch.autograd.grad(
                total, inputs, create_graph=self.training, allow_unused=True)

            updated = dict(cur)
            for name, grad in zip(touched, grads):
                if grad is None:
                    continue
                # logits' = logits - step * W_c * d(violation)/d(logits_c)
                # (message = -gradient; W_c >= 0 keeps it a descent direction).
                updated[name] = cur[name] - self.step_size * self._gate(name, grad) * grad
            cur = updated

        return cur


# --------------------------------------------------------------------------- #
# Building factors from the grounded graph (reuse, don't rebuild)
# --------------------------------------------------------------------------- #
#
# Coverage. A refinement message is only sound when both its edge map and its
# violation are *correct* for the constraint — a plausible-but-wrong message is
# worse than none, so any shape we cannot ground faithfully is skipped and
# reported, never approximated. The shapes built here:
#
#   implication   ifL(rel('x','y'), andL(A('x'), B('y')))   typing rule
#                 ifL(A('x'), B('x'))  /  ifL(A('x'), andL(B('x'), C('x')))
#                 ifL(rel1('x','y'), rel2('x','y'))          co-grounded
#   exclusion     nandL(...) , atMostL(..., limit=1)         at most one
#   at_least_one  orL(...)   , atLeastL(..., limit=1)        at least one
#   exactly_one   exactL(..., limit=1)                       exactly one
#
# Two grounding patterns cover their edges exactly:
#   * typing consequents are bound to a relation argument via a ``has_a`` path,
#     so a relation row maps to the linked entity row (same map ``groundingBinding``
#     recovers as ``(r // n_dest, r % n_dest)``);
#   * every other literal ranges over the *same* domain as the reference literal
#     (all entities, or all pairs), so its edges are the identity.
#
# Still skipped, deliberately: counting with limit != 1, sumL, iotaL/queryL,
# sameL/differentL, eqL-as-element, disjunctive/negated or nested-LC operands,
# and any literal whose domain cannot be aligned to the reference. These have no
# correct local message yet.


def _is_concept_tuple(el):
    from domiknows.graph.concept import Concept
    return isinstance(el, tuple) and len(el) >= 3 and isinstance(el[0], Concept)


def _is_v(el):
    """A variable binding ``V`` (a namedtuple with ``.name``/``.v``) — not a
    concept tuple (which is a plain tuple) nor a ``VarMaps`` error tuple."""
    return (hasattr(el, 'v') and hasattr(el, 'name')
            and not _is_concept_tuple(el))


def _literal_binding(v):
    """How a literal is grounded, from the ``V`` that follows its concept tuple.

    ``('argpath', rel_var, arg_name)`` when the literal is bound to a relation
    argument (``V.v == (rel_var, HasA(arg_name))``); ``('same',)`` otherwise,
    meaning it ranges over its own concept's domain (identity edges against the
    reference domain).
    """
    from domiknows.graph.relation import HasA
    path = getattr(v, 'v', None)
    if (isinstance(path, tuple) and len(path) >= 2
            and isinstance(path[0], str) and isinstance(path[1], HasA)):
        return ('argpath', path[0], path[1].name)
    return ('same',)


def _is_skippable(el):
    """Elements that carry no literal: ``VarMaps`` tuples, VARMAP dicts, the
    trailing count int. Anything else unrecognised makes the caller bail."""
    if isinstance(el, (int, dict)):
        return True
    if isinstance(el, tuple) and el and el[0] == 'VarMaps':
        return True
    return False


def _concept_bindings(elements):
    """Flatten ``elements`` to ``[(concept_tuple, binding), ...]`` in order.

    Pairs each concept tuple with the ``V`` that follows it, descends one level
    of ``andL``, and skips the non-literal elements. Returns None on anything it
    cannot account for (a disjunction/negation, a nested non-``andL`` LC, a
    concept with no binding), so an unsupported constraint is skipped whole
    rather than half-read.
    """
    from domiknows.graph.logicalConstrain import andL

    pairs = []
    i, n = 0, len(elements)
    while i < n:
        el = elements[i]
        if _is_concept_tuple(el):
            if i + 1 >= n or not _is_v(elements[i + 1]):
                return None
            pairs.append((el, _literal_binding(elements[i + 1])))
            i += 2
            continue
        if isinstance(el, andL):
            sub = _concept_bindings(el.e)
            if sub is None:
                return None
            pairs.extend(sub)
            i += 1
            continue
        if _is_skippable(el):
            i += 1
            continue
        return None
    return pairs


def _count_limit(lc):
    """Resolve a counting LC's limit exactly as ``_CountBaseL.__call__`` does."""
    fixed = getattr(lc, 'fixedLimit', None)
    if fixed is not None:
        return fixed
    explicit = getattr(lc, '_explicitLimit', None)
    if explicit is not None:
        return explicit
    e = getattr(lc, 'e', ())
    if e and isinstance(e[-1], int):
        return e[-1]
    return 1


def _factor_kind(lc):
    """The violation kind for ``lc``, or None if it has no faithful local factor."""
    from domiknows.graph.logicalConstrain import (
        ifL, nandL, orL, atMostL, atLeastL, exactL)
    t = type(lc)
    if t is ifL:
        return 'implication'
    if t is nandL:
        return 'exclusion'
    if t is orL:
        return 'at_least_one'
    if t is atMostL:
        return 'exclusion' if _count_limit(lc) == 1 else None
    if t is atLeastL:
        return 'at_least_one' if _count_limit(lc) == 1 else None
    if t is exactL:
        return 'exactly_one' if _count_limit(lc) == 1 else None
    return None


def _parse_literals(lc, kind):
    """Ordered ``[(concept_tuple, binding), ...]`` for ``lc``.

    For an implication the antecedent must be a single concept literal (a
    conjunctive antecedent is a different shape and is skipped); the rest are the
    consequents. Other kinds are a flat set of literals.
    """
    if kind == 'implication':
        els = lc.e
        if len(els) < 3 or not _is_concept_tuple(els[0]) or not _is_v(els[1]):
            return None
        antecedent = (els[0], _literal_binding(els[1]))
        consequents = _concept_bindings(els[2:])
        if not consequents:
            return None
        return [antecedent] + consequents
    return _concept_bindings(lc.e)


def build_constraint_factors(datanode, graph, key='/local/softmax'):
    """Build refinement factors from a populated datanode's head constraints.

    Emits one :class:`Factor` per supported constraint (see the coverage note
    above), with ``node_index`` tensors taken from the *actual* grounding — the
    ``has_a`` links for typing consequents, the identity for co-grounded
    literals. Unsupported or unresolvable constraints are returned in
    ``skipped`` so a caller can report coverage rather than drop them silently.
    """
    from domiknows.graph.candidates import findDatanodesForRootConcept

    factors: List[Factor] = []
    skipped: List[str] = []
    row_by_id: Dict[str, Dict[int, int]] = {}

    def rows_for(conceptName):
        if conceptName not in row_by_id:
            root = datanode.findRootConceptOrRelation(conceptName)
            dns = findDatanodesForRootConcept(datanode, root) if root is not None else []
            row_by_id[conceptName] = {id(dn): i for i, dn in enumerate(dns or [])}
        return row_by_id[conceptName]

    # Recursive so constraints declared in a subgraph (as in conll04) are seen.
    rec = getattr(graph, 'logicalConstrainsRecursive', None)
    constraints = rec if rec is not None else getattr(graph, 'logicalConstrains', {}).items()
    for _, lc in constraints:
        if not getattr(lc, 'headLC', False) or not getattr(lc, 'active', True):
            continue
        name = getattr(lc, 'lcName', repr(lc))
        factor = _build_factor(datanode, lc, findDatanodesForRootConcept, rows_for)
        if factor is None:
            skipped.append(name)
        else:
            factors.append(factor)
    return factors, skipped


#: Backwards-compatible alias; the builder now covers more than typing rules.
build_typing_factors = build_constraint_factors


def _build_factor(datanode, lc, find_dns, rows_for):
    kind = _factor_kind(lc)
    if kind is None:
        return None
    specs = _parse_literals(lc, kind)
    if specs is None or len(specs) < 2:
        return None

    # The reference domain (antecedent for an implication; the first literal
    # otherwise) fixes the factor's row count and order.
    ref_concept = specs[0][0][0]
    ref_root = datanode.findRootConceptOrRelation(ref_concept.name)
    ref_dns = find_dns(datanode, ref_root) if ref_root is not None else []
    if not ref_dns:
        return None

    literals = []
    for concept_tuple, binding in specs:
        node_index = _node_index(datanode, find_dns, ref_dns,
                                 concept_tuple, binding, rows_for)
        if node_index is None:
            return None
        literals.append(Literal(concept_tuple[0].name,
                                _pos_index(concept_tuple), node_index))
    return Factor(kind, literals, name=getattr(lc, 'lcName', ''))


def _node_index(datanode, find_dns, ref_dns, concept_tuple, binding, rows_for):
    """Map each reference row to the literal's belief-matrix row.

    ``argpath`` literals resolve through the pair's ``has_a`` link; ``same``
    literals must range over the reference domain and get the identity map.
    Returns None when the literal cannot be aligned, so the constraint is
    skipped rather than mis-wired.
    """
    R = len(ref_dns)
    conceptName = concept_tuple[0].name

    if binding[0] == 'argpath':
        _, _rel_var, arg_name = binding
        mapping = rows_for(conceptName)
        idx = torch.empty(R, dtype=torch.long)
        for r, pair_dn in enumerate(ref_dns):
            links = getattr(pair_dn, 'relationLinks', None)
            if not links:
                return None
            arg_dn = _arg_datanode(links, arg_name)
            if arg_dn is None:
                return None
            row = mapping.get(id(arg_dn))
            if row is None:
                return None
            idx[r] = row
        return idx

    # 'same': the literal ranges over the reference domain (identity edges). Only
    # sound when it really is the same node set in the same order.
    root = datanode.findRootConceptOrRelation(conceptName)
    dns = find_dns(datanode, root) if root is not None else []
    if len(dns) != R or any(a is not b for a, b in zip(dns, ref_dns)):
        return None
    return torch.arange(R, dtype=torch.long)


def _pos_index(e):
    """Positive-class column for a concept tuple.

    ``EnumConcept`` references carry their value index in ``e[2]``; a plain
    binary concept's positive class is column 1.
    """
    return e[2] if e[2] is not None else 1


def _arg_datanode(links, arg_name):
    """The datanode a pair links through ``has_a`` argument ``arg_name``.

    ``relationLinks`` keys are the argument relation names; match exactly, then
    fall back to endswith/contains so name-decorated keys still resolve.
    """
    for match in (lambda k: k == arg_name,
                  lambda k: k.endswith(arg_name),
                  lambda k: arg_name in k):
        for k, linked in links.items():
            if match(k) and linked:
                return linked[0]
    return None
