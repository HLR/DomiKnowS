"""R4 Phase B — constraint refinement layer.

The scientific risk for the whole R line is a refinement layer that trains fine
while doing nothing: extra parameters can absorb the loss and mask a no-op. So
these tests check *mechanism*, not just that a number moved:

* the refined beliefs move in the **direction each constraint demands**, per
  concept (not merely "the loss fell");
* zeroing the messages, zeroing the gate, or **shuffling the edge map** each
  undoes the correction — distinguishing "constraints helped" from "capacity
  helped";
* the factor edges equal the grounding binding
  (:meth:`ProbabilityStore.decode_relation_rows` vs ``groundingBinding``);
* :func:`build_typing_factors` recovers those edges from a real graph +
  datanode.

Run with ``-s`` to see the per-concept belief deltas.
"""

import pytest
import torch

from domiknows.program.model.refinement import (
    ConstraintRefinement, Factor, Literal,
    build_typing_factors, build_constraint_factors,
    _violation_implication, _violation_exclusion,
    _violation_at_least_one, _violation_exactly_one,
)


# --------------------------------------------------------------------------- #
# Factor violation semantics
# --------------------------------------------------------------------------- #

def test_implication_violation_is_product_ifL():
    # v = p_rel * (1 - p_A * p_B)
    cols = [torch.tensor([0.9]), torch.tensor([0.2]), torch.tensor([0.5])]
    v = _violation_implication(cols)
    assert v.item() == pytest.approx(0.9 * (1 - 0.2 * 0.5), abs=1e-6)


def test_exclusion_violation_is_pairwise_product():
    cols = [torch.tensor([0.6]), torch.tensor([0.7]), torch.tensor([0.2])]
    v = _violation_exclusion(cols)
    expected = 0.6 * 0.7 + 0.6 * 0.2 + 0.7 * 0.2
    assert v.item() == pytest.approx(expected, abs=1e-6)


def test_at_least_one_and_exactly_one_violations():
    cols = [torch.tensor([0.1]), torch.tensor([0.2])]
    at_least = _violation_at_least_one(cols)
    assert at_least.item() == pytest.approx(0.9 * 0.8, abs=1e-6)          # both off
    # exactly-one = at-most-one (pairwise) + at-least-one (all-off)
    exactly = _violation_exactly_one(cols)
    assert exactly.item() == pytest.approx(0.1 * 0.2 + 0.9 * 0.8, abs=1e-6)


# --------------------------------------------------------------------------- #
# A violated typing rule and its refinement
# --------------------------------------------------------------------------- #

K = 5  # entity_label / pair_label class count (conll04-shaped)


def _violated_case():
    """work_for confident on a pair whose arg1 is *not* people → violated.

    entity 0 is arg1, entity 1 is arg2; class 0 = people/work_for, 1 = organization.
    """
    ent = torch.zeros(2, K)
    pair = torch.zeros(1, K)
    pair[0, 0] = 3.0     # work_for confident
    ent[0, 0] = -1.0     # people(entity0) low  → the violation
    logits = {'entity_label': ent.clone().requires_grad_(True),
              'pair_label': pair.clone().requires_grad_(True)}
    factor = Factor('implication', [
        Literal('pair_label', 0, torch.tensor([0])),     # work_for @ pair 0
        Literal('entity_label', 0, torch.tensor([0])),   # people(arg1 = entity0)
        Literal('entity_label', 1, torch.tensor([1])),   # organization(arg2 = entity1)
    ], name='work_for_typing')
    return logits, factor


def _probs(logits):
    return {name: torch.softmax(t.detach(), dim=-1) for name, t in logits.items()}


def test_refinement_moves_each_concept_in_the_constraint_direction(capsys):
    """Verification 2: direction per concept, not just that the loss fell.

    The rule is violated because work_for is high while arg1 is not people. A
    correct correction must lower work_for AND raise people(arg1) and
    organization(arg2). Product messages give the antecedent a real gradient —
    the whole point over Gödel, which would leave work_for untouched.
    """
    logits, factor = _violated_case()
    before = _probs(logits)

    ref = ConstraintRefinement({'entity_label': K, 'pair_label': K},
                               steps=5, step_size=5.0)
    ref.eval()
    refined = ref(logits, [factor])
    after = _probs(refined)

    v_before = factor.violation(before).item()
    v_after = factor.violation(after).item()

    p_wf = (before['pair_label'][0, 0].item(), after['pair_label'][0, 0].item())
    p_pe = (before['entity_label'][0, 0].item(), after['entity_label'][0, 0].item())
    p_or = (before['entity_label'][1, 1].item(), after['entity_label'][1, 1].item())
    print(f'\nviolation {v_before:.4f} -> {v_after:.4f}')
    print(f'P(work_for)      {p_wf[0]:.4f} -> {p_wf[1]:.4f}  (want down)')
    print(f'P(people@arg1)   {p_pe[0]:.4f} -> {p_pe[1]:.4f}  (want up)')
    print(f'P(org@arg2)      {p_or[0]:.4f} -> {p_or[1]:.4f}  (want up)')

    assert v_after < v_before                    # the specific violation drops
    assert p_wf[1] < p_wf[0] - 1e-3              # antecedent lowered (vs Gödel!)
    assert p_pe[1] > p_pe[0] + 1e-4              # consequent raised
    assert p_or[1] > p_or[0] + 1e-4              # consequent raised


def test_zero_messages_is_identity():
    """Verification 3a: with no messages, refinement changes nothing."""
    logits, factor = _violated_case()
    ref = ConstraintRefinement({'entity_label': K, 'pair_label': K}, steps=3)
    ref.eval()
    out = ref(logits, [factor], zero_messages=True)
    for name in logits:
        assert torch.allclose(out[name], logits[name])


def test_zero_gate_is_identity():
    """Verification 3 (free-parameter guard): a zero gate does nothing.

    The gate is the layer's only parameter, so W=0 must recover the input — a
    refinement that still 'helps' with a zeroed gate would be capacity, not
    constraints.
    """
    logits, factor = _violated_case()
    ref = ConstraintRefinement({'entity_label': K, 'pair_label': K},
                               steps=5, step_size=5.0)
    with torch.no_grad():
        for p in ref.gate.values():
            p.zero_()
    ref.eval()
    out = ref(logits, [factor])
    for name in logits:
        assert torch.allclose(out[name], logits[name], atol=1e-6)


def test_shuffled_edges_correct_the_wrong_node():
    """Verification 3b: the edge map decides *which* node is corrected.

    With correct edges people(entity0) rises. Point the people literal at
    entity1 instead and entity0 is left alone while entity1 moves — proof the
    messages follow the binding, not just 'some entity'.
    """
    logits, factor = _violated_case()
    before = _probs(logits)

    ref = ConstraintRefinement({'entity_label': K, 'pair_label': K},
                               steps=5, step_size=5.0)
    ref.eval()

    correct = _probs(ref(dict(logits), [factor]))

    shuffled = Factor('implication', [
        Literal('pair_label', 0, torch.tensor([0])),
        Literal('entity_label', 0, torch.tensor([1])),   # people now reads entity1
        Literal('entity_label', 1, torch.tensor([0])),   # organization reads entity0
    ])
    shuffled_out = _probs(ref(dict(logits), [shuffled]))

    # Correct binding raises people(entity0); the shuffled binding does not —
    # it sends no people-message to entity0 (the only movement there is the
    # softmax coupling from an organization-message misdirected onto the row).
    assert correct['entity_label'][0, 0] > before['entity_label'][0, 0] + 1e-4
    assert shuffled_out['entity_label'][0, 0] < correct['entity_label'][0, 0] - 1e-3
    assert shuffled_out['entity_label'][0, 0] <= before['entity_label'][0, 0] + 1e-6
    # ...and instead the people-message lands on entity1, raising it there.
    assert shuffled_out['entity_label'][1, 0] > before['entity_label'][1, 0] + 1e-4


def test_unrolling_backprops_into_the_gate():
    """The learned gate must receive gradient through the unrolled steps.

    A downstream objective (here, the residual violation) trained end-to-end has
    to be able to tune the gate; if the unrolling did not backprop, the gate
    would have no gradient and never learn.
    """
    logits, factor = _violated_case()
    ref = ConstraintRefinement({'entity_label': K, 'pair_label': K},
                               steps=3, step_size=2.0)
    ref.train()
    refined = ref(logits, [factor])
    probs = {name: torch.softmax(t, dim=-1) for name, t in refined.items()}
    loss = factor.violation(probs).sum()
    loss.backward()
    grads = [p.grad for p in ref.gate.values() if p.grad is not None]
    assert grads, 'no gradient reached the refinement gate'
    assert any(g.abs().item() > 0 for g in grads)


def test_refinement_reduces_multiple_factors_jointly():
    """Messages from several factors on a shared node aggregate (sum) for free."""
    logits, f1 = _violated_case()
    # a second pair (row 0 reused) demanding organization(entity0) too
    f2 = Factor('implication', [
        Literal('pair_label', 1, torch.tensor([0])),
        Literal('entity_label', 1, torch.tensor([0])),   # organization(entity0)
        Literal('entity_label', 0, torch.tensor([1])),
    ])
    with torch.no_grad():
        logits['pair_label'][0, 1] = 3.0
    before = _probs(logits)
    ref = ConstraintRefinement({'entity_label': K, 'pair_label': K},
                               steps=5, step_size=5.0)
    ref.eval()
    after = _probs(ref(logits, [f1, f2]))
    assert f1.violation(after).item() < f1.violation(before).item()
    assert f2.violation(after).item() < f2.violation(before).item()


# --------------------------------------------------------------------------- #
# Verification 4: factor edges equal the grounding binding
# --------------------------------------------------------------------------- #

def test_decode_relation_rows_matches_grounding_binding():
    """The tensor edge map equals ``groundingBinding``'s ``(r//n_dest, r%n_dest)``."""
    from domiknows.solver.compiled.grounding import ProbabilityStore
    from domiknows.solver.logicalConstraintConstructor import LogicalConstraintConstructor

    n_src, n_dest = 4, 3
    n_rows = n_src * n_dest
    src, dst = ProbabilityStore.decode_relation_rows(n_rows, n_dest)

    class _Var:
        pass
    variable = _Var()
    variable.relVarInfo = {'x': object(), 'y': object()}
    lcVariablesDns = {'x': list(range(n_src)), 'y': list(range(n_dest))}
    names, keys = LogicalConstraintConstructor.groundingBinding(
        variable, [None] * n_rows, lcVariablesDns)

    assert names == ('x', 'y')
    assert src.tolist() == [k[0] for k in keys]
    assert dst.tolist() == [k[1] for k in keys]


def test_build_typing_factors_recovers_bindings(monkeypatch):
    """``build_typing_factors`` on a real graph + datanode binds A→src, B→dst.

    The full sensor/populate path is too heavy (and irrelevant) here, so the
    datanode is faked down to exactly what the builder touches —
    ``findRootConceptOrRelation`` and each pair's ``relationLinks`` — and
    ``findDatanodesForRootConcept`` is patched to return those fakes. The
    *logic under test* (typing-rule decode, name-based arg matching, row
    lookup) is the real code.
    """
    import domiknows.graph.candidates as candidates
    from domiknows.graph import Graph, Concept, Relation
    from domiknows.graph.logicalConstrain import ifL, andL
    from domiknows.solver.compiled.grounding import ProbabilityStore

    Graph.clear(); Concept.clear(); Relation.clear()
    with Graph('r4_typing') as graph:
        entity = Concept(name='entity')
        A = entity(name='A'); B = entity(name='B')
        pair = Concept(name='pair')
        (arg1, arg2) = pair.has_a(a1=entity, a2=entity)
        link = pair(name='link')
        ifL(link('x', 'y'), andL(A('x'), B('y')))

    class _DN:
        def __init__(self, name):
            self.name = name
            self.relationLinks = {}

    n_dest = 3
    entities = [_DN(f'e{i}') for i in range(n_dest)]
    pairs = []
    for s in range(n_dest):
        for d in range(n_dest):
            p = _DN(f'p{s}{d}')
            p.relationLinks = {'a1': [entities[s]], 'a2': [entities[d]]}
            pairs.append(p)

    concept_by_name = {'link': link, 'A': A, 'B': B, 'entity': entity}

    class _Root:
        def findRootConceptOrRelation(self, name):
            return concept_by_name.get(name)

    def fake_find(dn, root):
        name = root if isinstance(root, str) else root.name
        if name == 'link':
            return pairs
        if name in ('A', 'B', 'entity'):
            return entities
        return []

    monkeypatch.setattr(candidates, 'findDatanodesForRootConcept', fake_find)

    factors, skipped = build_typing_factors(_Root(), graph)
    assert not skipped
    assert len(factors) == 1

    lits = {lit.concept + str(i): lit for i, lit in enumerate(factors[0].literals)}
    rel, a_lit, b_lit = factors[0].literals
    src, dst = ProbabilityStore.decode_relation_rows(len(pairs), n_dest)

    assert rel.concept == 'link'
    assert rel.node_index.tolist() == list(range(len(pairs)))
    assert a_lit.concept == 'A' and a_lit.node_index.tolist() == src.tolist()
    assert b_lit.concept == 'B' and b_lit.node_index.tolist() == dst.tolist()


# --------------------------------------------------------------------------- #
# The other constraint shapes: refinement direction
# --------------------------------------------------------------------------- #

def test_refinement_separates_jointly_high_exclusive_pair():
    """Exclusion (nandL/atMostL-1): two co-active siblings are pushed apart."""
    logits = {'people': torch.tensor([[0.0, 2.0]], requires_grad=True),
              'org': torch.tensor([[0.0, 2.0]], requires_grad=True)}
    factor = Factor('exclusion', [
        Literal('people', 1, torch.tensor([0])),
        Literal('org', 1, torch.tensor([0])),
    ])
    before = _probs(logits)
    ref = ConstraintRefinement({'people': 2, 'org': 2}, steps=5, step_size=5.0)
    ref.eval()
    after = _probs(ref(logits, [factor]))

    assert factor.violation(after).item() < factor.violation(before).item()
    # both were confidently true; refinement lowers both
    assert after['people'][0, 1] < before['people'][0, 1] - 1e-3
    assert after['org'][0, 1] < before['org'][0, 1] - 1e-3


def test_refinement_lifts_an_all_off_group():
    """At-least-one (orL/atLeastL-1): an all-off group is lifted toward one true."""
    logits = {'people': torch.tensor([[2.0, -2.0]], requires_grad=True),
              'org': torch.tensor([[2.0, -2.0]], requires_grad=True)}
    factor = Factor('at_least_one', [
        Literal('people', 1, torch.tensor([0])),
        Literal('org', 1, torch.tensor([0])),
    ])
    before = _probs(logits)
    ref = ConstraintRefinement({'people': 2, 'org': 2}, steps=5, step_size=5.0)
    ref.eval()
    after = _probs(ref(logits, [factor]))

    assert factor.violation(after).item() < factor.violation(before).item()
    highest_before = max(before['people'][0, 1], before['org'][0, 1])
    highest_after = max(after['people'][0, 1], after['org'][0, 1])
    assert highest_after > highest_before + 1e-3


# --------------------------------------------------------------------------- #
# The other constraint shapes: factor building + coverage
# --------------------------------------------------------------------------- #

def _shapes_graph():
    """A graph exercising every supported shape plus two that must be skipped."""
    from domiknows.graph import Graph, Concept, Relation
    from domiknows.graph.logicalConstrain import (
        ifL, andL, nandL, orL, atMostL, atLeastL, exactL)

    Graph.clear(); Concept.clear(); Relation.clear()
    with Graph('r4_shapes') as graph:
        ent = Concept(name='ent')
        people = ent(name='people'); org = ent(name='org')
        person = ent(name='person'); loc = ent(name='loc')
        pair = Concept(name='pair')
        (a1, a2) = pair.has_a(arg1=ent, arg2=ent)
        wf = pair(name='wf'); li = pair(name='li')

        named = {
            'typing': ifL(wf('x', 'y'), andL(people('x'), org('y'))),
            'coimpl': ifL(people('x'), person('x')),
            'coimpl_and': ifL(people('x'), andL(person('x'), loc('x'))),
            'rel2rel': ifL(wf('x', 'y'), li('x', 'y')),
            'nand': nandL(people('x'), org('x')),
            'atmost1': atMostL(people('x'), org('x'), person('x')),
            'atleast1': atLeastL(people('x'), loc('x')),
            'exact1': exactL(people('x'), org('x'), limit=1),
            'orl': orL(people('x'), loc('x')),
            # must be skipped — no faithful local factor:
            'atmost2': atMostL(people('x'), org('x'), person('x'), limit=2),
            'conj_ante': ifL(andL(people('x'), org('x')), loc('x')),
        }
    return graph, {lc.lcName: tag for tag, lc in named.items()}


def _fake_relational_datanode(n_dest=3):
    """Fake entity + pair datanodes and a matching find function."""
    class _DN:
        def __init__(self, name):
            self.name = name
            self.relationLinks = {}

    entities = [_DN(f'e{i}') for i in range(n_dest)]
    pairs = []
    for s in range(n_dest):
        for d in range(n_dest):
            p = _DN(f'p{s}{d}')
            p.relationLinks = {'arg1': [entities[s]], 'arg2': [entities[d]]}
            pairs.append(p)

    ent_names = {'ent', 'people', 'org', 'person', 'loc'}

    class _Root:
        def findRootConceptOrRelation(self, name):
            return 'ENT' if name in ent_names else 'PAIR'

    def fake_find(dn, root):
        return entities if root == 'ENT' else pairs

    return _Root(), fake_find, entities, pairs


def test_build_factors_covers_all_supported_shapes(monkeypatch):
    """Every supported shape yields a factor of the right kind; the rest skip."""
    import domiknows.graph.candidates as candidates
    from domiknows.solver.compiled.grounding import ProbabilityStore

    graph, tag_of = _shapes_graph()
    root, fake_find, entities, pairs = _fake_relational_datanode(3)
    monkeypatch.setattr(candidates, 'findDatanodesForRootConcept', fake_find)

    factors, skipped = build_constraint_factors(root, graph)
    kinds = {tag_of.get(f.name, f.name): f.kind for f in factors}

    assert kinds == {
        'typing': 'implication',
        'coimpl': 'implication',
        'coimpl_and': 'implication',
        'rel2rel': 'implication',
        'nand': 'exclusion',
        'atmost1': 'exclusion',
        'atleast1': 'at_least_one',
        'exact1': 'exactly_one',
        'orl': 'at_least_one',
    }
    # the two unfaithful shapes are reported, not silently dropped or mis-built
    assert {tag_of.get(s, s) for s in skipped} == {'atmost2', 'conj_ante'}

    # co-grounded literals get identity edges; the typing rule gets the join.
    by_tag = {tag_of.get(f.name, f.name): f for f in factors}
    src, dst = ProbabilityStore.decode_relation_rows(len(pairs), 3)
    identity = list(range(len(entities)))

    coimpl = by_tag['coimpl']
    assert [lit.node_index.tolist() for lit in coimpl.literals] == [identity, identity]

    typing = by_tag['typing']
    assert typing.literals[1].node_index.tolist() == src.tolist()   # people ← arg1
    assert typing.literals[2].node_index.tolist() == dst.tolist()   # org    ← arg2

    rel2rel = by_tag['rel2rel']
    pair_identity = list(range(len(pairs)))
    assert [lit.node_index.tolist() for lit in rel2rel.literals] == [
        pair_identity, pair_identity]


def test_build_typing_factors_is_a_backwards_compatible_alias():
    assert build_typing_factors is build_constraint_factors


if __name__ == '__main__':
    pytest.main([__file__, '-s'])
