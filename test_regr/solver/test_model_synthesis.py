"""R4 Phase A — model synthesis + advisory exclusivity analysis.

The payoff R4 claims for synthesis is that a joint softmax head makes an
exclusivity constraint *impossible to violate* — architecture instead of
penalty. The key test asserts exactly that, and contrasts it with independent
binary heads, which can and do violate it. The rest checks that
``analyze_exclusivity`` is a correct, advisory reporter and that
``synthesize_model`` builds usable heads.
"""

from itertools import combinations

import pytest
import torch
from torch import nn

from domiknows.program.model.synthesis import (
    analyze_exclusivity, synthesize_model, SynthesizedModel,
)


def _binary_group_graph(n=3, with_enum=False):
    """A parent with ``n`` binary siblings, pairwise-``nandL`` exclusive."""
    from domiknows.graph import Graph, Concept, Relation
    from domiknows.graph.concept import EnumConcept
    from domiknows.graph.logicalConstrain import nandL

    Graph.clear(); Concept.clear(); Relation.clear()
    with Graph('r4_synth') as graph:
        thing = Concept(name='thing')
        siblings = [thing(name=chr(ord('a') + i)) for i in range(n)]
        for x, y in combinations(siblings, 2):
            nandL(x('p'), y('p'))
        if with_enum:
            thing(name='material', ConceptClass=EnumConcept,
                  values=['metal', 'rubber', 'glass'])
    return graph


# --------------------------------------------------------------------------- #
# analyze_exclusivity (advisory)
# --------------------------------------------------------------------------- #

def test_detects_pairwise_nandL_group(capsys):
    graph = _binary_group_graph(n=3)
    report = analyze_exclusivity(graph)

    binary = [g for g in report.groups if not g.already_enum]
    assert len(binary) == 1
    group = binary[0]
    assert group.parent == 'thing'
    assert set(group.members) == {'a', 'b', 'c'}
    assert group.complete                     # every pair excluded → a real clique
    assert len(group.redundant_constraints) == 3
    # the advice is a concrete EnumConcept declaration
    suggestion = group.suggestion()
    assert 'EnumConcept' in suggestion
    for m in ('a', 'b', 'c'):
        assert f"'{m}'" in suggestion
    print('\n' + report.render())


def test_partial_group_is_flagged_not_complete():
    """A missing pair means a joint head would lose an unconstrained combination."""
    from domiknows.graph import Graph, Concept, Relation
    from domiknows.graph.logicalConstrain import nandL

    Graph.clear(); Concept.clear(); Relation.clear()
    with Graph('r4_partial') as graph:
        thing = Concept(name='thing')
        a = thing(name='a'); b = thing(name='b'); c = thing(name='c')
        nandL(a('p'), b('p'))
        nandL(b('p'), c('p'))   # a–c pair intentionally missing

    group = [g for g in analyze_exclusivity(graph).groups if not g.already_enum][0]
    assert set(group.members) == {'a', 'b', 'c'}
    assert group.complete is False


def test_enum_group_reported_as_already_joint():
    graph = _binary_group_graph(n=2, with_enum=True)
    enum_groups = [g for g in analyze_exclusivity(graph).groups if g.already_enum]
    assert any(set(g.members) == {'metal', 'rubber', 'glass'} for g in enum_groups)


def test_no_false_positive_without_exclusion():
    """Sibling concepts with no nandL are not reported as an exclusive group."""
    from domiknows.graph import Graph, Concept, Relation

    Graph.clear(); Concept.clear(); Relation.clear()
    with Graph('r4_none') as graph:
        thing = Concept(name='thing')
        thing(name='a'); thing(name='b')

    assert [g for g in analyze_exclusivity(graph).groups if not g.already_enum] == []


# --------------------------------------------------------------------------- #
# Verification 1: exclusivity is architectural under a joint head
# --------------------------------------------------------------------------- #

class _IndependentHeads(nn.Module):
    """K independent sigmoid heads — the pre-synthesis architecture."""

    def __init__(self, in_dim, k):
        super().__init__()
        self.lin = nn.Linear(in_dim, k)

    def forward(self, x):
        return torch.sigmoid(self.lin(x))


def test_joint_head_cannot_violate_exclusivity_but_independent_heads_can():
    """The joint softmax makes at-most-one-true structural; sigmoids fight for it.

    Two exact, no-training checks on random inputs:

    * joint head: probabilities sum to 1 (so the expected-count exclusivity
      residual is 0) and the argmax decode has exactly one active class — a
      softmax *cannot* put mass on two classes such that both decode true;
    * independent sigmoid heads: for random inputs several classes exceed 0.5,
      i.e. the exclusivity constraint is violated — the case synthesis removes.
    """
    torch.manual_seed(0)
    in_dim, k, n = 8, 5, 256

    model = synthesize_model(_binary_group_graph(n=2, with_enum=True),
                             input_dim=in_dim, groups={'g': k})
    x = torch.randn(n, in_dim)
    probs = model(x)['g']

    # sums to exactly one → expected active count is exactly one
    assert torch.allclose(probs.sum(dim=-1), torch.ones(n), atol=1e-5)
    # argmax one-hot decode: exactly one active class for every row, always
    one_hot = torch.zeros_like(probs)
    one_hot[torch.arange(n), probs.argmax(dim=-1)] = 1.0
    assert torch.equal(one_hot.sum(dim=-1), torch.ones(n))

    # independent heads: exclusivity is violable and, on random inputs, violated
    indep = _IndependentHeads(in_dim, k)
    active = (indep(x) > 0.5).sum(dim=-1)
    assert (active > 1).any(), 'independent heads should violate exclusivity somewhere'


# --------------------------------------------------------------------------- #
# synthesize_model wiring
# --------------------------------------------------------------------------- #

def test_synthesize_builds_one_head_per_enum_group():
    graph = _binary_group_graph(n=2, with_enum=True)
    model = synthesize_model(graph, input_dim=8, hidden_dim=16)
    assert model.group_k == {'material': 3}

    out = model(torch.randn(4, 8))
    assert set(out) == {'material'}
    assert out['material'].shape == (4, 3)
    assert torch.allclose(out['material'].sum(-1), torch.ones(4), atol=1e-5)


def test_group_module_shares_trunk_and_normalises():
    model = synthesize_model(_binary_group_graph(n=2, with_enum=True),
                             input_dim=8, groups={'g': 4})
    gm = model.group_module('g')
    # shares the trunk object → a constraint gradient on one head reaches it
    assert gm.trunk is model.trunk
    y = gm(torch.randn(6, 8))
    assert y.shape == (6, 4)
    assert torch.allclose(y.sum(-1), torch.ones(6), atol=1e-5)


def test_synthesize_requires_a_group():
    """No EnumConcept and no explicit groups → an explicit error, not an empty model."""
    from domiknows.graph import Graph, Concept, Relation
    Graph.clear(); Concept.clear(); Relation.clear()
    with Graph('r4_empty') as graph:
        Concept(name='thing')
    with pytest.raises(ValueError):
        synthesize_model(graph, input_dim=8)


if __name__ == '__main__':
    pytest.main([__file__, '-s'])
