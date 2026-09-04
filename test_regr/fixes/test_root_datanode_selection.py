"""
Regression tests for issue #406: root DataNode selection must not depend on
the iteration order of ``DataNodeBuilder.existingDns``.

The builder used to pick the "first" root from its internal list, with a
single special-case hack that rotated the ``constraint`` concept out of
position 0. That hack broke whenever the ordering shifted, which it does
because the list is rebuilt from a set. These tests pin down the new
behavior:

* constraint DNs are never picked as the primary root automatically;
* the developer can explicitly nominate a concept via
  ``primaryRootConcept``;
* the automatic selection is deterministic (same inputs -> same choice).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domiknows.graph import Concept, Graph
from domiknows.graph.dataNode import DataNode, DataNodeBuilder


def _make_builder(roots_spec):
    """
    Build a graph with the given concept names and push one DataNode per
    concept into the builder's root list.

    ``roots_spec`` is a list of ``(concept_name, instance_id)`` tuples.
    Returns ``(builder, concepts_by_name, dns)``.
    """
    names = [name for name, _ in roots_spec]
    with Graph(name='g_' + '_'.join(names)) as g:
        concepts = {name: Concept(name=name) for name in names}

    builder = DataNodeBuilder({"graph": g})
    dns = []
    for name, iid in roots_spec:
        dn = DataNode(
            myBuilder=builder,
            instanceID=iid,
            instanceValue="",
            ontologyNode=concepts[name],
        )
        dns.append(dn)
    dict.__setitem__(builder, 'dataNode', dns)
    return builder, concepts, dns


class TestAutomaticSelection:
    def test_single_root_is_returned(self):
        builder, _, dns = _make_builder([('sentence', 0)])
        assert builder.findRootDataNode(dns) is dns[0]

    def test_constraint_is_never_picked_when_other_roots_exist(self):
        # constraint listed first on purpose: old code would have rotated it
        # to the end, new code must exclude it regardless of position.
        builder, _, dns = _make_builder([('constraint', 0), ('sentence', 1)])
        picked = builder.findRootDataNode(dns)
        assert picked.ontologyNode.name == 'sentence'

    def test_constraint_returned_only_when_nothing_else_left(self):
        builder, _, dns = _make_builder([('constraint', 0)])
        # fallback path: constraint is all we have
        assert builder.findRootDataNode(dns) is dns[0]

    def test_selection_is_order_independent(self):
        """Shuffling the input list must not change the result."""
        spec = [('alpha', 0), ('beta', 1), ('gamma', 2), ('constraint', 3)]
        builder, _, dns = _make_builder(spec)

        first = builder.findRootDataNode(list(dns))
        reversed_pick = builder.findRootDataNode(list(reversed(dns)))
        shuffled_pick = builder.findRootDataNode([dns[2], dns[0], dns[3], dns[1]])

        assert first.ontologyNode.name == reversed_pick.ontologyNode.name
        assert first.ontologyNode.name == shuffled_pick.ontologyNode.name

    def test_tie_broken_alphabetically_by_concept_name(self):
        # All three concepts are equally "rare" (one DN each) and have the
        # same link profile, so the deterministic name-based tiebreaker
        # should kick in.
        builder, _, dns = _make_builder([('zeta', 2), ('alpha', 0), ('mu', 1)])
        picked = builder.findRootDataNode(dns)
        assert picked.ontologyNode.name == 'alpha'


class TestExplicitPrimaryRoot:
    def test_via_constructor_kwarg(self):
        with Graph(name='g_ctor') as g:
            a = Concept(name='alpha')
            b = Concept(name='beta')
        builder = DataNodeBuilder({"graph": g}, primaryRootConcept='beta')
        dn_a = DataNode(myBuilder=builder, instanceID=0, instanceValue="", ontologyNode=a)
        dn_b = DataNode(myBuilder=builder, instanceID=1, instanceValue="", ontologyNode=b)
        dict.__setitem__(builder, 'dataNode', [dn_a, dn_b])

        assert builder.findRootDataNode([dn_a, dn_b]) is dn_b

    def test_via_setter_accepts_concept_object(self):
        builder, concepts, dns = _make_builder([('alpha', 0), ('beta', 1)])
        builder.setPrimaryRootConcept(concepts['beta'])
        assert builder.findRootDataNode(dns) is dns[1]

    def test_call_site_override_wins(self):
        builder, concepts, dns = _make_builder([('alpha', 0), ('beta', 1)])
        builder.setPrimaryRootConcept('alpha')
        # per-call arg should override the builder-level preference
        assert builder.findRootDataNode(dns, primaryRootConcept='beta') is dns[1]

    def test_unknown_concept_falls_back_to_automatic(self):
        builder, _, dns = _make_builder([('alpha', 0), ('beta', 1)])
        picked = builder.findRootDataNode(dns, primaryRootConcept='does_not_exist')
        # Still a valid selection, and deterministic
        assert picked.ontologyNode.name == 'alpha'

    def test_constraint_can_be_forced_if_developer_really_wants_it(self):
        builder, _, dns = _make_builder([('constraint', 0), ('sentence', 1)])
        picked = builder.findRootDataNode(dns, primaryRootConcept='constraint')
        assert picked.ontologyNode.name == 'constraint'


class TestUpdateRootDataNodeListOrdering:
    """``__updateRootDataNodeList`` should push constraint roots to the tail
    regardless of where they came in."""

    def _roots(self, builder):
        return dict.__getitem__(builder, 'dataNode')

    def test_constraint_ends_up_last(self):
        with Graph(name='g_tail') as g:
            real = Concept(name='sentence')
            constraint = Concept(name='constraint')
        builder = DataNodeBuilder({"graph": g})
        dn_real = DataNode(myBuilder=builder, instanceID=0, instanceValue="", ontologyNode=real)
        dn_constraint = DataNode(myBuilder=builder, instanceID=1, instanceValue="", ontologyNode=constraint)

        # Feed them in "wrong" order: constraint first.
        builder._DataNodeBuilder__updateRootDataNodeList([dn_constraint, dn_real])

        roots = self._roots(builder)
        assert roots[0] is dn_real
        assert roots[-1] is dn_constraint
