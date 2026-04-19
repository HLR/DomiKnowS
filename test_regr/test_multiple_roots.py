"""
Tests for handling multiple roots in DataNodeBuilder (issue #305).
Covers isRootUnique, needsBatchRootDN, addBatchRootDN, and _is_structural.
"""
import pytest
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.dataNode import DataNode, DataNodeBuilder


# ---------- helpers ----------

def _make_graph_and_concepts(n_concepts=2, name_prefix="type"):
    """Build a simple graph with n_concepts concepts using context manager."""
    with Graph(name='test_graph') as g:
        concepts = []
        for i in range(n_concepts):
            c = Concept(name=f'{name_prefix}_{i}')
            concepts.append(c)
    return g, concepts


def _make_builder_with_roots(concepts, ids=None):
    """
    Create a DataNodeBuilder and manually stuff DataNodes into its root list.
    Each concept gets one DataNode.
    """
    builder = DataNodeBuilder({"graph": concepts[0].sup})
    dns = []
    for i, c in enumerate(concepts):
        dn = DataNode(
            myBuilder=builder,
            instanceID=ids[i] if ids else i,
            instanceValue="",
            ontologyNode=c,
        )
        dns.append(dn)
    # shove them into the builder's root list directly
    dict.__setitem__(builder, 'dataNode', dns)
    return builder, dns


# ---------- _is_structural ----------

class TestIsStructural:
    def test_regular_concept_not_structural(self):
        g, (c,) = _make_graph_and_concepts(1)
        builder = DataNodeBuilder({"graph": g})
        dn = DataNode(myBuilder=builder, instanceID=0, instanceValue="", ontologyNode=c)
        assert builder._is_structural(dn) is False

    def test_constraint_concept_is_structural(self):
        with Graph(name='g_constraint') as g:
            c = Concept(name='constraint')
        builder = DataNodeBuilder({"graph": g})
        dn = DataNode(myBuilder=builder, instanceID=0, instanceValue="", ontologyNode=c)
        assert builder._is_structural(dn) is True


# ---------- isRootUnique / needsBatchRootDN ----------

class TestIsRootUnique:
    def test_empty_builder_returns_false(self):
        g, _ = _make_graph_and_concepts(1)
        builder = DataNodeBuilder({"graph": g})
        # no dataNode key at all
        assert builder.isRootUnique() is False
        assert builder.needsBatchRootDN() is True

    def test_single_root_returns_true(self):
        g, (c,) = _make_graph_and_concepts(1)
        builder, _ = _make_builder_with_roots([c])
        assert builder.isRootUnique() is True
        assert builder.needsBatchRootDN() is False

    def test_two_same_type_roots_returns_false(self):
        g, (c,) = _make_graph_and_concepts(1)
        builder = DataNodeBuilder({"graph": g})
        dn1 = DataNode(myBuilder=builder, instanceID=0, instanceValue="", ontologyNode=c)
        dn2 = DataNode(myBuilder=builder, instanceID=1, instanceValue="", ontologyNode=c)
        dict.__setitem__(builder, 'dataNode', [dn1, dn2])
        # two roots of same type -> not unique
        assert builder.isRootUnique() is False
        assert builder.needsBatchRootDN() is True

    def test_two_diff_type_roots_returns_false(self):
        g, concepts = _make_graph_and_concepts(2)
        builder, _ = _make_builder_with_roots(concepts)
        assert builder.isRootUnique() is False

    def test_structural_plus_one_real_is_unique(self):
        """If only one non-structural root remains, it's unique."""
        with Graph(name='g_str') as g:
            real = Concept(name='sentence')
            constraint = Concept(name='constraint')
        builder, _ = _make_builder_with_roots([real, constraint])
        assert builder.isRootUnique() is True
        assert builder.needsBatchRootDN() is False


# ---------- addBatchRootDN ----------

class TestAddBatchRootDN:
    def test_raises_on_empty_builder(self):
        g, _ = _make_graph_and_concepts(1)
        builder = DataNodeBuilder({"graph": g})
        with pytest.raises(ValueError, match="no DataNode"):
            builder.addBatchRootDN()

    def test_noop_on_single_root(self):
        g, (c,) = _make_graph_and_concepts(1)
        builder, dns = _make_builder_with_roots([c])
        builder.addBatchRootDN()
        # still just 1 root
        roots = dict.__getitem__(builder, 'dataNode')
        assert len(roots) == 1
        assert roots[0].getOntologyNode().name == c.name

    def test_wraps_same_type_roots(self):
        g, (c,) = _make_graph_and_concepts(1)
        builder = DataNodeBuilder({"graph": g})
        dn1 = DataNode(myBuilder=builder, instanceID=0, instanceValue="", ontologyNode=c)
        dn2 = DataNode(myBuilder=builder, instanceID=1, instanceValue="", ontologyNode=c)
        dict.__setitem__(builder, 'dataNode', [dn1, dn2])

        builder.addBatchRootDN()
        roots = dict.__getitem__(builder, 'dataNode')
        assert len(roots) == 1
        assert roots[0].getOntologyNode().name == 'batch'

    def test_wraps_diff_type_roots(self):
        """This is the key scenario from issue #305 — mixed types should still get wrapped."""
        g, concepts = _make_graph_and_concepts(2)
        builder, dns = _make_builder_with_roots(concepts)

        builder.addBatchRootDN()
        roots = dict.__getitem__(builder, 'dataNode')
        assert len(roots) == 1
        assert roots[0].getOntologyNode().name == 'batch'

    def test_batch_root_has_children(self):
        g, concepts = _make_graph_and_concepts(3)
        builder, dns = _make_builder_with_roots(concepts)

        builder.addBatchRootDN()
        batchRoot = dict.__getitem__(builder, 'dataNode')[0]
        # batch root should contain all original roots as children
        children = batchRoot.getChildDataNodes()
        child_ids = {c.instanceID for c in children}
        assert child_ids == {0, 1, 2}

    def test_idempotent_double_call(self):
        """Calling addBatchRootDN twice shouldn't create nested batches."""
        g, concepts = _make_graph_and_concepts(2)
        builder, _ = _make_builder_with_roots(concepts)

        builder.addBatchRootDN()
        builder.addBatchRootDN()  # second call — should be a noop now
        roots = dict.__getitem__(builder, 'dataNode')
        assert len(roots) == 1
        assert roots[0].getOntologyNode().name == 'batch'


# ---------- createBatchRootDN vs addBatchRootDN comparison ----------

class TestCreateVsAdd:
    def test_createBatchRootDN_skips_mixed_types(self):
        """Original createBatchRootDN should NOT wrap when types differ."""
        g, concepts = _make_graph_and_concepts(2)
        builder, dns = _make_builder_with_roots(concepts)

        builder.createBatchRootDN()
        roots = dict.__getitem__(builder, 'dataNode')
        # should still have 2 roots — createBatchRootDN bails on mixed types
        assert len(roots) == 2

    def test_addBatchRootDN_wraps_mixed_types(self):
        """addBatchRootDN should wrap even when types differ."""
        g, concepts = _make_graph_and_concepts(2)
        builder, dns = _make_builder_with_roots(concepts)

        builder.addBatchRootDN()
        roots = dict.__getitem__(builder, 'dataNode')
        assert len(roots) == 1
        assert roots[0].getOntologyNode().name == 'batch'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# ---------- 10k stress / fuzz test ----------

# We generate 10000 random scenarios to hammer at the root-handling logic.
# Each scenario picks a random number of concept types (1-5), random number
# of DataNodes per type (1-4), optionally includes a 'constraint' structural
# node, and then runs isRootUnique / needsBatchRootDN / addBatchRootDN and
# verifies invariants hold.

def _generate_stress_params(n=10000, seed=305):
    """Generate n random test configs as (n_types, dns_per_type, include_constraint)."""
    rng = random.Random(seed)
    params = []
    for i in range(n):
        n_types = rng.randint(1, 5)
        dns_per_type = rng.randint(1, 4)
        include_constraint = rng.choice([True, False])
        params.append((i, n_types, dns_per_type, include_constraint))
    return params

_STRESS_PARAMS = _generate_stress_params(10000)


class TestStress10k:
    """Parametrized stress tests — 10 000 random root configurations."""

    @pytest.mark.parametrize("case_id,n_types,dns_per_type,incl_constraint", _STRESS_PARAMS)
    def test_isRootUnique_invariants(self, case_id, n_types, dns_per_type, incl_constraint):
        with Graph(name=f'stress_{case_id}') as g:
            concepts = [Concept(name=f'c{t}') for t in range(n_types)]
            if incl_constraint:
                constraint_c = Concept(name='constraint')

        builder = DataNodeBuilder({"graph": g})
        dns = []
        idx = 0
        for c in concepts:
            for _ in range(dns_per_type):
                dn = DataNode(myBuilder=builder, instanceID=idx, instanceValue="", ontologyNode=c)
                dns.append(dn)
                idx += 1
        if incl_constraint:
            dn = DataNode(myBuilder=builder, instanceID=idx, instanceValue="", ontologyNode=constraint_c)
            dns.append(dn)
            idx += 1

        dict.__setitem__(builder, 'dataNode', dns)
        total_real_roots = len(dns) - (1 if incl_constraint else 0)

        unique = builder.isRootUnique()
        needs = builder.needsBatchRootDN()

        # invariant: unique and needs are always opposites
        assert unique != needs, f"case {case_id}: isRootUnique={unique} but needsBatchRootDN={needs}"

        if n_types == 1 and dns_per_type == 1:
            # single real concept root (maybe + constraint)
            assert unique is True, f"case {case_id}: single type single dn should be unique"
        elif n_types == 1 and dns_per_type > 1:
            # multiple roots of same type -> not unique
            assert unique is False, f"case {case_id}: multiple same-type roots not unique"
        elif n_types > 1:
            # multiple different types -> not unique
            assert unique is False, f"case {case_id}: mixed types not unique"

    @pytest.mark.parametrize("case_id,n_types,dns_per_type,incl_constraint", _STRESS_PARAMS)
    def test_addBatchRootDN_always_produces_single_root(self, case_id, n_types, dns_per_type, incl_constraint):
        with Graph(name=f'stress_add_{case_id}') as g:
            concepts = [Concept(name=f'c{t}') for t in range(n_types)]
            if incl_constraint:
                constraint_c = Concept(name='constraint')

        builder = DataNodeBuilder({"graph": g})
        dns = []
        idx = 0
        for c in concepts:
            for _ in range(dns_per_type):
                dn = DataNode(myBuilder=builder, instanceID=idx, instanceValue="", ontologyNode=c)
                dns.append(dn)
                idx += 1
        if incl_constraint:
            dn = DataNode(myBuilder=builder, instanceID=idx, instanceValue="", ontologyNode=constraint_c)
            dns.append(dn)
            idx += 1

        dict.__setitem__(builder, 'dataNode', dns)
        original_count = len(dns)

        builder.addBatchRootDN()
        roots_after = dict.__getitem__(builder, 'dataNode')

        if original_count <= 1:
            # noop expected
            assert len(roots_after) == original_count
        else:
            # should have been wrapped into a single batch root
            assert len(roots_after) == 1, f"case {case_id}: expected 1 root after addBatchRootDN, got {len(roots_after)}"
            assert roots_after[0].getOntologyNode().name == 'batch'
            children = roots_after[0].getChildDataNodes()
            assert len(children) == original_count, f"case {case_id}: batch root should have {original_count} children, got {len(children)}"
