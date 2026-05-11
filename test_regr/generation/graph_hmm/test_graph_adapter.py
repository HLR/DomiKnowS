"""
Tests for DomiKnowSGraphAdapter - converting DomiKnows graphs to HMM constraints.

This module tests the adapter that transforms DomiKnows knowledge graphs into HMM and
automaton constraints:
- Graph extraction: Extracting concepts and relations from graphs
- Explicit constraints: Applying constraint specifications (allowed/forbidden transitions/emissions)
- Logical constraint compilation: Converting DomiKnows logical constraints to masks
- Factorized state spaces: Working with structured state representations
- Constraint reports: Documenting applied and unsupported constraints

The adapter enables domain experts to specify knowledge as DomiKnows graphs, which are then
automatically converted to precise HMM constraints.
"""
import torch

from domiknows.graph import Concept, Graph, Relation
from domiknows.graph.logicalConstrain import andL, ifL, notL, orL
from domiknows.generation.graph_hmm import (
    AllowedEmissionsSpec,
    AllowedTransitionsSpec,
    DomiKnowSGraphAdapter,
    EmissionMaskSpec,
    FactorizedStateSpace,
    ForbiddenEmissionsSpec,
    ForbiddenTransitionsSpec,
    StatePredicateTransitionSpec,
    TransitionMaskSpec,
)


def test_graph_adapter_extracts_concepts_and_explicit_relation_mask():
    """Test that the adapter correctly extracts concepts and relations from a DomiKnows graph."""
    Graph.clear()
    Concept.clear()
    Relation.clear()
    with Graph("hmm_adapter") as graph:
        Concept(name="A")
        Concept(name="B")

    adapter = DomiKnowSGraphAdapter(
        graph,
        relations=[("A", "B")],
        n_hidden_states=2,
        state_names=["A", "B"],
        symbols=["x"],
    )

    assert adapter.concepts() == ["A", "B"]
    assert adapter.relations() == ["A->B"]
    assert torch.allclose(adapter.allowed_transition_mask(), torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float64))


def test_graph_adapter_applies_explicit_constraint_specs():
    """Test that explicit constraint specifications (forbidden transitions/emissions) are applied."""
    adapter = DomiKnowSGraphAdapter(
        graph=None,
        constraints=[
            {"forbid_transition": ("S0", "S1")},
            {"forbid_emission": ("S1", "bad")},
        ],
        n_hidden_states=2,
        state_names=["S0", "S1"],
        symbols=["good", "bad"],
    )

    transition_mask = adapter.allowed_transition_mask()
    emission_mask = adapter.emission_type_mask()

    assert transition_mask[0, 1].item() == 0.0
    assert emission_mask[1, 1].item() == 0.0
    assert emission_mask[0, 1].item() == 1.0


def test_graph_adapter_normalizes_typed_constraint_specs():
    """
    Test that multiple constraint specification formats are normalized correctly.
    
    The adapter accepts various constraint formats (TransitionMaskSpec, AllowedTransitionsSpec,
    ForbiddenTransitionsSpec, etc.) and combines them into final masks.
    """
    adapter = DomiKnowSGraphAdapter(
        graph=None,
        constraints=[
            TransitionMaskSpec([[1.0, 1.0], [1.0, 0.0]], name="base-transition"),
            AllowedTransitionsSpec((("S0", "S0"), ("S0", "S1"), ("S1", "S0")), name="allowed-transition"),
            ForbiddenTransitionsSpec((("S0", "S1"),), name="forbid-transition"),
            EmissionMaskSpec([[1.0, 1.0], [1.0, 1.0]], name="base-emission"),
            AllowedEmissionsSpec((("S0", "good"), ("S1", "bad")), name="allowed-emission"),
            ForbiddenEmissionsSpec((("S1", "bad"),), name="forbid-emission"),
        ],
        n_hidden_states=2,
        state_names=["S0", "S1"],
        symbols=["good", "bad"],
    )

    transition_mask = adapter.allowed_transition_mask()
    emission_mask = adapter.emission_type_mask()

    assert torch.allclose(transition_mask, torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64))
    assert torch.allclose(emission_mask, torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64))
    assert any("base-transition" in message for message in adapter.report.applied)


def test_state_predicate_transition_spec_uses_factorized_state_space():
    """
    Test that StatePredicateTransitionSpec works with factorized state spaces.
    
    Predicates are evaluated on structured state representations where states are
    represented as dictionaries of factor values rather than flat integers.
    """
    state_space = FactorizedStateSpace.from_factors({"relation": ["holding", "on_table"]})
    adapter = DomiKnowSGraphAdapter(
        graph=None,
        constraints=[
            StatePredicateTransitionSpec(
                lambda src, dst: not (src["relation"] == "holding" and dst["relation"] == "on_table"),
                name="holding-not-on-table",
            )
        ],
        n_hidden_states=len(state_space),
        state_space=state_space,
        symbols=["x"],
    )

    mask = adapter.allowed_transition_mask()
    holding = state_space.state_id(relation="holding")
    on_table = state_space.state_id(relation="on_table")

    assert mask[holding, on_table].item() == 0.0
    assert mask[on_table, holding].item() == 1.0


def test_graph_adapter_compiles_static_if_not_transition_lc():
    """
    Test that simple if-not logical constraints are compiled to transition masks.
    
    Example: ifL(a("x"), notL(b("y"))) means "if A transitions, then B must not".
    This is compiled to constraint masks that enforce this relationship.
    """
    Graph.clear()
    Concept.clear()
    Relation.clear()
    with Graph("hmm_lc_if_not") as graph:
        a = Concept(name="A")
        b = Concept(name="B")
        Concept(name="C")
        ifL(a("x"), notL(b("y")))

    adapter = DomiKnowSGraphAdapter(graph, n_hidden_states=3, state_names=["A", "B", "C"], symbols=["x"])
    mask = adapter.allowed_transition_mask()

    assert mask[0, 1].item() == 0.0
    assert mask[0, 0].item() == 1.0
    assert mask[0, 2].item() == 1.0
    assert mask[1, 1].item() == 1.0


def test_graph_adapter_compiles_boolean_transition_lcs():
    """
    Test that complex boolean logical constraints with AND/OR/IF are compiled correctly.
    
    Example: andL(ifL(a("x"), orL(b("y"), c("y"))), ifL(c("x"), a("y")))
    Multiple constraints are combined using boolean logic.
    """
    Graph.clear()
    Concept.clear()
    Relation.clear()
    with Graph("hmm_lc_bool") as graph:
        a = Concept(name="A")
        b = Concept(name="B")
        c = Concept(name="C")
        andL(ifL(a("x"), orL(b("y"), c("y"))), ifL(c("x"), a("y")))

    adapter = DomiKnowSGraphAdapter(graph, n_hidden_states=3, state_names=["A", "B", "C"], symbols=["x"])
    mask = adapter.allowed_transition_mask()

    assert mask[0, 0].item() == 0.0
    assert mask[0, 1].item() == 1.0
    assert mask[0, 2].item() == 1.0
    assert mask[2, 0].item() == 1.0
    assert mask[2, 1].item() == 0.0


def test_graph_adapter_compiles_static_emission_typing_lc():
    """Test that logical constraints on emission types are correctly compiled to masks."""
    Graph.clear()
    Concept.clear()
    Relation.clear()
    with Graph("hmm_lc_emission") as graph:
        loc = Concept(name="LOC")
        location_token = Concept(name="location_token")
        ifL(loc("x"), location_token("y"))

    adapter = DomiKnowSGraphAdapter(
        graph,
        n_hidden_states=2,
        state_names=["LOC", "O"],
        symbols=["location_token", "other_token"],
    )

    emission_mask = adapter.emission_type_mask()

    assert emission_mask[0, 0].item() == 1.0
    assert emission_mask[0, 1].item() == 0.0
    assert emission_mask[1, 0].item() == 1.0
    assert emission_mask[1, 1].item() == 1.0


def test_graph_adapter_reports_unsupported_static_lc():
    """Test that unsupported constraint types are documented in the constraint report."""
    Graph.clear()
    Concept.clear()
    Relation.clear()
    with Graph("hmm_lc_unsupported") as graph:
        a = Concept(name="A")
        notL(a("x"))

    adapter = DomiKnowSGraphAdapter(graph, n_hidden_states=2, state_names=["A", "B"], symbols=["x"])
    mask = adapter.allowed_transition_mask()

    assert torch.allclose(mask, torch.ones((2, 2), dtype=torch.float64))
    assert any("unsupported static HMM transition logical constraint" in message for message in adapter.report.unsupported)
