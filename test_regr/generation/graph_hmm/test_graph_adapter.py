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
from domiknows.graph.logicalConstrain import V, andL, atLeastAL, atMostAL, atMostL, equivalenceL, ifL, notL, orL
from domiknows.generation.learners import (
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


def test_graph_adapter_compiles_reversed_variable_binding_transition_lc():
    """Destination/source variable bindings should affect the correct axis."""
    Graph.clear()
    Concept.clear()
    Relation.clear()
    with Graph("hmm_lc_reverse_vars") as graph:
        a = Concept(name="A")
        b = Concept(name="B")
        Concept(name="C")
        ifL(a("y"), notL(b("x")))

    adapter = DomiKnowSGraphAdapter(graph, n_hidden_states=3, state_names=["A", "B", "C"], symbols=["x"])
    mask = adapter.allowed_transition_mask()

    assert mask[1, 0].item() == 0.0
    assert mask[0, 1].item() == 1.0
    assert mask[2, 0].item() == 1.0


def test_graph_adapter_compiles_relation_path_endpoint_transition_lc():
    """Single-hop path endpoints are treated as next/destination bindings."""
    fake_if_l = type("ifL", (), {})
    lc = fake_if_l()
    lc.e = [
        (None, "A", None, 1),
        V(name="x"),
        (None, "B", None, 1),
        V(name="z", v=("x", "next", "y")),
    ]
    adapter = DomiKnowSGraphAdapter(
        graph=None,
        constraints=[lc],
        n_hidden_states=3,
        state_names=["A", "B", "C"],
        symbols=["x"],
    )

    mask = adapter.allowed_transition_mask()

    assert mask[0, 0].item() == 0.0
    assert mask[0, 1].item() == 1.0
    assert mask[0, 2].item() == 0.0


def test_graph_adapter_compiles_local_count_and_equivalence_lcs():
    """Local count and equivalence operators compile over finite state pairs."""
    Graph.clear()
    Concept.clear()
    Relation.clear()
    with Graph("hmm_lc_count_equiv") as graph:
        a = Concept(name="A")
        b = Concept(name="B")
        c = Concept(name="C")
        andL(
            atMostL(a("x"), b("y"), limit=1),
            equivalenceL(c("x"), c("y")),
        )

    adapter = DomiKnowSGraphAdapter(graph, n_hidden_states=3, state_names=["A", "B", "C"], symbols=["x"])
    mask = adapter.allowed_transition_mask()

    assert mask[0, 1].item() == 0.0
    assert mask[2, 2].item() == 1.0
    assert mask[2, 0].item() == 0.0
    assert mask[0, 2].item() == 0.0


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


def test_graph_adapter_compiles_accumulated_zero_forbid_as_static_masks():
    """atMostAL(..., 0) is a safe static global forbiddance fragment."""
    Graph.clear()
    Concept.clear()
    Relation.clear()
    with Graph("hmm_lc_accum_zero") as graph:
        bad_state = Concept(name="BAD")
        bad_token = Concept(name="bad_token")
        atMostAL(bad_state("x"), limit=0)
        atMostAL(bad_token("y"), limit=0)

    adapter = DomiKnowSGraphAdapter(
        graph,
        n_hidden_states=2,
        state_names=["OK", "BAD"],
        symbols=["good_token", "bad_token"],
    )

    transition_mask = adapter.allowed_transition_mask()
    emission_mask = adapter.emission_type_mask()

    assert torch.all(transition_mask[1, :] == 0)
    assert torch.all(transition_mask[:, 1] == 0)
    assert torch.all(emission_mask[1, :] == 0)
    assert torch.all(emission_mask[:, 1] == 0)


def test_graph_adapter_registers_nonlocal_accumulated_lc_for_dfa_export():
    """Non-zero accumulated counts are not approximated as local matrices."""
    Graph.clear()
    Concept.clear()
    Relation.clear()
    with Graph("hmm_lc_accum_nonlocal") as graph:
        required = Concept(name="REQUIRED")
        atLeastAL(required("x"), limit=1)

    adapter = DomiKnowSGraphAdapter(graph, n_hidden_states=2, state_names=["A", "B"], symbols=["x"])
    mask = adapter.allowed_transition_mask()

    assert torch.allclose(mask, torch.ones((2, 2), dtype=torch.float64))
    assert any("DFA-export constraint spec" in message for message in adapter.report.applied)


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
