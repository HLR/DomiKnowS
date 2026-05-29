"""
Tests for GraphSpectralAutomaton - spectral learning with graph constraints.

This module tests automata learning using spectral methods with integrated constraint support:
- Spectral learning: Learning from positive examples using Hankel matrices
- Constraint masks: Enforcing transition/emission constraints during learning
- Symbol operators: Learning transition operators for each symbol
- Scoring and sampling: Computing probabilities and generating sequences
- Optional DFA: Filtering allowed strings based on external constraints
- Hankel reconstruction: Verifying learned models match training data

Spectral learning provides theoretically grounded automata learning that can learn
from positive examples with guaranteed convergence properties.
"""
import math

import pytest
import torch

from domiknows.generation.dfa import DFA
from domiknows.generation.learners import GraphSpectralAutomaton, sequence_has_legal_path


def test_sequence_has_legal_path_respects_transition_and_emission_masks():
    """Test that the sequence validation respects both transition and emission constraints."""
    transition_mask = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    emission_mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)

    assert sequence_has_legal_path([0, 1], transition_mask, emission_mask)
    assert not sequence_has_legal_path([1, 0], transition_mask, emission_mask)


def test_graph_spectral_hankel_zeroes_graph_invalid_strings():
    """
    Test that invalid strings (according to constraints) are zeroed in the Hankel matrix.
    
    The Hankel matrix records input-output behavior. Entries for invalid strings should be
    zero to indicate they are impossible under the constraints.
    """
    automaton = GraphSpectralAutomaton(
        symbols=["x", "y"],
        transition_mask=torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64),
        emission_mask=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
    )
    automaton.fit(
        [["x"], ["x", "y"], ["y", "y"], ["y", "x"]],
        prefixes=[(), ("x",), ("y",)],
        suffixes=[(), ("x",), ("y",)],
        rank=2,
    )

    hankel = automaton.build_hankel()
    assert hankel[automaton.prefixes.index(("y",)), automaton.suffixes.index(("x",))].item() == 0.0
    assert hankel[automaton.prefixes.index(("x",)), automaton.suffixes.index(("y",))].item() > 0.0
    assert automaton.fit_result_.constrained_query_count < automaton.fit_result_.total_query_count


def test_graph_spectral_learns_symbol_operators_and_scores():
    """Test that spectral learning successfully learns operators and can score sequences."""
    automaton = GraphSpectralAutomaton(symbols=["x", "y"])
    automaton.fit(
        [["x"], ["y"], ["x", "y"], ["x", "y"]],
        prefixes=[(), ("x",), ("y",)],
        suffixes=[(), ("x",), ("y",)],
        rank=2,
    )

    assert automaton.operator("x").shape == (2, 2)
    assert automaton.operator("y").shape == (2, 2)
    assert automaton.prefix_state(["x"]).shape == (2,)
    assert math.isfinite(automaton.score(["x", "y"]))
    assert automaton.allowed_symbols(["x"]) == ("x", "y")
    assert automaton.fit_result_.diagnostics["retained_singular_mass"] <= 1.0


def test_graph_spectral_optional_dfa_filters_hankel_entries():
    """
    Test that an optional external DFA constraint filters the Hankel matrix entries.
    
    The DFA specifies which strings should be queried during learning. Invalid strings
    (according to the DFA) are not queried, reducing unnecessary computation.
    """
    dfa = DFA(
        states=frozenset({"start", "seen_x", "dead"}),
        alphabet=frozenset({"x", "y"}),
        transitions={
            ("start", "x"): "seen_x",
            ("start", "y"): "dead",
            ("seen_x", "x"): "seen_x",
            ("seen_x", "y"): "seen_x",
            ("dead", "x"): "dead",
            ("dead", "y"): "dead",
        },
        start_state="start",
        accepting_states=frozenset({"seen_x"}),
        dead_states=frozenset({"dead"}),
    )
    automaton = GraphSpectralAutomaton(symbols=["x", "y"], dfa=dfa)
    automaton.fit(
        [["x"], ["y"], ["x", "y"]],
        prefixes=[(), ("x",), ("y",)],
        suffixes=[(), ("x",), ("y",)],
        rank=1,
    )

    hankel = automaton.build_hankel()
    assert hankel[automaton.prefixes.index(("y",)), automaton.suffixes.index(())].item() == 0.0
    assert automaton.allowed_symbols(()) == ("x",)


def test_graph_spectral_hard_score_filters_invalid_reconstruction_leakage():
    """Hard spectral inference must use legality checks, not only WFA scores."""
    dfa = DFA(
        states=frozenset({"start", "seen_x", "dead"}),
        alphabet=frozenset({"x", "y"}),
        transitions={
            ("start", "x"): "seen_x",
            ("start", "y"): "dead",
            ("seen_x", "x"): "seen_x",
            ("seen_x", "y"): "seen_x",
            ("dead", "x"): "dead",
            ("dead", "y"): "dead",
        },
        start_state="start",
        accepting_states=frozenset({"seen_x"}),
        dead_states=frozenset({"dead"}),
    )
    automaton = GraphSpectralAutomaton(symbols=["x", "y"], dfa=dfa)
    automaton.fit(
        [["x"], ["x", "y"], ["x", "x"], ["y"]],
        prefixes=[(), ("x",), ("y",)],
        suffixes=[(), ("x",), ("y",)],
        rank=1,
    )

    assert not automaton.is_sequence_allowed(["y"])
    assert automaton.score(["y"], enforce_constraints=True) == 0.0
    assert automaton.hard_score(["y"]) == 0.0
    assert automaton.score(["x"], enforce_constraints=True) == automaton.score(["x"])


def test_graph_spectral_rejects_invalid_basis_and_rank():
    """Test that invalid prefix/suffix basis or rank values are rejected with clear errors."""
    automaton = GraphSpectralAutomaton(symbols=["x"])
    with pytest.raises(ValueError, match="empty prefix"):
        automaton.fit([["x"]], prefixes=[("x",)], suffixes=[()], rank=1)
    with pytest.raises(ValueError, match="rank"):
        automaton.fit([["x"]], prefixes=[()], suffixes=[()], rank=2)
