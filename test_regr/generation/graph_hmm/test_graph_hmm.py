"""
Tests for DomiKnowSAwareHMM - constraint-aware Hidden Markov Models.

This module tests the core HMM implementation with integrated constraint support:
- Constrained training: Baum-Welch algorithm respects constraint masks
- Inference: Scoring and Viterbi decoding with constraints
- Sampling: Generating sequences that respect constraints
- Parameter projection: Reapplying masks after parameter mutations
- Initialization: Spectral and random initialization methods
- Error handling: Clear error messages for invalid inputs

DomiKnowSAwareHMM extends standard HMM with hard constraints (masks) that ensure
probabilities of forbidden transitions/emissions are always zero.
"""
import math

import torch
import pytest

from domiknows.generation.learners import DomiKnowSAwareHMM, FiniteStateDynamicConstraint


def _learner(**kwargs):
    """
    Create a simple 2-state, 2-symbol HMM for testing.
    
    Returns a DomiKnowSAwareHMM with:
    - 2 states: "A" and "B"
    - 2 symbols: "x" and "y"
    - Default constraints: no explicit masks (all transitions/emissions allowed)
    - Seeded randomness for reproducibility
    """
    return DomiKnowSAwareHMM(
        graph=None,
        n_hidden_states=2,
        state_names=["A", "B"],
        symbols=["x", "y"],
        random_seed=13,
        **kwargs,
    )


def test_constrained_baum_welch_keeps_forbidden_probabilities_zero():
    """Test that Baum-Welch training maintains zero probabilities for forbidden transitions/emissions."""
    model = _learner(
        transition_mask=torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64),
        emission_mask=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
    )

    model.fit([["x", "y", "y"], ["x", "x", "y"]], max_iter=5)

    assert model.transition_[1, 0].item() == 0.0
    assert model.emission_[0, 1].item() == 0.0
    assert model.emission_[1, 0].item() == 0.0
    assert all(math.isfinite(value) for value in model.fit_result_.log_likelihoods)


def test_score_returns_negative_infinity_for_impossible_sequence():
    """Test that sequences violating emission masks receive -infinity score."""
    model = _learner(emission_mask=torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64))
    model.fit([["x"], ["x", "x"]], max_iter=2)
    assert model.score(["y"]) == float("-inf")


def test_viterbi_respects_masks():
    """Test that Viterbi (best path) decoding respects constraint masks."""
    model = _learner(emission_mask=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64))
    model.fit([["x", "y"], ["x", "y"]], max_iter=3)
    decoded = model.viterbi(["x", "y"])
    assert decoded.states == ("A", "B")
    assert math.isfinite(decoded.score)


def test_sampling_respects_emission_mask():
    """Test that sampled sequences only emit symbols allowed by the emission mask."""
    model = _learner(emission_mask=torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64))
    model.fit([["x"], ["x", "x"]], max_iter=2)
    generator = torch.Generator()
    generator.manual_seed(7)
    assert model.sample(5, generator=generator) == ["x", "x", "x", "x", "x"]


def test_inference_reprojects_mutated_parameters_before_use():
    """
    Test that inference operations reapply masks to parameters before use.
    
    This handles the case where internal parameters might be mutated (e.g., for
    gradient computation). The constraints should still be enforced at inference time.
    """
    model = _learner(emission_mask=torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64))
    model.fit([["x"], ["x", "x"]], max_iter=2)
    model.emission_ = torch.tensor([[0.01, 0.99], [0.01, 0.99]], dtype=torch.float64)

    assert model.score(["y"]) == float("-inf")
    assert model.viterbi(["y"]).score == float("-inf")
    generator = torch.Generator()
    generator.manual_seed(12)
    assert model.sample(3, generator=generator) == ["x", "x", "x"]


def test_to_constraint_dfa_rejects_forbidden_observation():
    """Test that the HMM can be converted to a Deterministic Finite Automaton that enforces constraints."""
    model = _learner(emission_mask=torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64))
    model.fit([["x"], ["x", "x"]], max_iter=2)
    dfa = model.to_constraint_dfa()
    assert dfa.accepts(["x", "x"])
    assert not dfa.accepts(["y"])
    assert not dfa.accepts([])


def test_to_constraint_dfa_uses_reachable_state_sets_not_argmax_path():
    """DFA export should accept if any legal hidden path emits the string."""
    init = {
        "initial": torch.tensor([0.9, 0.1], dtype=torch.float64),
        "transition": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
        "emission": torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
    }
    model = _learner().fit([["x"], ["x", "y"]], max_iter=0, init=init)

    dfa = model.to_constraint_dfa()

    assert dfa.accepts(["x"])
    assert dfa.accepts(["x", "y"])
    assert frozenset({0, 1}) in dfa.states


def test_to_constraint_dfa_refuses_arbitrary_dynamic_callback_by_default():
    """Arbitrary prefix callbacks are not exactly DFA-exportable by default."""
    model = _learner(dynamic_transition=lambda context: torch.ones((2, 2), dtype=torch.float64))
    model.fit([["x"], ["x", "y"]], max_iter=1)

    with pytest.raises(ValueError, match="dynamic_transition cannot be exported exactly"):
        model.to_constraint_dfa()


def test_to_constraint_dfa_static_mode_ignores_dynamic_callback_intentionally():
    """Static mode exports only static support when dynamic callbacks are unsupported."""
    init = {
        "initial": torch.tensor([1.0, 0.0], dtype=torch.float64),
        "transition": torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64),
        "emission": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
    }

    def dynamic_transition(context):
        if context.prefix == ("x",):
            return torch.zeros((2, 2), dtype=torch.float64)
        return None

    model = _learner(dynamic_transition=dynamic_transition).fit([["x", "y"]], max_iter=0, init=init)

    dfa = model.to_constraint_dfa(on_unsupported_dynamic="static")

    assert dfa.accepts(["x", "y"])


def test_to_constraint_dfa_refuses_soft_energy_by_default():
    """Soft transition energies are scoring biases, not hard DFA support constraints."""
    model = _learner(transition_energy=lambda context: torch.zeros((2, 2), dtype=torch.float64))
    model.fit([["x"], ["x", "y"]], max_iter=1)

    with pytest.raises(ValueError, match="transition_energy is a soft scoring bias"):
        model.to_constraint_dfa()


def test_to_constraint_dfa_with_finite_state_dynamic_constraint():
    """Finite-state dynamic constraints are exported as product DFA states."""
    init = {
        "initial": torch.tensor([1.0, 0.0], dtype=torch.float64),
        "transition": torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64),
        "emission": torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
    }
    model = _learner().fit([["x"], ["x", "y"], ["x", "x"]], max_iter=0, init=init)

    def transition_mask(dynamic_state, reachable_states, metadata):
        if dynamic_state == "seen_x":
            return torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        return torch.ones((2, 2), dtype=torch.float64)

    def advance(dynamic_state, symbol, next_reachable_states, metadata):
        return "seen_x" if symbol == "x" else dynamic_state

    finite = FiniteStateDynamicConstraint(
        start_state="start",
        transition_mask=transition_mask,
        advance=advance,
    )

    dfa = model.to_constraint_dfa(finite_state_dynamic=finite)

    assert dfa.accepts(["x"])
    assert dfa.accepts(["x", "x"])
    assert not dfa.accepts(["x", "y"])


def test_to_constraint_dfa_finite_state_accepting_callback_filters_terminals():
    """Finite-state monitors can reject terminal prefixes via is_accepting."""
    init = {
        "initial": torch.tensor([1.0, 0.0], dtype=torch.float64),
        "transition": torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64),
        "emission": torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64),
    }
    model = _learner().fit([["x"], ["x", "y"], ["y"]], max_iter=0, init=init)
    finite = FiniteStateDynamicConstraint(
        start_state="open",
        transition_mask=lambda dynamic_state, reachable_states, metadata: torch.ones((2, 2), dtype=torch.float64),
        advance=lambda dynamic_state, symbol, next_reachable_states, metadata: "done" if symbol == "y" else dynamic_state,
        is_accepting=lambda dynamic_state: dynamic_state == "done",
    )

    dfa = model.to_constraint_dfa(finite_state_dynamic=finite)

    assert not dfa.accepts(["x"])
    assert dfa.accepts(["y"])
    assert dfa.accepts(["x", "y"])


def test_to_constraint_dfa_support_threshold_removes_low_probability_support():
    """Positive thresholds intentionally approximate support by pruning small probabilities."""
    init = {
        "initial": torch.tensor([0.99, 0.01], dtype=torch.float64),
        "transition": torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64),
        "emission": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
    }
    model = _learner().fit([["x"], ["y"]], max_iter=0, init=init)

    exact = model.to_constraint_dfa(support_threshold=0.0)
    pruned = model.to_constraint_dfa(support_threshold=0.05)

    assert exact.accepts(["y"])
    assert not pruned.accepts(["y"])


def test_fit_is_deterministic_with_fixed_seed():
    """Test that training with the same random seed produces identical models."""
    seqs = [["x", "y"], ["x", "x", "y"]]
    first = _learner().fit(seqs, max_iter=4)
    second = _learner().fit(seqs, max_iter=4)
    assert torch.allclose(first.transition_, second.transition_)
    assert torch.allclose(first.emission_, second.emission_)


def test_spectral_initialization_path_still_fits():
    """Test that the spectral initialization method (using Hankel matrices) produces valid models."""
    model = _learner().fit([["x", "y"], ["x", "x", "y"]], max_iter=2, init="spectral")
    assert model.fit_result_.iterations == 2
    assert torch.allclose(model.transition_.sum(dim=1), torch.ones(2, dtype=torch.float64))
    assert torch.allclose(model.emission_.sum(dim=1), torch.ones(2, dtype=torch.float64))


def test_unknown_symbol_raises_clear_error_after_fit():
    """Test that attempting to score/sample with unknown symbols raises a clear error."""
    model = _learner().fit([["x"], ["y"]], max_iter=1)
    with pytest.raises(ValueError, match="unknown symbol"):
        model.score(["z"])


def test_constraint_weight_softens_non_binary_masks():
    """Test that constraint_weight affects non-binary mask strengths."""
    init = {
        "initial": torch.tensor([1.0, 0.0], dtype=torch.float64),
        "transition": torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float64),
        "emission": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
    }
    mask = torch.tensor([[1.0, 0.25], [1.0, 1.0]], dtype=torch.float64)

    weak = _learner(transition_mask=mask, constraint_weight=1.0).fit([["x", "y"]], max_iter=0, init=init)
    strong = _learner(transition_mask=mask, constraint_weight=2.0).fit([["x", "y"]], max_iter=0, init=init)

    assert strong.transition_[0, 1].item() < weak.transition_[0, 1].item()
    assert strong.score(["x", "y"]) < weak.score(["x", "y"])
