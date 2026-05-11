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

from domiknows.generation.graph_hmm import DomiKnowSAwareHMM


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
