import pytest
import torch

from domiknows.generation.learners.hmm.core import (
    DiscreteHMM,
    ProbabilisticAutomaton,
    all_sequences,
    baum_welch_train,
    compare_hmm_dfa,
)


def test_toy_hmm_extracts_argmax_dfa_and_compares_sequences():
    hmm = ProbabilisticAutomaton(
        transition=[[0.8, 0.2], [0.3, 0.7]],
        emission=[[0.9, 0.1], [0.2, 0.8]],
        initial=[0.9, 0.1],
        symbols=["a", "b"],
    )
    dfa = hmm.extract_argmax_dfa()

    assert dfa.accepts(["a", "b"])
    summary = compare_hmm_dfa(hmm, dfa, all_sequences(["a", "b"], max_length=2))
    assert 0.0 <= summary["precision"] <= 1.0
    assert 0.0 <= summary["recall"] <= 1.0
    assert summary["mean_hmm_probability"] > 0.0


def test_discrete_hmm_batched_forward_backward_viterbi_and_serialization(tmp_path):
    hmm = DiscreteHMM(
        transition=[[0.8, 0.2], [0.3, 0.7]],
        emission=[[0.9, 0.1], [0.2, 0.8]],
        initial=[0.9, 0.1],
        symbols=["a", "b"],
        state_names=["hot", "cold"],
        dtype=torch.float64,
    )
    observations, lengths = hmm.encode([["a", "b", "a"], ["b"]])

    factors = hmm.forward_backward(observations, lengths)
    log_probs = hmm.log_prob(observations, lengths)
    paths, scores = hmm.viterbi(observations, lengths)

    assert factors.alpha.shape == (2, 3, 2)
    assert factors.xi.shape == (2, 2, 2, 2)
    assert torch.isfinite(log_probs).all()
    assert paths.shape == observations.shape
    assert scores.shape == (2,)
    assert hmm.sequence_probability(["a", "b"]) == pytest.approx(float(torch.exp(hmm.log_prob(torch.tensor([[0, 1]])))[0]))

    hmm.save_pretrained(tmp_path)
    loaded = DiscreteHMM.from_pretrained(tmp_path, dtype=torch.float64)
    assert loaded.symbols == ("a", "b")
    assert loaded.state_names == ("hot", "cold")
    assert torch.allclose(loaded.transition, hmm.transition)


def test_baum_welch_likelihood_is_non_decreasing_and_rows_are_stochastic():
    result = baum_welch_train(
        [["a", "a", "a"], ["a", "a", "b"], ["b", "b", "b"], ["b", "b", "a"]],
        symbols=["a", "b"],
        state_count=2,
        max_iter=8,
        random_seed=7,
    )

    assert result.iterations == len(result.log_likelihoods)
    assert result.log_likelihoods
    assert all(
        after + 1e-8 >= before
        for before, after in zip(result.log_likelihoods, result.log_likelihoods[1:])
    )

    assert float(result.model.initial.sum()) == pytest.approx(1.0)
    assert torch.allclose(result.model.transition.sum(dim=-1), torch.ones(2, dtype=result.model.transition.dtype))
    assert torch.allclose(result.model.emission.sum(dim=-1), torch.ones(2, dtype=result.model.emission.dtype))


def test_baum_welch_fixed_seed_is_deterministic():
    sequences = [["a", "b", "a"], ["a", "a", "b"], ["b", "b", "a"]]
    first = baum_welch_train(sequences, ["a", "b"], 2, max_iter=5, random_seed=3)
    second = baum_welch_train(sequences, ["a", "b"], 2, max_iter=5, random_seed=3)

    assert second.log_likelihoods == first.log_likelihoods
    assert torch.allclose(second.model.initial, first.model.initial)
    assert torch.allclose(second.model.transition, first.model.transition)
    assert torch.allclose(second.model.emission, first.model.emission)


def test_baum_welch_accepts_explicit_initial_model():
    init = ProbabilisticAutomaton(
        transition=[[0.9, 0.1], [0.1, 0.9]],
        emission=[[0.8, 0.2], [0.2, 0.8]],
        initial=[0.5, 0.5],
        symbols=["a", "b"],
    )

    result = baum_welch_train(
        [["a", "a", "a"], ["a", "a", "b"], ["b", "b", "b"], ["b", "b", "a"]],
        symbols=["a", "b"],
        state_count=2,
        max_iter=5,
        init=init,
    )

    assert result.model.symbols == ("a", "b")
    assert result.model.emission[0, 0] > result.model.emission[0, 1]
    assert result.model.emission[1, 1] > result.model.emission[1, 0]


def test_trained_model_still_extracts_dfa_and_compares_with_checker():
    result = baum_welch_train(
        [["a", "a", "a"], ["a", "b", "a"], ["b", "b", "b"]],
        symbols=["a", "b"],
        state_count=2,
        max_iter=6,
        random_seed=11,
    )

    dfa = result.model.extract_argmax_dfa()
    summary = compare_hmm_dfa(result.model, dfa, all_sequences(["a", "b"], max_length=3))

    assert dfa.accepts(["a"])
    assert 0.0 <= summary["precision"] <= 1.0
    assert 0.0 <= summary["recall"] <= 1.0
    assert summary["mean_hmm_probability"] > 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sequences": []}, "sequences must not be empty"),
        ({"symbols": []}, "symbols must not be empty"),
        ({"state_count": 0}, "state_count must be at least 1"),
        ({"sequences": [[]]}, "empty sequences are not supported"),
        ({"sequences": [["a", "c"]]}, "unknown symbol 'c'"),
        ({"max_iter": 0}, "max_iter must be at least 1"),
        ({"tol": -1.0}, "tol must be non-negative"),
        ({"smoothing": -1.0}, "smoothing must be non-negative"),
    ],
)
def test_baum_welch_rejects_invalid_inputs(kwargs, message):
    args = {
        "sequences": [["a", "b"]],
        "symbols": ["a", "b"],
        "state_count": 2,
    }
    args.update(kwargs)

    with pytest.raises(ValueError, match=message):
        baum_welch_train(**args)
