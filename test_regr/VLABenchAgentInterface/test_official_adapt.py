import itertools

import pytest
import torch

from .official_adapt import adapt_emission, episode_seed, expected_counts


def tiny_hmm(states=3, vocabulary=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    initial = torch.softmax(torch.randn(states, generator=generator, dtype=torch.double), -1)
    transition = torch.softmax(torch.randn(states, states, generator=generator, dtype=torch.double), -1)
    emission = torch.softmax(torch.randn(states, vocabulary, generator=generator, dtype=torch.double), -1)
    return initial, transition, emission


def brute_force_counts(initial, transition, emission, sequences):
    states, vocabulary = emission.shape
    counts = torch.zeros(states, vocabulary, dtype=torch.double)
    for sequence in sequences:
        total = 0.0
        partial = torch.zeros(states, vocabulary, dtype=torch.double)
        for path in itertools.product(range(states), repeat=len(sequence)):
            weight = float(initial[path[0]])
            for step, state in enumerate(path):
                weight *= float(emission[state, sequence[step]])
                if step + 1 < len(path):
                    weight *= float(transition[state, path[step + 1]])
            total += weight
            for step, state in enumerate(path):
                partial[state, sequence[step]] += weight
        counts += partial / total
    return counts


def test_expected_counts_equal_enumeration_over_hidden_paths():
    initial, transition, emission = tiny_hmm()
    sequences = [[0, 1, 2, 1], [3, 3, 0], [5]]
    tokens, counts = expected_counts(initial, transition, emission, sequences)
    dense = torch.zeros_like(emission)
    dense[:, tokens] = counts
    assert torch.allclose(dense, brute_force_counts(initial, transition, emission, sequences), atol=1e-10)
    assert float(counts.sum()) == pytest.approx(sum(len(sequence) for sequence in sequences))


def test_adaptation_keeps_rows_normalised_and_moves_towards_the_samples():
    initial, transition, emission = tiny_hmm(seed=3)
    sequences = [[1, 1, 1, 1, 1]] * 4
    adapted = adapt_emission(initial, transition, emission, sequences, 0.3)
    assert torch.allclose(adapted.sum(-1), torch.ones(3, dtype=torch.double))
    assert (adapted[:, 1] > emission[:, 1]).all()
    assert (adapted[:, 0] < emission[:, 0]).all()
    assert torch.equal(adapt_emission(initial, transition, emission, sequences, 0.0), emission)
    assert torch.equal(adapt_emission(initial, transition, emission, [], 0.3), emission)


def test_full_mixing_reproduces_the_expected_count_estimate():
    initial, transition, emission = tiny_hmm(seed=5)
    sequences = [[0, 2, 2], [4, 0]]
    tokens, counts = expected_counts(initial, transition, emission, sequences)
    adapted = adapt_emission(initial, transition, emission, sequences, 1.0)
    seen = counts.sum(-1) > 0
    assert torch.allclose(adapted[seen][:, tokens], counts[seen] / counts[seen].sum(-1, keepdim=True))


def test_episode_seed_is_reproducible_and_depends_on_the_prompt():
    first = [["text", "pick the apple"], ["image", "a.png"]]
    assert episode_seed(first, 7) == episode_seed(first, 7)
    assert episode_seed(first, 7) != episode_seed([["text", "pick the pear"]], 7)
    assert episode_seed(first, 7) != episode_seed(first, 8)
