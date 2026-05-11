import pytest
import torch

from domiknows.generation import (
    HMMGenerationHead,
    PromptConditionedHMMGenerationHead,
    PromptConditionedSpectralWFAGenerationHead,
    SpectralWFAGenerationHead,
    WeightedFiniteAutomaton,
    apply_hmm_transition_potential,
    apply_wfa_transition_potential,
    combine_transition_potentials,
    forbid_hmm_transition,
    penalize_hmm_transition,
    transition_potential_matrix,
    wfa_sequence_energy_loss,
    hmm_sequence_nll,
)
from domiknows.generation.automata import DiscreteHMM


def test_hmm_hard_potential_masks_and_renormalizes_transition_row():
    transition = torch.tensor([[0.3, 0.7], [0.4, 0.6]])
    potential = forbid_hmm_transition(0, 1, state_count=2)

    reweighted = apply_hmm_transition_potential(transition, potential)

    assert torch.allclose(reweighted[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(reweighted.sum(dim=-1), torch.ones(2))


def test_hmm_soft_potential_reduces_transition_without_eliminating_it():
    transition = torch.tensor([[0.3, 0.7], [0.4, 0.6]])
    potential = penalize_hmm_transition(0, 1, state_count=2, penalty=0.1)

    reweighted = apply_hmm_transition_potential(transition, potential)

    assert 0.0 < reweighted[0, 1] < transition[0, 1]
    assert torch.allclose(reweighted.sum(dim=-1), torch.ones(2))


def test_combined_transition_potentials_multiply_factors():
    first = forbid_hmm_transition(0, 1, state_count=2, strength=0.5)
    second = penalize_hmm_transition(1, 0, state_count=2, penalty=0.25)
    combined = combine_transition_potentials([first, second])

    values = combined.tensor_for(torch.ones((2, 2)))

    assert values[0, 1].item() == pytest.approx(0.5)
    assert values[1, 0].item() == pytest.approx(0.25)


def test_hmm_potential_rejects_invalid_values_and_zero_rows():
    transition = torch.tensor([[0.3, 0.7], [0.4, 0.6]])

    with pytest.raises(ValueError, match="non-negative"):
        apply_hmm_transition_potential(transition, torch.tensor([[1.0, -1.0], [1.0, 1.0]]))

    with pytest.raises(ValueError, match="all-zero"):
        apply_hmm_transition_potential(transition, torch.tensor([[0.0, 0.0], [1.0, 1.0]]))

    with pytest.raises(ValueError, match="broadcast"):
        apply_hmm_transition_potential(transition, torch.ones(3, 3))


def test_wfa_potential_multiply_and_add_preserve_signed_behavior():
    transitions = torch.tensor(
        [
            [[1.0, -2.0], [0.5, -0.5]],
            [[-1.0, 2.0], [-0.25, 0.75]],
        ]
    )
    factor = torch.tensor([[2.0, 0.5], [1.0, 3.0]])

    multiplied = apply_wfa_transition_potential(transitions, factor, mode="multiply")
    added = apply_wfa_transition_potential(transitions, factor, mode="add")

    assert multiplied[0, 0, 1] == pytest.approx(-1.0)
    assert multiplied[1, 0, 0] == pytest.approx(-2.0)
    assert added[0, 0, 1] == pytest.approx(-1.5)
    assert added[1, 0, 0] == pytest.approx(1.0)


def test_discrete_hmm_with_transition_potential_changes_log_prob_and_viterbi():
    hmm = DiscreteHMM(
        transition=[[0.2, 0.8], [0.1, 0.9]],
        emission=[[0.9, 0.1], [0.1, 0.9]],
        initial=[1.0, 0.0],
        symbols=["a", "b"],
        normalize=False,
    )
    obs = torch.tensor([[0, 1]])
    potential = forbid_hmm_transition(0, 1, state_count=2)

    base_log_prob = hmm.log_prob(obs)
    reweighted_log_prob = hmm.log_prob(obs, transition_potential=potential)
    base_path, _ = hmm.viterbi(obs)
    reweighted_path, _ = hmm.viterbi(obs, transition_potential=potential)
    reweighted_hmm = hmm.with_transition_potential(potential)

    assert reweighted_log_prob < base_log_prob
    assert base_path.tolist() == [[0, 1]]
    assert reweighted_path.tolist() == [[0, 0]]
    assert torch.allclose(reweighted_hmm.transition[0], torch.tensor([1.0, 0.0]))


def test_discrete_hmm_sample_uses_transition_potential():
    hmm = DiscreteHMM(
        transition=[[0.1, 0.9], [0.0, 1.0]],
        emission=[[1.0, 0.0], [0.0, 1.0]],
        initial=[1.0, 0.0],
        symbols=["a", "b"],
        normalize=False,
    )
    potential = forbid_hmm_transition(0, 1, state_count=2)

    sample = hmm.sample(1, 2, generator=torch.Generator().manual_seed(0), transition_potential=potential)

    assert sample.tolist() == [[0, 0]]


def test_weighted_finite_automaton_transition_potential_changes_scores():
    wfa = WeightedFiniteAutomaton(
        initial=[1.0, 0.0],
        transitions={
            "a": [[1.0, 1.0], [0.0, 0.5]],
            "b": [[0.2, -0.1], [0.4, 0.3]],
        },
        final=[1.0, -1.0],
        symbols=["a", "b"],
    )
    potential = transition_potential_matrix([[1.0, 0.0], [1.0, 1.0]])

    base = wfa.sequence_probability(("a", "b"))
    reweighted = wfa.sequence_probability(("a", "b"), transition_potential=potential)
    reweighted_wfa = wfa.with_transition_potential(potential)

    assert reweighted != pytest.approx(base)
    assert reweighted == pytest.approx(reweighted_wfa.sequence_probability(("a", "b")))


def test_hmm_head_next_logits_and_sequence_log_probs_respect_potential():
    head = HMMGenerationHead(label_count=3, state_count=2, pad_size=3, trainable=False, random_seed=3)
    potential = forbid_hmm_transition(0, 1, state_count=2)

    base_logits = head.next_label_logits(torch.tensor([0, 1]))
    reweighted_logits = head.next_label_logits(torch.tensor([0, 1]), transition_potential=potential)
    no_potential = head.sequence_log_probs(torch.tensor([1, 2, 0]))
    explicit_none = head.sequence_log_probs(torch.tensor([1, 2, 0]), transition_potential=None)
    with_potential = head.sequence_log_probs(torch.tensor([1, 2, 0]), transition_potential=potential)

    assert not torch.allclose(base_logits, reweighted_logits)
    assert torch.allclose(no_potential, explicit_none)
    assert not torch.allclose(no_potential, with_potential)
    assert torch.isfinite(hmm_sequence_nll(head, torch.tensor([1, 2, 0]), transition_potential=potential))


def test_wfa_head_next_logits_and_sequence_log_probs_respect_potential():
    head = SpectralWFAGenerationHead(label_count=3, state_count=2, pad_size=3, trainable=False, random_seed=4)
    potential = transition_potential_matrix([[1.0, 0.0], [1.0, 1.0]])

    base_logits = head.next_label_logits(torch.tensor([0, 1]))
    reweighted_logits = head.next_label_logits(torch.tensor([0, 1]), transition_potential=potential)
    no_potential = head.sequence_log_probs(torch.tensor([1, 2, 0]))
    explicit_none = head.sequence_log_probs(torch.tensor([1, 2, 0]), transition_potential=None)
    with_potential = head.sequence_log_probs(torch.tensor([1, 2, 0]), transition_potential=potential)

    assert not torch.allclose(base_logits, reweighted_logits)
    assert torch.allclose(no_potential, explicit_none)
    assert not torch.allclose(no_potential, with_potential)
    assert torch.isfinite(wfa_sequence_energy_loss(head, torch.tensor([1, 2, 0]), transition_potential=potential))


def test_prompt_conditioned_heads_apply_potential_after_prompt_dynamics():
    prompt = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([1, 2, 0])
    potential = forbid_hmm_transition(0, 1, state_count=2)
    hmm_head = PromptConditionedHMMGenerationHead(
        label_count=3,
        state_count=2,
        pad_size=3,
        dynamics_conditioning="gated",
        dynamics_expert_count=2,
        random_seed=5,
    )
    wfa_head = PromptConditionedSpectralWFAGenerationHead(
        label_count=3,
        state_count=2,
        pad_size=3,
        dynamics_conditioning="gated",
        dynamics_expert_count=2,
        random_seed=6,
    )
    wfa_potential = transition_potential_matrix([[1.0, 0.0], [1.0, 1.0]])

    hmm_base = hmm_head(None, prompt, labels)
    hmm_reweighted = hmm_head(None, prompt, labels, transition_potential=potential)
    wfa_base = wfa_head(None, prompt, labels)
    wfa_reweighted = wfa_head(None, prompt, labels, transition_potential=wfa_potential)

    assert not torch.allclose(hmm_base, hmm_reweighted)
    assert torch.allclose(
        hmm_head.prompt_transition_probs(prompt, transition_potential=potential).sum(dim=-1),
        torch.ones(2),
        atol=1e-5,
    )
    assert not torch.allclose(wfa_base, wfa_reweighted)
