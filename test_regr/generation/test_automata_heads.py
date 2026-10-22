from __future__ import annotations

import torch

from domiknows.generation.dfa._constraints import (
    eos_closure_dfa,
    required_token_dfa,
)
from domiknows.generation.dfa import product_dfa
from domiknows.generation.dfa.vocabulary import TokenVocabulary
from domiknows.generation.learners import (
    DiscreteHMM,
    HMMGenerationHead,
    SpectralWFAGenerationHead,
    WeightedFiniteAutomaton,
    allowed_mass_loss,
    hmm_sequence_nll,
    wfa_sequence_energy_loss,
)


def test_hmm_head_returns_valid_log_probs():
    head = HMMGenerationHead(label_count=4, state_count=2, pad_size=3, trainable=False)

    log_probs = head(None, torch.tensor([[9]]), torch.tensor([1, 2, 0]))

    assert log_probs.shape == (3, 4)
    assert torch.isfinite(log_probs).all()
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(3), atol=1e-5)


def test_hmm_head_exposes_batched_production_core():
    head = HMMGenerationHead(label_count=4, state_count=2, pad_size=3, trainable=True)

    batched = head.sequence_log_probs(torch.tensor([[1, 2, 0], [2, 0, 0]]), lengths=torch.tensor([3, 2]))
    core = head.production_hmm()

    assert batched.shape == (2, 3, 4)
    assert torch.isfinite(batched).all()
    assert core.symbols == (0, 1, 2, 3)
    assert torch.allclose(core.transition, head.transition_probs)


def test_hmm_head_can_wrap_existing_discrete_hmm():
    hmm = DiscreteHMM(
        transition=((0.7, 0.3), (0.2, 0.8)),
        emission=((0.6, 0.3, 0.1), (0.1, 0.4, 0.5)),
        initial=(0.8, 0.2),
        symbols=(0, 1, 2),
    )
    head = HMMGenerationHead(hmm, pad_size=2, trainable=False)

    assert head.label_count == 3
    assert head.state_count == 2
    assert head.next_label_logits(torch.tensor([0, 1])).shape == (3,)


def test_wfa_head_returns_finite_log_probs_for_signed_scores():
    wfa = WeightedFiniteAutomaton(
        initial=(1.0, -0.4),
        transitions={
            0: ((0.2, -0.1), (0.3, 0.4)),
            1: ((-0.5, 0.2), (0.1, -0.3)),
            2: ((0.4, 0.1), (-0.2, 0.5)),
        },
        final=(0.7, -0.6),
        symbols=(0, 1, 2),
    )
    head = SpectralWFAGenerationHead(wfa, pad_size=3, trainable=False)

    logits = head.next_label_logits(torch.tensor([1, 2]))
    log_probs = head(None, torch.tensor([[9]]), torch.tensor([1, 2, 0]))

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()
    assert log_probs.shape == (3, 3)
    assert torch.isfinite(log_probs).all()


def test_wfa_head_exposes_batched_production_core():
    head = SpectralWFAGenerationHead(label_count=4, state_count=2, pad_size=3, trainable=True)

    batched = head.sequence_log_probs(torch.tensor([[1, 2, 0], [2, 0, 0]]), lengths=torch.tensor([3, 2]))
    core = head.production_wfa()

    assert batched.shape == (2, 3, 4)
    assert torch.isfinite(batched).all()
    assert core.symbols == (0, 1, 2, 3)
    assert torch.allclose(core.transition_tensor, head.transitions)


def test_frozen_and_trainable_head_parameters():
    frozen = HMMGenerationHead(label_count=3, state_count=2, trainable=False)
    trainable = HMMGenerationHead(label_count=3, state_count=2, trainable=True)
    wfa = SpectralWFAGenerationHead(label_count=3, state_count=2, trainable=True)

    assert frozen.trainable_parameter_names() == []
    assert trainable.trainable_parameter_names() == [
        "initial_logits",
        "transition_logits",
        "emission_logits",
    ]
    assert wfa.trainable_parameter_names() == ["initial", "transitions", "final"]


def test_hmm_auxiliary_loss_is_finite_and_differentiable():
    head = HMMGenerationHead(label_count=4, state_count=2, pad_size=3, trainable=True)

    loss = hmm_sequence_nll(head, torch.tensor([1, 2, 0]))
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_wfa_auxiliary_loss_is_finite_and_differentiable():
    head = SpectralWFAGenerationHead(label_count=4, state_count=2, pad_size=3, trainable=True)

    loss = wfa_sequence_energy_loss(head, torch.tensor([1, 2, 0]))
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_allowed_mass_loss_is_finite_and_differentiable():
    vocab = TokenVocabulary(["<eos>", " A"], eos_token="<eos>")
    dfa = product_dfa([eos_closure_dfa(vocab), required_token_dfa(vocab, " A")])
    logits = torch.tensor(
        [
            [0.1, 3.0, -1.0],
            [2.0, 0.2, -1.0],
        ],
        requires_grad=True,
    )
    probs = torch.softmax(logits, dim=-1)

    loss = allowed_mass_loss(probs, dfa)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_allowed_mass_loss_follows_a_given_path_instead_of_the_greedy_one():
    import math
    import torch
    from domiknows.generation.dfa.core import DFA
    from domiknows.generation.learners import allowed_mass_loss

    # state 0 allows only label 0 (-> state 1); state 1 allows only label 1 (-> state 2)
    dfa = DFA(states=frozenset({0, 1, 2}), alphabet=frozenset({0, 1}),
              transitions={(0, 0): 1, (1, 1): 2, (2, 0): 2}, start_state=0, accepting_states=frozenset({2}))
    probs = torch.tensor([[0.3, 0.7], [0.9, 0.1]])
    # greedy: label 1 leaves the language at step 1, so nothing is scored afterwards
    assert math.isclose(float(allowed_mass_loss(probs, dfa)), -math.log(0.3), abs_tol=1e-4)
    # along the target 0, 1: step 2 is scored in state 1 (label 1 allowed)
    assert math.isclose(float(allowed_mass_loss(probs, dfa, path=torch.tensor([0, 1]))),
                        -(math.log(0.3) + math.log(0.1)) / 2, abs_tol=1e-4)


def test_allowed_mass_loss_scores_only_the_requested_steps():
    import math
    import torch
    from domiknows.generation.dfa.core import DFA
    from domiknows.generation.learners import allowed_mass_loss

    dfa = DFA(states=frozenset({0, 1, 2}), alphabet=frozenset({0, 1}),
              transitions={(0, 0): 1, (1, 1): 2, (2, 0): 2}, start_state=0, accepting_states=frozenset({2}))
    probs = torch.tensor([[0.3, 0.7], [0.9, 0.1], [0.5, 0.5]])
    path = torch.tensor([0, 1, 0])
    # the state still advances through the unscored steps: step 2 is scored in state 2 (label 0 allowed, 0.5)
    assert math.isclose(float(allowed_mass_loss(probs, dfa, path=path, score_steps=[2])), -math.log(0.5), abs_tol=1e-4)
    assert math.isclose(float(allowed_mass_loss(probs, dfa, path=path, score_steps=[0, 1])),
                        -(math.log(0.3) + math.log(0.1)) / 2, abs_tol=1e-4)


def test_allowed_mass_loss_without_lookahead_uses_the_labels_valid_at_the_step():
    import math
    import torch
    from domiknows.generation.dfa.core import DFA
    from domiknows.generation.learners import allowed_mass_loss

    class Counting:
        """Records the remaining-step budget that is passed; valid labels: 0 at state 0 and 1 afterwards."""
        def __init__(self):
            self.budgets = []
            self.start_state = 0
        def allowed_tokens(self, state, remaining_steps=None):
            self.budgets.append(remaining_steps)
            return {0} if state == 0 else {1}
        def step(self, state, symbol):
            return state + 1

    probs = torch.tensor([[0.3, 0.7], [0.2, 0.8]])
    with_budget, without = Counting(), Counting()
    allowed_mass_loss(probs, with_budget, path=[0, 1])
    value = allowed_mass_loss(probs, without, path=[0, 1], lookahead=False)
    assert with_budget.budgets == [1, 0] and without.budgets == [None, None]
    assert math.isclose(float(value), -(math.log(0.3) + math.log(0.8)) / 2, abs_tol=1e-4)


def test_masked_cross_entropy_restricts_the_softmax_and_falls_back_when_the_target_is_disallowed():
    import math
    import torch
    from domiknows.generation.learners import allowed_label_sets, allowed_log_mass, masked_cross_entropy

    logits = torch.log(torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8], [0.2, 0.3, 0.5]]))
    targets = torch.tensor([0, 2, 1])
    # step 0 allows {0, 1}; step 1 allows {1, 2}; step 2 has no set (full softmax)
    sets = {0: [0, 1], 1: [1, 2]}
    loss, misses = masked_cross_entropy(logits, targets, sets)
    expected = (-math.log(0.5 / 0.8) - math.log(0.8 / 0.9) - math.log(0.3)) / 3
    assert misses == 0 and math.isclose(float(loss), expected, abs_tol=1e-5)
    # a target outside its set keeps the full softmax at that step and is counted
    loss, misses = masked_cross_entropy(logits, targets, {0: [1, 2], 1: [1, 2]})
    assert misses == 1 and math.isclose(float(loss), (-math.log(0.5) - math.log(0.8 / 0.9) - math.log(0.3)) / 3, abs_tol=1e-5)
    log_mass = allowed_log_mass(torch.log_softmax(logits, dim=-1), sets)
    assert torch.allclose(log_mass.exp(), torch.tensor([0.8, 0.9]), atol=1e-5)


def test_allowed_label_sets_follow_the_path_and_respect_score_steps():
    from domiknows.generation.dfa.core import DFA
    from domiknows.generation.learners import allowed_label_sets

    dfa = DFA(states=frozenset({0, 1, 2}), alphabet=frozenset({0, 1}),
              transitions={(0, 0): 1, (1, 1): 2, (2, 0): 2}, start_state=0, accepting_states=frozenset({2}))
    assert allowed_label_sets(dfa, [0, 1, 0], 2, lookahead=False) == {0: [0], 1: [1], 2: [0]}
    assert allowed_label_sets(dfa, [0, 1, 0], 2, score_steps=[1], lookahead=False) == {1: [1]}
