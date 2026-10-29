"""Training masks must not get the gold sequence length as privileged input."""

import pytest
import torch

from domiknows.generation.dfa.core import DFA
from domiknows.generation.dfa.stop_policy import StopPolicy, remaining_steps_for
from domiknows.generation.learners.common.losses import allowed_label_sets, masked_cross_entropy


def stopping_dfa():
    # 0 = continue, 1 = EOS. Continuing requires another token to reach acceptance.
    return DFA(states=frozenset({0, 1}), alphabet=frozenset({0, 1}),
               transitions={(0, 0): 0, (0, 1): 1, (1, 1): 1},
               start_state=0, accepting_states=frozenset({1}))


def test_training_masks_match_decoding_at_every_gold_prefix():
    dfa = stopping_dfa()
    labels = [0, 1]
    policy = StopPolicy(max_steps=6)
    sets = allowed_label_sets(dfa, labels, 2, max_steps=policy.max_steps)
    state = dfa.start_state
    for step, label in enumerate(labels):
        assert sets[step] == sorted(dfa.allowed_tokens(state, remaining_steps=remaining_steps_for(policy, step)))
        state = dfa.step(state, label)
    assert sets[1] == [0, 1]
    assert allowed_label_sets(dfa, labels, 2)[1] == [1]  # legacy gold-length leak


def test_eos_gets_a_gradient_against_legal_continuations():
    targets = torch.tensor([0, 1])
    logits = torch.zeros(2, 2, requires_grad=True)
    sets = allowed_label_sets(stopping_dfa(), targets, 2, max_steps=6)
    loss, misses = masked_cross_entropy(logits, targets, sets)
    loss.backward()
    assert misses == 0
    assert logits.grad[1, 1] < 0  # learn to stop
    assert logits.grad[1, 0] > 0  # learn to prefer EOS over another action


def test_explicit_budget_keeps_scoring_and_no_lookahead_options():
    dfa = stopping_dfa()
    assert allowed_label_sets(dfa, [0, 1], 2, max_steps=6, score_steps=[1]) == {1: [0, 1]}
    assert allowed_label_sets(dfa, [0, 1], 2, max_steps=2, lookahead=False) == {0: [0, 1], 1: [0, 1]}
    with pytest.raises(ValueError, match="shorter than"):
        allowed_label_sets(dfa, [0, 1], 2, max_steps=1)
