"""Stopping and DFA reachability use one after-candidate budget convention."""
import pytest
import torch

from domiknows.generation import DFA, PredicateConstraintProgram, PredicateMonitorDFA, ProductiveDFA, SemanticTransition
from domiknows.generation.dfa.conditioning import ContextualDFA
from domiknows.generation.dfa.runtime import ComposedRuntimeDFA
from domiknows.generation.dfa.stop_policy import (
    DecodeProgress, StopPolicy, remaining_steps_for, remaining_tokens_for,
    should_stop_decoding, should_stop_on_token,
)
from domiknows.generation.dfa.decoder import constrained_label_greedy_decode
from domiknows.generation.dfa.vocabulary import TokenVocabulary


def test_shared_budget_helpers_distinguish_dfa_budget_from_inclusive_hmm_horizon():
    policy = StopPolicy(max_steps=3)
    assert [remaining_steps_for(policy, i) for i in range(3)] == [2, 1, 0]
    assert [remaining_tokens_for(policy, i) for i in range(3)] == [3, 2, 1]
    assert remaining_steps_for(StopPolicy(), 7) is None
    assert remaining_tokens_for(StopPolicy(), 7) is None
    progress = DecodeProgress(3, 0.0, None, 0)
    assert should_stop_decoding(policy, progress)
    assert should_stop_on_token(policy, eos_emitted=True, accepted=True)
    assert not should_stop_on_token(policy, eos_emitted=True, accepted=False)


class TransparentAdapter:
    start_state = 0
    def step(self, state, symbol):
        return SemanticTransition(state + 1)
    def is_accepting(self, state):
        return True
    def acceptance_events(self, state):
        return ()


def grammar():
    # 1,0 completes in two tokens; the preferred 2,1,0 needs three.
    return DFA(frozenset({0, 1, 2, 3}), frozenset({0, 1, 2}),
               {(0, 1): 1, (0, 2): 2, (2, 1): 1, (1, 0): 3}, 0, frozenset({3}))


@pytest.mark.parametrize("wrap", [lambda dfa: dfa, lambda dfa: ComposedRuntimeDFA(dfa),
                                      lambda dfa: ContextualDFA(dfa, (), {}),
                                      lambda dfa: PredicateMonitorDFA(dfa, PredicateConstraintProgram(), TransparentAdapter()),
                                      lambda dfa: ProductiveDFA(dfa)])
def test_all_dfa_views_allow_final_token_with_zero_additional_steps(wrap):
    dfa = wrap(grammar())
    assert dfa.allowed_tokens(dfa.start_state, remaining_steps=1) == {1}
    state = dfa.step(dfa.start_state, 1)
    assert dfa.allowed_tokens(state, remaining_steps=0) == {0}
    assert dfa.is_accepting(dfa.step(state, 0))


@pytest.mark.parametrize("monitored", [False, True])
def test_shared_decoder_cannot_select_a_route_longer_than_stop_policy(monitored):
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>")
    class Head:
        label_to_token_id = (0, 1, 2, None)
        def next_label_logits(self, input_ids):
            return torch.tensor([0.0, 1.0, 100.0, -10.0])
        def token_id_for_label(self, label):
            token_id = self.label_to_token_id[label]
            if token_id is None:
                raise ValueError("label is not emittable")
            return token_id
    dfa = grammar()
    if monitored:
        dfa = PredicateMonitorDFA(dfa, PredicateConstraintProgram(), TransparentAdapter())
    result = constrained_label_greedy_decode(Head(), [1], vocab, dfa, stop_policy=StopPolicy(max_steps=2))
    assert result.labels == [1, 0]
    assert result.accepted
