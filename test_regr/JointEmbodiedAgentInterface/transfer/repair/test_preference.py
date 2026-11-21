import pytest
import torch

from test_regr.VLABenchAgentInterface.plan_codec import PlanVocabulary
from test_regr.VLABenchAgentInterface.research.runtime import joint_log_bias

from ...generalization.executor_policy import EXECUTOR, SUBJECT, bind_plan_policy, pick_place_profile
from .preference import MATCH, LOWER, UNKNOWN, WEIGHT, bind_visual_evidence, source_preference


def evidence(source, scores):
    return {'phrases': {'source': source}, 'diagnostics': {'source': {
        'ranked_candidates': [{'id': i, 'score': s} for i, s in scores]}}}


def test_graph_compiles_only_a_finite_penalty_for_a_lower_visual_match():
    assert source_preference() == pytest.approx({MATCH: 0, LOWER: -WEIGHT, UNKNOWN: 0, 'pick_source': 0})


@pytest.mark.parametrize('source,scores', [(None, [(0, .7), (1, .1)]),
                                        ('block', [(0, .7)]), ('block', [(0, .7), (1, .696)])])
def test_missing_or_ambiguous_evidence_abstains(source, scores):
    assert not bind_visual_evidence(evidence(source, scores), 4).evidence


def test_joint_decode_hook_reranks_legal_pointers_without_changing_mask():
    vocabulary = PlanVocabulary.default(max_entities=4)
    label = vocabulary.label_for_token
    dfa = bind_plan_policy(EXECUTOR, vocabulary, 4,
        {'entity_table': ['a', 'b', 'c', 'fixture'], 'generation_static_fixture_ids': [3]}, pick_place_profile())
    preference = bind_visual_evidence(evidence('red block', [(0, .1), (1, .7)]), 4)
    dfa.generation_preference = preference
    state = dfa.start_state
    assert not joint_log_bias(dfa, state, dfa.allowed_tokens(state))
    for token in ('skill:pick', 'arg:' + SUBJECT):
        state = dfa.step(state, label(token))
    allowed = dfa.allowed_tokens(state)
    ids = [label(f'obj:{index}') for index in range(4)]
    assert set(allowed) == set(ids[:3])
    bias = joint_log_bias(dfa, state, allowed)
    assert bias == {ids[0]: pytest.approx(-WEIGHT)}
    logits = torch.tensor([1.1, 1.0, 0.9, 9.0])
    for i, token in enumerate(ids):
        logits[i] = logits[i] + bias.get(token, 0) if token in allowed else -torch.inf
    assert logits.argmax().item() == 1
    assert set(dfa.allowed_tokens(state)) == set(ids[:3])
    assert not preference.log_bias('place', 'target_container_name', allowed, vocabulary)

    # A visually preferred fixture cannot induce a bias against the legal choices.
    dfa.generation_preference = bind_visual_evidence(evidence('fixture', [(0, .1), (3, .7)]), 4)
    assert not joint_log_bias(dfa, state, allowed)
