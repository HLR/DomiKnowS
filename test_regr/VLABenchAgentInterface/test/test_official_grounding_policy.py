import json
from dataclasses import replace

import pytest
import torch

from ..object_grounding import ObjectConstraints
from ..official.grounding_policy import SoftCandidatePreference, soft_observation, verified_observation
from ..official.perception_observations import EpisodePerception, selection_graph
from ..official.selection_observations import selection_request
from ..native_planning import NativeConstraintProcessor
from .test_official_knowledge_dfa import knowledge, labels, op


def state_for(dfa, text):
    state = dfa.start_state
    for token in text.encode():
        state = dfa.step(state, token)
        assert state is not None
    return state


PREFIX = '[{"name":"pick","params":{"target_entity_name":'
PREFER_ONE = ObjectConstraints((('pick', 'target_entity_name', (1,)),))


def scores_for(dfa, state):
    result = torch.full((1, 201), -torch.inf)
    result[0, list(dfa.allowed_tokens(state))] = 0
    return result


def test_soft_candidate_can_help_close_decisions_but_cannot_forbid_a_wrong_guess():
    dfa = knowledge()
    state = state_for(dfa, PREFIX)
    preference = SoftCandidatePreference(dfa, PREFER_ONE, .5)
    result = scores_for(dfa, state)
    finite = result.isfinite().clone()
    result[0, ord('1')], result[0, ord('2')] = 1, 1.2
    preference.apply(result, state)
    assert int(result.argmax()) == ord('1')
    assert torch.equal(result.isfinite(), finite)  # identical hard legality
    result[0, ord('2')] = 2
    preference.apply(result, state)
    assert int(result.argmax()) == ord('2')  # strong planner evidence can override a detector


def test_native_processor_applies_preference_after_the_same_dfa_mask():
    dfa = knowledge()
    prefix = torch.tensor([list(PREFIX.encode())])
    logits = torch.full((1, 201), -2.0)
    logits[0, ord('1')] = 0
    logits[0, ord('2')] = .2
    plain = NativeConstraintProcessor(dfa, 0, 500)(prefix, logits)
    processor = NativeConstraintProcessor(dfa, 0, 500, object_preferences=PREFER_ONE)
    soft = processor(prefix, logits)
    assert int(plain.argmax()) == ord('2') and int(soft.argmax()) == ord('1')
    assert torch.equal(plain.isfinite(), soft.isfinite())


@pytest.mark.parametrize('spelling,expected', [('10', .5*11/12), ('11', -.5/12), ('2', -.5/12)])
def test_multi_token_object_ids_do_not_accumulate_length_penalties(spelling, expected):
    dfa = knowledge(entity_count=12)
    state = state_for(dfa, PREFIX)
    preference = SoftCandidatePreference(dfa, ObjectConstraints((('pick', 'target_entity_name', (10,)),)), .5)
    penalty = 0
    for token in (spelling + '}}').encode():
        scores = scores_for(dfa, state)
        before = scores[0, token].item()
        preference.apply(scores, state)
        penalty += scores[0, token].item() - before
        state = dfa.step(state, token)
        assert state is not None
    assert penalty == pytest.approx(expected)


def test_soft_grounding_does_not_reward_changing_the_skill():
    dfa = knowledge()
    state = state_for(dfa, '[{"name":"')
    scores = scores_for(dfa, state)
    before = scores.clone()
    SoftCandidatePreference(dfa, PREFER_ONE).apply(scores, state)
    assert torch.equal(before, scores)


def test_finite_preferences_do_not_create_a_whitespace_escape_tie():
    dfa = knowledge()
    prefix = torch.tensor([list(PREFIX.encode())])
    logits = torch.zeros(1,201)
    logits[0,ord('2')] = .2
    processor = NativeConstraintProcessor(dfa,0,500,object_preferences=PREFER_ONE)
    result = processor(prefix,logits)
    assert int(result.argmax()) == ord('1')
    assert result[0,ord('1')] > result[0,ord(' ')]


def test_soft_observation_preserves_unknown_objects_and_disables_hard_hypotheses():
    observed = EpisodePerception((1,), {'complete': True}, {}, categories=((1,'salt'),), candidates=PREFER_ONE)
    soft = soft_observation(observed)
    assert soft.visible is None and soft.grounding is None and not soft.candidates.restrictions
    assert soft.soft_candidates == PREFER_ONE
    dfa = knowledge(visible_entities=soft.visible, object_candidates=soft.candidates)
    assert dfa.accepts(labels([op('pick', 2), op('lift')]))
    empty = soft_observation(EpisodePerception((), None, {}))
    assert empty.visible is None and not empty.soft_candidates.restrictions


def test_unknown_objects_stay_legal_but_do_not_receive_positive_identity_preferences():
    observed = EpisodePerception((1,2), None, {}, categories=((1,'salt'),),
        candidates=ObjectConstraints((('pick','target_entity_name',(1,2)),)))
    soft = soft_observation(observed)
    assert soft.soft_candidates.restrictions == (('pick','target_entity_name',(1,)),)
    assert not soft.candidates.restrictions and soft.visible is None


def test_ordinal_preference_abstains_without_an_exhaustive_observed_group():
    observed = EpisodePerception((1,2), None, {'selection_request': {'kind':'value'}, 'uncertain_inventory':True},
        cards=((1,(14,'hearts')),(2,(10,'spades'))),
        candidates=ObjectConstraints((('pick','target_entity_name',(2,)),)))
    assert not soft_observation(observed).soft_candidates.restrictions


def observation(visible=(1, 2), cards=(), categories=(), boxes=None):
    return EpisodePerception(visible, None, {'uncertain_inventory': False, 'fingerprint': 'test',
                                            'boxes': boxes or {}}, categories=categories, cards=cards)


def test_verifier_disagreement_abstains_instead_of_injecting_a_card_answer():
    a = observation(cards=((1, (12, 'hearts')), (2, (4, 'hearts'))))
    b = observation(cards=((1, (4, 'hearts')), (2, (12, 'hearts'))))
    result = verified_observation(a, b, 'Pick the poker queen of hearts', selection_graph())
    assert not result.cards and not result.candidates.restrictions


def test_verifier_agreement_resolves_named_card_from_pixels_and_instruction_only():
    a = observation(cards=((1, (12, 'hearts')), (2, (4, 'hearts'))))
    b = replace(a, audit={**a.audit, 'fingerprint': 'second-model'})
    result = soft_observation(verified_observation(a, b, 'Pick the poker queen of hearts', selection_graph()))
    assert result.soft_candidates.restrictions == (('pick', 'target_entity_name', (1,)),)
    assert not result.candidates.restrictions


def test_box_union_abstains_when_two_readings_disagree_on_left_right_order():
    a = observation(categories=((1, 'book'), (2, 'book')),
                    boxes={'1': (.1, .1, .2, .2), '2': (.6, .1, .7, .2)})
    b = replace(a, audit={**a.audit, 'boxes': {'1': (.6, .1, .7, .2), '2': (.1, .1, .2, .2)}})
    result = verified_observation(a, b, 'Pick the 1st book from left to right', selection_graph())
    assert not result.candidates.restrictions


def test_missing_verifier_facts_cannot_supply_candidates():
    result = verified_observation(observation(cards=((1, (12, 'hearts')),)),
                                  EpisodePerception(None, None, {'status': 'unavailable'}),
                                  'Pick the poker queen of hearts', selection_graph())
    assert result.visible is None and not result.candidates.restrictions


@pytest.mark.parametrize('rank', ['0', '1', '15', '99'])
def test_unsupported_named_card_ranks_abstain(rank):
    assert selection_request(f'Pick the poker {rank} of hearts') is None


@pytest.mark.parametrize('weight', [-1, float('nan'), float('inf'), 3])
def test_invalid_soft_weights_are_rejected(weight):
    with pytest.raises(ValueError, match='weight'):
        SoftCandidatePreference(knowledge(), PREFER_ONE, weight)


def test_joint_enrichment_preserves_soft_candidates_without_hard_visibility(tmp_path):
    from PIL import Image
    from ..dataset import PlanningExample
    from ..research.config import _PERCEPTION_POLICY
    from ..research.perception import PerceptionConfig, enrich_examples
    class Model:
        config = PerceptionConfig(model_id='test-frozen')
        def infer(self, raw, numbered, instruction):
            assert instruction == 'Pick the salt'
            return {'objects': [{'id': i, 'confidence': .95, 'category': name,
                                 'bbox': [.1+i*.3, .1, .2+i*.3, .2], 'states': {'visible': True}}
                                for i,name in enumerate(('sugar', 'salt'))],
                    'requests': [{'action': 'pick', 'role': 'object'}], 'hypotheses': [], 'complete': False}
    paths = [tmp_path/'raw.png', tmp_path/'numbered.png']
    for p in paths:Image.new('RGB', (8,8)).save(p)
    e = PlanningExample('e', 'Pick the salt', (), (paths[0],), (paths[1],), None,
                        entities=('sugar', 'salt'))
    token = _PERCEPTION_POLICY.set({'policy':'soft', 'weight':.5, 'verifier_model':None})
    try:
        enriched = enrich_examples([e], Model())[0]
        changed = enrich_examples([replace(e, operation_sequence=({'name':'pick','params':{'target_entity_name':0}},))], Model())[0]
        assert enriched.visible_entities is None and enriched.object_candidates is None
        observed = enriched.planning_facts['research']['perception']
        assert observed['soft_candidates'] == (('pick', 'target_entity_name', (1,)),)
        assert enriched.planning_facts == changed.planning_facts  # references never ground candidates
    finally:
        _PERCEPTION_POLICY.reset(token)


def test_joint_bias_changes_only_object_ranking_not_legal_choices():
    from ..world_graph import build_vlabench_world_graph
    from ..research.runtime import joint_log_bias
    from .test_official_knowledge_level import policy
    world = build_vlabench_world_graph('soft_joint_bias', knowledge_level='official')
    vocab, _, dfa = policy(world, {'planning_facts': {'research': {'perception': {
        'constraint_policy': 'soft', 'soft_candidates': (('pick','target_entity_name',(1,)),), 'preference_weight': .5}}}})
    state = dfa.start_state
    for token in ['skill:pick','arg:target_entity_name']:
        state = dfa.step(state, vocab.label_for_token(token))
    allowed = dfa.allowed_tokens(state, remaining_steps=20)
    assert {vocab.token_for_label(x) for x in allowed} == {'obj:0','obj:1','obj:2','obj:3'}
    bias = joint_log_bias(dfa,state,allowed)
    assert bias[vocab.label_for_token('obj:0')] == -.5
    assert vocab.label_for_token('obj:1') not in bias


def test_checkpoint_resume_rejects_changed_perception_policy_but_keeps_legacy_compatibility():
    from types import SimpleNamespace
    from dataclasses import asdict
    from ..research.config import _PERCEPTION_OPTIONS, _PERCEPTION_POLICY, ResearchConfig
    from ..research.perception import PerceptionConfig
    from ..research.checkpoint import checkpoint_metadata, restore_metadata
    options = _PERCEPTION_OPTIONS.set(asdict(PerceptionConfig(model_id='frozen')))
    policy = _PERCEPTION_POLICY.set({'policy':'legacy','weight':.5,'verifier_model':None})
    try:
        world = SimpleNamespace(research_config=ResearchConfig())
        planner = SimpleNamespace()
        payload = checkpoint_metadata(world,planner)
        payload['research_configuration'].pop('perception_evidence_policy')  # old checkpoints
        restore_metadata(payload,world,planner,training=True)
        _PERCEPTION_POLICY.set({'policy':'soft','weight':.5,'verifier_model':None})
        with pytest.raises(ValueError, match='evidence policy'):
            restore_metadata(payload,world,planner,training=True)
        restore_metadata(payload,world,planner,training=False)
        assert payload['research_evaluation_override']
    finally:
        _PERCEPTION_OPTIONS.reset(options)
        _PERCEPTION_POLICY.reset(policy)


def test_native_cli_routes_soft_policy_and_records_it_without_changing_prompt(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from .. import main
    from ..native_planning import FrozenNativePlanner
    from ..official import official_protocol as protocol
    from ..research.config import _PERCEPTION_OPTIONS
    from ..research.perception import PerceptionConfig
    from dataclasses import asdict
    from .test_official_protocol import make_episode
    make_episode(tmp_path/'data','select_poker',1,'Pick the queen of hearts',[op('pick',1),op('lift')])
    args = main.build_parser().parse_args(['evaluate-planner','--protocol','official','--knowledge','official',
        '--planning-dir',str(tmp_path/'data'),'--planner-model','frozen','--official-output-dir',str(tmp_path/'run')])
    planner = SimpleNamespace(model=torch.nn.Linear(1,1),processor=object())
    monkeypatch.setattr(FrozenNativePlanner,'load',lambda *a,**k:planner)
    captured={}
    def generator(*a,**kwargs):captured['decoder']=kwargs;return lambda _: 'unused'
    def run(*a,**kwargs):captured['run']=kwargs;return {'test':'done'}
    monkeypatch.setattr(protocol,'make_grammar_dfa_generator',generator)
    monkeypatch.setattr(protocol,'run_official_protocol',run)
    token = _PERCEPTION_OPTIONS.set(asdict(PerceptionConfig(model_id='frozen')))
    try:main.command_evaluate_official(args)
    finally:_PERCEPTION_OPTIONS.reset(token)
    assert captured['decoder']['perception_policy']=='soft'
    assert captured['run']['perception']['constraint_policy']=='soft'
    assert captured['run']['perception']['preference_weight']==.5
    assert 'image-perception-soft' in captured['run']['decoding']
    assert not any(p.requires_grad for p in planner.model.parameters())


@pytest.mark.parametrize('weight,confidence', [(0.8,.85),(.5,.7)])
def test_joint_soft_tuning_requires_independent_calibration(weight,confidence):
    from ..research.config import apply_research_arguments
    from .test_official_knowledge_level import research_args
    args=research_args(knowledge='official',official_perception_policy='soft',
                       official_perception_weight=weight,perception_confidence=confidence)
    with pytest.raises(ValueError,match='independent.*calibration'):
        apply_research_arguments(args)


def test_soft_policy_cannot_silently_reintroduce_hard_grounding():
    from ..official.official_protocol import make_grammar_dfa_generator
    with pytest.raises(ValueError,match='hard research grounding'):
        make_grammar_dfa_generator(object(),perception_policy='soft',grounding_policy='joint')


def test_same_model_verification_is_rejected_before_generation():
    from types import SimpleNamespace
    from ..official.official_protocol import make_grammar_dfa_generator
    model=SimpleNamespace(config=SimpleNamespace(model_id='same'))
    with pytest.raises(ValueError,match='different frozen model'):
        make_grammar_dfa_generator(object(),perception=model,perception_verifier=model,perception_policy='soft')
