from types import SimpleNamespace
import pytest
from .verification_protocol import snapshot_key, selected_datasets, enrich_decision, exact_policy_search, paired_interval
from .verification_freeze import assert_distinct


def test_initialization_and_legacy_epoch_interfaces_are_unambiguous():
    assert snapshot_key(snapshot='initialization') == ('initialization', 'initialization')
    assert snapshot_key(epoch=1) == snapshot_key(snapshot='epoch-1') == ('1', 'epoch-1')
    for args in ({}, {'epoch':1, 'snapshot':'initialization'}, {'epoch':0}):
        with pytest.raises(ValueError): snapshot_key(**args)
    assert selected_datasets('executor') == ('libero',)
    with pytest.raises(ValueError): selected_datasets('executor', ['vlabench'])
    assert selected_datasets('full-metadata') == ('vlabench',)


def test_decision_margin_distinguishes_masking_an_alternative_from_changing_choice():
    original = {'unmasked_logits': {'good':5., 'wrong_but_legal':4., 'invalid':-2.},
                'allowed':['good','wrong_but_legal'], 'chosen':'good'}
    r = enrich_decision(original)
    assert r['raw_top_logit_margin'] == 1 and r['best_allowed_minus_best_rejected'] == 7
    assert 'raw_top_logit_margin' not in original


def test_confirmation_duplicate_scene_or_pixels_stops_without_filtering():
    rows = [{'episode_id':str(i), 'scene':str(i), 'pixels':'p'+str(i)} for i in range(100)]
    assert_distinct(rows, [], ('scene','pixels'), 'confirmation')
    with pytest.raises(ValueError): assert_distinct(rows, [{'scene':'50','pixels':'different'}], ('scene','pixels'), 'confirmation')
    with pytest.raises(ValueError): assert_distinct(rows, [{'scene':'different','pixels':'p12'}], ('scene','pixels'), 'confirmation')
    with pytest.raises(ValueError): assert_distinct(rows[:99], [], ('scene',), 'confirmation')


def test_cluster_bootstrap_keeps_all_five_scenes_and_fixed_pairing():
    ids=[f'task{t}/state{s}' for t in range(20) for s in range(5)]
    before={i:{'success':False} for i in ids};after={i:{'success':True} for i in ids}
    r=paired_interval(before,after,ids,confidence=.975,samples=10000)
    assert r['wins']==100 and r['losses']==0 and r['task_clusters']==20
    assert r['task_cluster_interval']==[1.,1.] and r==paired_interval(before,after,ids,confidence=.975,samples=10000)
    with pytest.raises(ValueError):paired_interval(before,after,ids[:-1])


def test_unknown_exact_search_is_never_converted_to_an_allowed_set():
    from domiknows.generation.dfa.productive import SemanticSearchLimitExceeded
    search=SimpleNamespace(max_state_expansions=10000)
    def always_unknown(state,remaining_steps=None):raise SemanticSearchLimitExceeded('unknown')
    dfa=SimpleNamespace(_reachability=search,allowed_tokens=always_unknown);events=[]
    with pytest.raises(SemanticSearchLimitExceeded):
        with exact_policy_search(dfa,events,maximum=100000):dfa.allowed_tokens(0,remaining_steps=7)
    assert search.max_state_expansions==10000 and events[0]['hard_constraints_relaxed'] is False
