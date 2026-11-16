from types import SimpleNamespace
import pytest
from .verification_diagnostics import same_inputs, factual_decisions, failure_partition


def test_training_comparison_requires_identical_prompts_across_checkpoints():
    assert same_inputs([{'input_hashes':{'e':'same'}},{'input_hashes':{'e':'same'}}]) == {'e':'same'}
    with pytest.raises(ValueError,match='generation inputs'):
        same_inputs([{'input_hashes':{'e':'same'}},{'input_hashes':{'e':'changed'}}])


def test_factual_rule_join_does_not_claim_rejected_alternative_changed_top():
    rows=[{'episode_id':'e','raw_argmax':'pick','raw_argmax_allowed':True,'prefix':[]},
          {'episode_id':'e','raw_argmax':'pour','raw_argmax_allowed':False,'prefix':['pick']},
          {'episode_id':'e','rule_trace':{'executor_events':[
              {'step':0,'candidate_label':1,'rejected_by':['capability']},
              {'step':1,'candidate_label':1,'rejected_by':['capability']}]}}]
    result=list(factual_decisions(rows,SimpleNamespace(token_for_label=lambda n:['pick','pour'][n])))
    assert result[0]['raw_top_rejected_by']==[]
    assert result[0]['rejected_alternatives_by_rule']=={'pour':['capability']}
    assert result[1]['raw_top_rejected_by']==['capability']


def test_physical_failure_partition_preserves_invalid_and_wrong_object_denominators():
    plans={str(i):{'prediction':[i],'generation_error':i==0,'valid_plan':i>1,'goal_success':i>2} for i in range(5)}
    outcomes={str(i):{'prediction':[i],'success':i==4} for i in range(5)}
    counts=failure_partition(plans,outcomes)
    assert list(counts.values())==[1,1,1,1,1] and sum(counts.values())==5
    with pytest.raises(ValueError,match='denominators'):failure_partition(plans,{})


def test_native_goal_is_authoritative_when_abstract_scoring_disagrees():
    plans={'e':{'prediction':['alternative'],'generation_error':None,'valid_plan':True,'goal_success':False}}
    outcomes={'e':{'prediction':['alternative'],'success':True}}
    counts=failure_partition(plans,outcomes)
    assert counts['native_success']==1 and counts['legal_wrong_object_or_destination']==0
