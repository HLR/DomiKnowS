"""A visual prior must induce the same distribution in collection and RL replay."""
import torch

from .test_stage1_dfa_loss import joint_fixture
from .test_sampling_cache import planner_for


def test_soft_object_bias_matches_replay_and_remains_differentiable(joint_fixture):
    _, runtime = joint_fixture
    planner = planner_for(runtime, returns_cache=False)
    planner.free_decoding = False
    head = planner.label_heads['vlabench']
    vocab = runtime.vlabench_vocabulary
    with torch.no_grad():
        head.weight.zero_(); head.bias.fill_(-20)
        for token in ['skill:press','arg:target_entity_name','<eos>']:
            head.bias[vocab.label_for_token(token)] = 5
        head.bias[vocab.label_for_token('obj:0')] = .2
        head.bias[vocab.label_for_token('obj:1')] = 0
    context = {'instruction':'Press a button', 'images':(), 'entity_table':('button0','button1'),
               'planning_facts':{'research':{'perception':{'constraint_policy':'soft',
                   'soft_candidates':(('press','target_entity_name',(1,)),), 'preference_weight':.5}}}}
    records=[];planner.decision_observer=records.append
    with runtime.domain_scope('vlabench'):
        with torch.no_grad():
            sequence, collected = planner.sample_labels('vlabench',context,runtime.vlabench_dfa,max_steps=6,deterministic=True)
        assert vocab.label_for_token('obj:1') in sequence
        object_decision = next(r for r in records if r['chosen']=='obj:1')
        assert max(object_decision['allowed'], key=object_decision['unmasked_logits'].get)=='obj:0'
        assert object_decision['log_bias']['obj:0']==-.5
        replay = planner.replay_labels_logprob('vlabench',context,sequence,runtime.vlabench_dfa,max_steps=6)
        assert torch.allclose(replay.detach(),collected,atol=1e-6)
        (-replay).backward()
        assert torch.isfinite(head.bias.grad).all() and head.bias.grad.abs().sum()>0
