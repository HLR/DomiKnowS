"""Semantic counterexamples for cached and parallel startup validation."""
import multiprocessing as mp
import pytest
import torch
from types import SimpleNamespace
from .audits import SourceAudits
from ..protocol import read


class Vocabulary:
    eos_label=3
    def token_for_label(self,i):return ('skill:pick','obj:0','obj:1','eos')[i]
    def label_for_token(self,t):return ('skill:pick','obj:0','obj:1','eos').index(t)


class DFA:
    start_state=0
    def __init__(self,pointer):self.pointer=pointer
    def allowed_tokens(self,state):return ({0},{self.pointer},{3},set())[state]
    def step(self,state,label):return state+1 if label in self.allowed_tokens(state) else None
    def is_accepting(self,state):return state==3


class Runtime:
    vlabench_vocabulary=Vocabulary()
    def dfa_for(self,domain,context):
        assert 'target_plan_labels' not in context
        return DFA(1+context['entity_table'].index('apple'))


def item(order=('apple','bowl'),pointer=1):
    return {'instruction':'pick the apple','entity_table':order,
            'target_plan_labels':torch.tensor([0,pointer,3])}


def test_reference_proofs_resume_and_reject_changed_targets(tmp_path,monkeypatch):
    from . import audits
    checker=SourceAudits(tmp_path,{'checkpoint':'one'},workers=1)
    checker.references(Runtime(),{'train':[item()]},tmp_path/'output')
    assert read(tmp_path/'output/training_constraint_audit.json')=={'examples':1,'conflicts':[]}
    monkeypatch.setattr(audits,'_reference',lambda *args:pytest.fail('cached proof recomputed'))
    checker.references(Runtime(),{'train':[item()]},tmp_path/'output')
    assert read(tmp_path/'status.json')['cached']==1
    with pytest.raises(ValueError,match='input or proof differs'):
        checker.references(Runtime(),{'train':[item(pointer=2)]},tmp_path/'output')
    with pytest.raises(ValueError,match='fingerprint differs'):
        SourceAudits(tmp_path,{'checkpoint':'different'},workers=1)


def test_references_fail_wrong_object_instead_of_binding_gold(tmp_path):
    checker=SourceAudits(tmp_path,{'code':'one'},workers=1)
    with pytest.raises(ValueError,match='source audit failed'):
        checker.references(Runtime(),{'train':[item(pointer=2)]},tmp_path/'output')
    assert not (tmp_path/'output/training_constraint_audit.json').exists()


def test_equivalence_preserves_all_allowed_tokens_and_rejects_wrong_mapping(tmp_path):
    checker=SourceAudits(tmp_path,{'code':'one'},workers=1)
    checker.equivalence(Runtime(),[item()],[item(('bowl','apple'),2)],[[1,0]],'good')
    with pytest.raises(ValueError,match='source audit failed'):
        checker.equivalence(Runtime(),[item()],[item(('bowl','apple'),1)],[[1,0]],'wrong')
    with pytest.raises(ValueError,match='coverage differs'):
        checker.equivalence(Runtime(),[item()],[],[],'missing')


@pytest.mark.skipif('fork' not in mp.get_all_start_methods(),reason='Linux training worker integration')
def test_parallel_audits_match_serial_and_join_workers(tmp_path):
    for workers in (1,2):
        checker=SourceAudits(tmp_path/str(workers),{'code':'one'},workers=workers)
        checker.references(Runtime(),{'train':[item(),item(('bowl','apple'),2)]},tmp_path/str(workers)/'out')
        assert read(tmp_path/str(workers)/'references/complete.json')['count']==2
    assert not mp.active_children()
