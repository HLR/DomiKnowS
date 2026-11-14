"""Executor capability overlay; no task identity, object target or goal templates."""
from dataclasses import dataclass
from collections import Counter


@dataclass(frozen=True)
class State:
    base:object
    phase:str='skill'
    action:str=''
    held:int|None=None
    operations:int=0
    ended:bool=False
    @property
    def grammar(self):return self.base.grammar
    @property
    def semantic(self):return self.base.semantic


class ExecutorContract:
    def __init__(self,base,vocabulary,public,capabilities,max_operations=12):
        if capabilities!={'pick':'target_entity_name','place':'target_container_name'}:
            raise ValueError('this state contract covers the declared pick/place interface only')
        self.base_dfa=base;self.vocabulary=vocabulary;self.max_operations=max_operations
        if (not public['entities'] or len(set(public['entities']))!=len(public['entities'])
            or len(public['entities'])>vocabulary.max_entities or not set(public['movable'])<=set(public['entities'])
            or max_operations<2):raise ValueError('invalid observed entity/capability inventory')
        self.count=len(public['entities']);self.movable={public['entities'].index(n) for n in public['movable']}
        self.skills={vocabulary.label_for_token('skill:'+s):s for s in capabilities}
        self.arguments={s:vocabulary.label_for_token('arg:'+k) for s,k in capabilities.items()}
        self.pointers={vocabulary.label_for_token('obj:'+str(i)):i for i in range(self.count)}
        self.eos=vocabulary.eos_label;self.start_state=State(base.start_state)

    def reason(self,state,label):
        if state is None or state.ended:return 'executor_finished'
        if state.phase=='skill':
            if label==self.eos:
                return None if state.held is None and state.operations else 'executor_finish_requires_complete_operation'
            action=self.skills.get(label)
            if action is None:return 'executor_unsupported_skill'
            if action=='pick' and (state.held is not None or state.operations+2>self.max_operations):return 'executor_pick_precondition'
            if action=='place' and (state.held is None or state.operations+1>self.max_operations):return 'executor_place_precondition'
            return None
        if state.phase=='argument':return None if label==self.arguments[state.action] else 'executor_argument_schema'
        pointer=self.pointers.get(label)
        if pointer is None:return 'executor_pointer_range'
        if state.action=='pick' and pointer not in self.movable:return 'executor_pick_static_fixture'
        if state.action=='place' and pointer==state.held:return 'executor_anti_self_placement'
        return None

    def _advance(self,state,label,base):
        if label==self.eos:return State(base,operations=state.operations,ended=True)
        if state.phase=='skill':return State(base,'argument',self.skills[label],state.held,state.operations)
        if state.phase=='argument':return State(base,'pointer',state.action,state.held,state.operations)
        return State(base,held=self.pointers[label] if state.action=='pick' else None,operations=state.operations+1)

    @staticmethod
    def remaining(state):
        if state.ended:return 0
        if state.phase=='skill':return 4 if state.held is not None else (1 if state.operations else 7)
        return (6 if state.phase=='argument' else 5) if state.action=='pick' else (3 if state.phase=='argument' else 2)

    def allowed_tokens(self,state,remaining_steps=None):
        if state is None or state.ended:return frozenset()
        allowed=set()
        for label in self.base_dfa.allowed_tokens(state.base,remaining_steps=remaining_steps):
            if self.reason(state,label) is not None:continue
            # remaining_steps is the budget AFTER the candidate emission.
            next_state=self._advance(state,label,state.base)
            if remaining_steps is None or self.remaining(next_state)<=remaining_steps:allowed.add(int(label))
        return frozenset(allowed)

    def step(self,state,label):
        if self.reason(state,label) is not None:return None
        base=self.base_dfa.step(state.base,label)
        if base is None:return None
        return self._advance(state,label,base)

    def is_accepting(self,state):
        return state is not None and state.ended and self.base_dfa.is_accepting(state.base)

    def trace(self,labels):
        from ....generalization.shared_constraints import rule_trace
        result=rule_trace(self.base_dfa,labels);counts=Counter();events=[];state=self.start_state
        for index,label in enumerate(labels):
            for candidate in self.base_dfa.allowed_tokens(state.base):
                reason=self.reason(state,candidate)
                if reason:
                    counts[reason]+=1;events.append({'step':index,'candidate_label':int(candidate),'rejected_by':[reason]})
            state=self.step(state,label)
            if state is None:break
        result['executor_rejections']=dict(counts);result['executor_events']=events
        return result
