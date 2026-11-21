"""Lossless JSON spellings projected into existing DomiKnowS-compiled rules.

This module is a serializer/lexer. Every semantic acceptance decision is
delegated to CompiledLanguage and the shared graph declarations, never a
handwritten task, capability, or object-selection rule.
"""
from functools import lru_cache
from itertools import product,combinations,permutations
import json

from ..native_baseline import ProjectedSharedDFA,CompiledLanguage
from ...generalization.executor_policy import bind_plan_policy

SURFACE_VERSION='normalized-public-alias-json-v1'


class AliasJSON(ProjectedSharedDFA):
    def __init__(self,*args,entities,**kwargs):
        super().__init__(*args,**kwargs)
        aliases_by_text={name:i for i,name in enumerate(entities) if isinstance(name,str)}
        aliases_by_text.update({str(i):i for i in range(len(entities))})
        aliases_by_text.update({'numbered object '+str(i):i for i in range(len(entities))})
        if any(not a.isascii() for a in aliases_by_text):raise ValueError('public anonymous aliases must be ASCII')
        self.operations=[{}];self.ends={};self.possible=[set()]
        self.in_string=[False];self.whitespace_boundary=[False];nodes=[set()]
        self.actions=[];self.action_tokens=[];self.end_action={};self.action_index={}
        self.skill_actions={skill:[] for skill in self.skill_arguments}
        optional=dict(self.language.vocabulary.optional_skill_arguments)
        for skill,required in self.skill_arguments.items():
            choices=optional.get(skill,())
            for size in range(len(choices)+1):
                for included in combinations(choices,size):
                    roles=(*required,*included)
                    for pointers in product(range(self.language.vocabulary.max_entities),repeat=len(roles)):
                        action=len(self.actions);self.actions.append((skill,pointers));self.skill_actions[skill].append(action)
                        self.action_tokens.append((f'skill:{skill}',)+tuple(token for role,pointer in zip(roles,pointers)
                            for token in (f'arg:{role}',f'obj:{pointer}')))
                        values=[(pointer,*(text for text,index in aliases_by_text.items() if index==pointer)) for pointer in pointers]
                        for spelling in product(*values):
                            entries=tuple(zip(roles,spelling))
                            for ordered in permutations(entries):
                                params=dict(ordered)
                                for value in ({'name':skill,'params':params},{'params':params,'name':skill}):
                                    node=0;quoted=False
                                    self.possible[node].add(skill);nodes[node].add(action)
                                    for byte in json.dumps(value,separators=(',',':')).encode():
                                        if byte==34:quoted=not quoted
                                        if byte not in self.operations[node]:
                                            self.operations[node][byte]=len(self.operations)
                                            self.operations.append({});self.possible.append(set());nodes.append(set())
                                            self.in_string.append(quoted)
                                            self.whitespace_boundary.append(not quoted and byte in b'{}:,"0123456789')
                                        node=self.operations[node][byte];self.possible[node].add(skill);nodes[node].add(action)
                                    self.ends[node]=skill;self.end_action[node]=action
        self.node_actions=[tuple(sorted(v)) for v in nodes]

    def byte_step(self,state,byte):
        # The original integer-only lexer has no literal spaces inside strings.
        # Reuse its decoded-character path rather than treating alias spaces as
        # insignificant JSON whitespace. This changes serialization only.
        if state[2]=='operation' and self.in_string[state[3]] and not state[4].startswith('escape') and byte==32:
            return super().byte_step((*state[:4],'escape_u002'),ord('0'))
        return super().byte_step(state,byte)


class AliasSurfaceDFA:
    """Token projection, including the balanced fences accepted by parse_plan."""
    def __init__(self,inner):
        self.inner=inner;self.eos_label=inner.eos_label
        self.start_state=(inner.start_state,'start',0)
        # JSON aliases are anonymous ASCII, but parse_plan accepts arbitrary
        # UTF-8 text on the opening fence's language/header line. Preserve the
        # complete tokenizer byte inventory for that lexical state too.
        self.tokens=[{}];self.token_ends={}
        for label,data in inner.token_bytes.items():
            if not data:continue
            node=0
            for byte in data:
                if byte not in self.tokens[node]:
                    self.tokens[node][byte]=len(self.tokens);self.tokens.append({})
                node=self.tokens[node][byte]
            self.token_ends.setdefault(node,[]).append(label)

    def byte_step(self,state,byte):
        core,mode,position=state
        if mode=='start':
            if byte in b' \t\r\n':return state
            if byte==96:return core,'ticks',1
            following=self.inner.byte_step(core,byte)
            return (following,'bare',0) if following is not None else None
        if mode=='ticks':return (core,'header',0) if byte==96 and position==2 else (core,'ticks',position+1) if byte==96 else None
        if mode=='header':return (core,'fenced',0) if byte==10 else state
        if mode in ('bare','fenced'):
            if mode=='fenced' and core[2]=='closed' and byte==10:return core,'close_wait',0
            following=self.inner.byte_step(core,byte)
            return (following,mode,0) if following is not None else None
        if mode=='close_wait':
            if byte in b' \t\r\n':return state
            return (core,'close_ticks',1) if byte==96 else None
        if mode=='close_ticks':
            if byte!=96:return None
            return core,('fence_closed' if position==2 else 'close_ticks'),position+1
        if mode=='fence_closed':return state if byte in b' \t\r\n' else None
        return None

    @lru_cache(maxsize=None)
    def transitions(self,state):
        if state is None or state[1]=='ended':return ()
        result=[]
        def visit(node,current):
            result.extend((label,current) for label in self.token_ends.get(node,()))
            for byte,child in self.tokens[node].items():
                following=self.byte_step(current,byte)
                if following is not None:visit(child,following)
        visit(0,state)
        if (state[1]=='bare' and state[0][2]=='closed') or state[1]=='fence_closed':
            result.append((self.eos_label,(state[0],'ended',0)))
        return tuple(result)

    def allowed_tokens(self,state,remaining_steps=None):
        return {label for label,target in self.transitions(state) if remaining_steps!=0 or self.is_accepting(target)}
    def step(self,state,label):return dict(self.transitions(state)).get(int(label))
    def is_accepting(self,state):return state is not None and state[1]=='ended'
    def accepts(self,labels):
        state=self.start_state
        for label in labels:
            state=self.step(state,label)
            if state is None:return False
        return self.is_accepting(state)


def surface_decoder(token_bytes,eos_id,public,vocabulary,max_operations=12, *,
                    policy_name='shared-verified', executor_profile=None, constraint_bindings=None):
    context = constraint_bindings if constraint_bindings is not None else {
        'entity_table': public['entities'], 'generation_static_fixture_ids': [
            i for i, n in enumerate(public['entities']) if n not in public['movable']]}
    policy=bind_plan_policy(policy_name,vocabulary,max_operations,context,executor_profile)
    return AliasSurfaceDFA(AliasJSON(token_bytes,eos_id,entities=public['entities'],
        entity_count=len(public['entities']),max_operations=max_operations,grammar_profile='world',
        language=CompiledLanguage(vocabulary,policy)))
