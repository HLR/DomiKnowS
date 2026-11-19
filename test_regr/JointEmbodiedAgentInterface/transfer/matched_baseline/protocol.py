"""Public input contract and lossless output-ID normalization, not constraints."""
from contextlib import contextmanager
from copy import deepcopy
import json

VERSION = 'matched-libero-public-core-v1'
MODELS = ('base', 'A', 'initialization', 'B')
POLICIES = ('none', 'shared-verified')


def prompt_core(public, vocabulary):
    ids = {name:i for i,name in enumerate(public['entities'])}
    if len(ids) != len(public['entities']):raise ValueError('ambiguous entity IDs')
    relations = []
    for predicate,source,destination in public['initial_relations']:
        if source not in ids or destination not in ids:raise ValueError('unbound observation relation')
        relations.append([predicate,ids[source],ids[destination]])
    return ('Return only a JSON array of robot operations, without markdown or explanation. '
        'Each operation has keys "name" and "params". '
        '"params" is a JSON object mapping each required argument name to an integer entity ID. '
        'The numeric tags in the images identify the entities. '
        'pick acquires an object; place releases the held object at a destination.\n'
        'Skills and required arguments: '+json.dumps(dict(vocabulary.skill_arguments))+'\n'
        'Entities: '+json.dumps({i:'numbered object '+str(i) for i in range(len(ids))})+'\n'
        'Initial scene relations (integer entity IDs): '+json.dumps(relations)+'\n'
        'Observed static fixture IDs: '+json.dumps([i for name,i in ids.items() if name not in public['movable']])+'\n'
        'Instruction: '+public['instruction'])


def role_prompt(core, roles=None):
    if roles is None:return core
    def value(role):return 'unknown' if roles[role] is None else str(roles[role])
    return core+'\nPredicted primary visual roles (uncertain): source ID '+value('source')+'; destination ID '+value('destination')+'. Verify against the images.'


def chat(processor, text, image_count):
    content = [{'type':'image'} for _ in range(image_count)]+[{'type':'text','text':text}]
    return processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,
        add_generation_prompt=True,enable_thinking=False)


@contextmanager
def common_prompt(planner):
    """Keep the normal causal label alignment; replace only the public prefix."""
    old_prompt,old_chat = planner._prompt_base,planner._chat
    def base(domain,context):
        if domain=='vlabench' and '_matched_core' in context:
            return role_prompt(context['_matched_core'],context.get('_matched_roles'))
        return old_prompt(domain,context)
    planner._prompt_base=base
    planner._chat=lambda text,count:chat(planner.processor,text,count)
    try:yield
    finally:planner._prompt_base,planner._chat=old_prompt,old_chat


def normalize_ids(plan, entities):
    """Exact public aliases only. Never select, reorder, remove, or repair actions."""
    result=deepcopy(plan);changes=[];aliases={name:i for i,name in enumerate(entities)}
    aliases.update({str(i):i for i in range(len(entities))})
    aliases.update({'numbered object '+str(i):i for i in range(len(entities))})
    for index,operation in enumerate(result):
        for role,value in operation.get('params',{}).items():
            if isinstance(value,str) and value in aliases:
                pointer=aliases[value]
                changes.append({'operation':index,'argument':role,'from':value,'to':pointer})
                operation['params'][role]=pointer
    return result,changes


def raw_format_valid(plan,vocabulary,entity_count):
    """Serialization diagnostic only; executable validity is scored separately."""
    if not isinstance(plan,list) or not plan:return False
    for operation in plan:
        if not isinstance(operation,dict) or set(operation)!={'name','params'}:return False
        name=operation['name'];params=operation['params']
        if name not in vocabulary.skill_argument_map or not isinstance(params,dict):return False
        if set(params)!=set(vocabulary.skill_argument_map[name]):return False
        if any(type(p) is not int or not 0<=p<entity_count for p in params.values()):return False
    return True
