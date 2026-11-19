"""Frozen native-Qwen LIBERO baseline; shared policy is compiled from DomiKnowS.

The native JSON adapter only translates bytes to canonical graph labels. It
declares no task, object, holding, or executor legality rules of its own.
"""
import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import random

from .protocol import digest, episodes, freeze, public_input, read, sha
from ..generalization.shared_constraints import bind_shared, SHARED_RULES
from test_regr.VLABenchAgentInterface.native_planning import NativePlanDFA


class CompiledLanguage:
    """Translate complete JSON actions into the existing compiled shared DFA."""
    def __init__(self, vocabulary, policy):
        self.vocabulary, self.policy = vocabulary, policy
        self.start = policy.start_state

    def advance(self, state, tokens):
        for token in tokens:
            label = self.vocabulary.label_for_token(token)
            if label not in self.policy.allowed_tokens(state):
                return None
            state = self.policy.step(state, label)
            if state is None:
                return None
        return state

    def closes(self, state):
        label = self.vocabulary.eos_label
        return (label in self.policy.allowed_tokens(state)
                and self.policy.is_accepting(self.policy.step(state, label)))


class ProjectedSharedDFA(NativePlanDFA):
    """Byte lexer for a compiled policy whose states distinguish entity IDs."""
    @lru_cache(maxsize=None)
    def allowed_actions(self, grammar):
        return frozenset(i for i in range(len(self.actions))
                         if self.next_action_state(grammar, i) is not None)

    @lru_cache(maxsize=None)
    def allowed_skills(self, grammar, count=None):
        return frozenset(self.actions[i][0] for i in self.allowed_actions(grammar))

    def byte_step(self, state, byte):
        result = super().byte_step(state, byte)
        if result is not None and result[2] == 'operation':
            # Prune incomplete JSON spellings only when the compiled policy
            # rejects every complete action that could extend that spelling.
            if not self.allowed_actions(result[1]).intersection(self.node_actions[result[3]]):
                return None
        return result


def compiled_decoder(token_bytes, eos_id, public, vocabulary, max_operations):
    context = {'entity_table': public['entities'],
               'generation_static_fixture_ids': [i for i, name in enumerate(public['entities'])
                                                  if name not in public['movable']]}
    policy = bind_shared(vocabulary, max_operations, context)
    return ProjectedSharedDFA(token_bytes, eos_id, entity_count=len(public['entities']),
        max_operations=max_operations, grammar_profile='world',
        language=CompiledLanguage(vocabulary, policy))


def prompt(public, vocabulary):
    """Public-only, anonymous inputs; identical for both decoding conditions."""
    return ('Return only a JSON array of robot operations, without markdown or explanation. '
        'Each operation has keys "name" and "params". '
        '"params" must be a JSON object (not a list), mapping each required argument name '
        'to an integer entity ID. '
        'The numeric tags in the images identify the entities. '
        'pick acquires an object; place releases the held object at a destination.\n'
        'Skills and required arguments: '+json.dumps(dict(vocabulary.skill_arguments))+'\n'
        'Entities: '+json.dumps({i: 'numbered object '+str(i) for i in range(len(public['entities']))})+'\n'
        'Initial scene relations: '+json.dumps(public['initial_relations'])+'\n'
        'Instruction: '+public['instruction'])


def parse_plan(text):
    value = text.strip()
    if value.startswith('```'):
        lines = value.splitlines()
        if len(lines) < 3 or lines[-1].strip() != '```':
            raise ValueError('unclosed JSON fence')
        value = '\n'.join(lines[1:-1])
    plan = json.loads(value)
    if not isinstance(plan, list) or not all(isinstance(op, dict) for op in plan):
        raise ValueError('expected a JSON array of operations')
    if any(set(op) != {'name','params'} or not isinstance(op['name'],str)
           or not isinstance(op['params'],dict) for op in plan):
        raise ValueError('invalid operation JSON schema')
    return plan


def main():
    import torch
    from PIL import Image
    from transformers import LogitsProcessorList
    from test_regr.VLABenchAgentInterface.native_planning import FrozenNativePlanner, NativeConstraintProcessor
    from test_regr.VLABenchAgentInterface.plan_codec import PlanVocabulary
    from ..generalization.libero_protocol import abstract_score
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--constraint-policy', choices=['none', 'shared-verified'], required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-new-tokens', type=int, default=768)
    p.add_argument('--max-operations', type=int, default=12)
    a = p.parse_args(); root = a.output; root.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); torch.manual_seed(a.seed)
    selected = episodes(read(a.manifest))
    settings = {'manifest_sha256': sha(a.manifest), 'model': a.model,
        'model_files': {str(f.relative_to(a.model)):sha(f) for f in sorted(Path(a.model).iterdir())
                        if f.is_file() and f.suffix in ('.safetensors','.json','.model')},
        'source_files': {str(f):sha(f) for folder in (Path(__file__).parents[1],
            Path(__file__).parents[2]/'VLABenchAgentInterface',Path(__file__).parents[3]/'domiknows')
            for f in folder.rglob('*.py')},
        'policy': a.constraint_policy, 'seed':a.seed, 'max_new_tokens':a.max_new_tokens,
        'max_operations':a.max_operations, 'dtype':'bfloat16', 'load_in_4bit':False,
        'trained_joint_heads':False, 'grounding_head':False, 'siglip_hints':False,
        'decoder':'native Qwen language-model head; greedy',
        'constraints':sorted(SHARED_RULES) if a.constraint_policy != 'none' else [],
        'prompt_protocol':'anonymous-numbered-images-public-instruction-initial-relations-v2-explicit-params-object'}
    freeze(root/'settings.json',settings)
    if (root/'report.json').exists():
        print(json.dumps(read(root/'report.json')),flush=True);return
    rows = [json.loads(s) for s in (root/'predictions.jsonl').read_text().splitlines()] if (root/'predictions.jsonl').exists() else []
    if [r['episode_id'] for r in rows] != [e['episode_id'] for e in selected[:len(rows)]]:
        raise ValueError('resume episode order differs')
    planner = FrozenNativePlanner.load(a.model,load_in_4bit=False)
    vocabulary = PlanVocabulary.default(max_entities=max(len(public_input(e)['entities']) for e in selected))
    decoders = {}
    with (root/'predictions.jsonl').open('a') as output, (root/'decisions.jsonl').open('a') as decisions:
        for index, episode in enumerate(selected):
            if index < len(rows):continue
            public = public_input(episode); rendered = prompt(public,vocabulary)
            images = [Image.open(path).convert('RGB') for path in public['images']]
            content = [{'type':'image','image':image} for image in images]+[{'type':'text','text':rendered}]
            text = planner.processor.apply_chat_template([{'role':'user','content':content}],
                tokenize=False,add_generation_prompt=True,enable_thinking=False)
            record = {'episode_id':episode['episode_id'],'prompt':text,
                      'images':{path:sha(path) for path in public['images']}}
            freeze(root/'inputs'/f'{index:04d}.json',record)
            inputs = planner.processor(text=[text],images=images,return_tensors='pt').to(next(planner.model.parameters()).device)
            length = inputs.input_ids.shape[1];raw='';tokens=[];prediction=[];error=None;dfa=None
            processor = None
            if a.constraint_policy != 'none':
                key = (tuple(public['entities']),tuple(public['movable']))
                if key not in decoders:
                    decoders[key] = compiled_decoder(planner.token_bytes,planner.tokenizer.eos_token_id,public,vocabulary,a.max_operations)
                dfa = decoders[key]
                processor = NativeConstraintProcessor(dfa,length,a.max_new_tokens)
            class Observe:
                def __call__(self, ids, scores):
                    step = ids.shape[1]-length; original = scores[0].float()
                    top = int(original.argmax());changed = processor(ids,scores) if processor else scores
                    decisions.write(json.dumps({'episode_id':episode['episode_id'],'step':step,
                        'raw_top':top,'selected_top':int(changed[0].argmax()),
                        'raw_top_rejected':not bool(torch.isfinite(changed[0,top])),
                        'allowed_count':int(torch.isfinite(changed[0]).sum())})+'\n')
                    return changed
            try:
                with torch.inference_mode():
                    tokens = planner.model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False,use_cache=True,
                        logits_processor=LogitsProcessorList([Observe()]),eos_token_id=planner.tokenizer.eos_token_id)[0,length:].tolist()
                raw = planner.tokenizer.decode(tokens,skip_special_tokens=True)
                if dfa is not None and not dfa.accepts(tokens):raise ValueError('constrained plan truncated before acceptance')
                prediction = parse_plan(raw)
            except (ValueError,RuntimeError,KeyError) as exc:
                if isinstance(exc,torch.cuda.OutOfMemoryError):raise
                raw = getattr(exc,'raw',raw);tokens = list(getattr(exc,'tokens',tokens))
                error = {'type':type(exc).__name__,'message':str(exc)}
            finally:
                for image in images:image.close()
            # Hidden BDDL goals/references are first accessed after generation.
            if sha(episode['scoring_private']) != episode['files'][episode['scoring_private']]:
                raise ValueError('scoring-private artifact changed')
            score = abstract_score(prediction,read(episode['scoring_private']))
            row = {'episode_id':episode['episode_id'],'prediction':prediction,'generation_error':error,
                   'input_hash':digest(record),'raw':raw,'tokens':tokens,**score}
            output.write(json.dumps(row)+'\n');output.flush();decisions.flush();rows.append(row)
            print(f'[native-baseline] {a.constraint_policy} {index+1}/100 valid={score["valid_plan"]} goal={score["goal_success"]} error={error}',flush=True)
    report = {'examples':100,'valid_plan_rate':sum(r['valid_plan'] and not r['generation_error'] for r in rows)/100,
        'abstract_goal_success':sum(r['goal_success'] and not r['generation_error'] for r in rows)/100,
        'generation_errors':sum(bool(r['generation_error']) for r in rows),
        'input_hashes':{r['episode_id']:r['input_hash'] for r in rows},'settings':settings}
    freeze(root/'report.json',report);print('[native-baseline] COMPLETE '+json.dumps({k:v for k,v in report.items() if k not in ('settings','input_hashes')}),flush=True)


if __name__ == '__main__':main()
