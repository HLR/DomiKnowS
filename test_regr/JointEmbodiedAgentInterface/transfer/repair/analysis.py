"""Predeclared paired confirmation contrasts; no checkpoint selection here."""
import argparse
import json
from collections import Counter
from pathlib import Path
from ..protocol import read,freeze,episodes,sha
from ..anonymous_grounding.planner_binding.verification_protocol import paired_interval
from .evaluate import SYSTEMS
from ...generalization.libero_protocol import wilson
from .input_protocol import protocol_from_settings


def collect(root, output=None):
    output = Path(output or root)
    output.mkdir(parents=True, exist_ok=True)
    from .preservation_audit import collect as audit_preservation
    preservation = audit_preservation(root)
    c=read(root/'config.json');ids=[e['episode_id'] for e in episodes(read(root/'confirmation/manifest.json'))]
    outcomes={};rows=[];common=None;coverage={};failures={};selection=read(root/'frozen_selection.json');input_protocol=None
    for system in SYSTEMS:
        fingerprints=[]
        for policy in ('none','shared-verified'):
            folder=root/'evaluation'/system/policy;report=read(folder/'report.json')
            condition_protocol=protocol_from_settings(read(folder/'settings.json'))
            if input_protocol is None:input_protocol=condition_protocol
            if condition_protocol!=input_protocol:raise ValueError('evaluation input protocols differ')
            plans={r['episode_id']:r for r in map(json.loads,(folder/'predictions.jsonl').read_text().splitlines())}
            native=[json.loads(s) for s in (folder/'execution/episodes.jsonl').read_text().splitlines()]
            actual={r['episode_id']:r for r in native}
            if len(native)!=100 or set(actual)!=set(ids) or set(plans)!=set(ids):raise ValueError('incomplete native condition')
            if common is None:common=report['input_hashes']
            if common!=report['input_hashes']:raise ValueError('matched neural inputs differ')
            if max(r['steps'] for r in native)>1000:raise ValueError('horizon exceeded')
            if any(actual[i]['prediction']!=plans[i]['prediction'] for i in ids):raise ValueError('execution used other predictions')
            outcomes[system,policy]=actual
            counts=Counter();rules=Counter()
            for line in (folder/'decisions.jsonl').open():
                decision=json.loads(line)
                if 'raw_top_rejected' in decision:
                    counts['decisions']+=1;counts['raw_top_rejected']+=bool(decision['raw_top_rejected'])
                elif 'raw_argmax_allowed' in decision:
                    counts['decisions']+=1;counts['raw_top_rejected']+=not decision['raw_argmax_allowed']
                else:rules.update(decision.get('compiled_rule_trace',{}).get('triggered_rejections',{}))
            coverage[system+':'+policy]={'token_policy':dict(counts),'compiled_rules_rejecting_alternatives':dict(rules),
                'note':'Native token-policy rejections include lexical/schema projection; canonical rule traces are separately attributed.'}
            failure=Counter()
            for i in ids:
                p=plans[i]
                if p['generation_error']:failure['generation_error']+=1
                elif not p['valid_plan']:failure['syntax_capability_or_precondition']+=1
                elif not p['goal_success']:
                    failure['legal_grounding_or_goal_error']+=1
                    failure['legal_wrong_source']+=not p['grounding']['source_correct']
                    failure['legal_wrong_destination']+=not p['grounding']['destination_correct']
                elif not actual[i]['success']:failure['abstract_success_physical_failure']+=1
                failure['truncated']+=p.get('truncated',False)
            failures[system+':'+policy]=dict(failure)
            successes=sum(r['success'] for r in native)
            rows.append({'system':system,'policy':policy,'native_percent':sum(r['success'] for r in native),
                'source_qualified':selection['systems'][system]['qualified'],
                'selected_source_epoch':selection['systems'][system].get('source_epoch',0),
                'native_ci95_percent':[100*v for v in wilson(successes,100)],
                'completed_percent':sum(r['success'] and r['termination_reason']=='completed' for r in native),
                **{k:report[k] for k in ('abstract_percent','valid_percent','raw_format_percent','normalized_episodes','generation_errors')}})
    primary={arm:paired_interval(outcomes['base','none'],outcomes[arm+'-trained','none'],ids,confidence=.975)
             for arm in ('native','joint')}
    own={'native':paired_interval(outcomes['base','none'],outcomes['native-trained','none'],ids),
         'joint':paired_interval(outcomes['joint-initialization','none'],outcomes['joint-trained','none'],ids)}
    constraints={k:paired_interval(outcomes[k,'none'],outcomes[k,'shared-verified'],ids) for k in SYSTEMS}
    result={'complete':True,'rows':rows,'primary_97_5':primary,'own_initialization_secondary_95':own,
        'dfa_secondary_95':constraints,'bootstrap_samples':10000,'seed':42,'task_clusters':20,
        'effective_coverage':coverage,'failure_partition':failures,
        'greedy_preservation':{'passed':preservation['passed'],
            'sha256':sha(root/'confirmation/preservation_audit.json')},
        'input_protocol':input_protocol,
        'scope':'Planner transfer through fixed executor; 20 existing LIBERO tasks, confirmation states 35-39',
        'selection_sha256':sha(root/'frozen_selection.json')}
    freeze(output/'results.json',result)
    columns=('system','policy','native_percent','completed_percent','abstract_percent','valid_percent','raw_format_percent','normalized_episodes','generation_errors')
    lines=['# Fresh decoder repair confirmation','',result['scope'],'',f'Input protocol: {input_protocol}.','',
        '| System | Policy | Native % | Completed % | Abstract % | Valid % | Integer format % | Episodes with ID conversions | Errors |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    lines+=['| '+' | '.join(str(row[k]) for k in columns)+' |' for row in rows]
    for arm,contrast in primary.items():
        lines+=['',f"{arm} minus Base, no DFA: {contrast['delta']*100:+.1f} pp; wins/losses {contrast['wins']}/{contrast['losses']}; 97.5% paired task-cluster interval {[100*v for v in contrast['task_cluster_interval']]}."]
    lines+=['','Both arms started independently from identical pretrained Qwen3-VL-8B-Instruct weights. Four source-only VLABench + EAI epochs used anonymous entities, consistent ID permutations, balanced full-vocabulary CE, preservation KL 0.1, EAI weight 1.0, no grounding hints/residuals and no training DFA/mass regularizer. LoRA LR 5e-6, graph heads 1e-4, AdamW weight decay 0.01, 10% warm-up, cosine decay, seed 42, batch 8. Native JSON uses the frozen pretrained LM head; Joint uses its trainable graph-label head. Source validation alone selects epochs; failed gates remain failed. Legality comes from DomiKnowS declarations compiled to DFA; the repaired surface preserves public ID aliases and JSON fences. Physical execution uses the unchanged qualified bridge and 1,000-step horizon.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path);a=p.parse_args();collect(a.root,a.output)
