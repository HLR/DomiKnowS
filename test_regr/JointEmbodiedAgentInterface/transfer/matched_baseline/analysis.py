"""Paired, task-clustered reporting for a frozen eight-condition campaign."""
import argparse
import json
from pathlib import Path
from ..protocol import read,freeze,episodes
from ...generalization.libero_protocol import wilson
from ..anonymous_grounding.planner_binding.verification_protocol import paired_interval
from .protocol import MODELS,POLICIES


def collect(root,config):
    ids=[e['episode_id'] for e in episodes(read(config['manifest']))]
    rows=[];outcomes={};cores=None;input_hashes={}
    for key in MODELS:
        for policy in POLICIES:
            folder=Path(root)/'evaluation'/key/policy;report=read(folder/'report.json')
            plans=[json.loads(s) for s in (folder/'predictions.jsonl').read_text().splitlines()]
            native=[json.loads(s) for s in (folder/'execution/episodes.jsonl').read_text().splitlines()]
            summary=read(folder/'execution/summary.json');byid={r['episode_id']:r for r in plans}
            actual={r['episode_id']:r for r in native}
            if len(plans)!=100 or len(native)!=100 or set(byid)!=set(ids) or set(actual)!=set(ids):
                raise ValueError('incomplete condition '+key+'/'+policy)
            if cores is None:cores=report['core_hashes']
            if cores!=report['core_hashes']:raise ValueError('public prompt core differs')
            if any(actual[i]['prediction']!=byid[i]['prediction'] for i in ids):raise ValueError('executor used another plan')
            if max(r['steps'] for r in native)>1000:raise ValueError('physical horizon exceeded')
            successes=sum(r['success'] for r in native)
            if successes!=summary['successes']:raise ValueError('native score arithmetic differs')
            input_hashes[key,policy]=report['input_hashes'];outcomes[key,policy]=actual
            rows.append({'model':key,'policy':policy,'examples':100,'native_percent':successes,
                'native_ci95_percent':[v*100 for v in wilson(successes,100)],
                'completed_percent':sum(r['success'] and r['termination_reason']=='completed' for r in native),
                'abstract_percent':report['abstract_percent'],'valid_percent':report['valid_percent'],
                'raw_format_percent':report['raw_format_percent'],'normalized_episodes':report['normalized_episodes'],
                'generation_errors':report['generation_errors']})
        if input_hashes[key,'none']!=input_hashes[key,'shared-verified']:
            raise ValueError('constraint conditions changed neural inputs '+key)
    for policy in POLICIES:
        if input_hashes['initialization',policy]!=input_hashes['B',policy]:
            raise ValueError('matched training comparison changed neural inputs')
    training={p:paired_interval(outcomes['initialization',p],outcomes['B',p],ids,confidence=.975) for p in POLICIES}
    systems={k+':'+p:paired_interval(outcomes['base',p],outcomes[k,p],ids) for k in ('A','B') for p in POLICIES}
    constraints={k:paired_interval(outcomes[k,'none'],outcomes[k,'shared-verified'],ids) for k in MODELS}
    result={'complete':True,'rows':rows,'training_primary_97_5':training,'system_secondary_95':systems,
        'constraint_secondary_95':constraints,'bootstrap_samples':10000,'seed':42,'task_clusters':20,
        'same_public_prompt_core':True,'paired_policy_inputs_equal':True,'initialization_B_full_inputs_equal':True,
        'scope':'previously inspected 20 LIBERO tasks, states 0-4; no retraining; whole-system vs matched-continuation contrasts separated'}
    freeze(Path(root)/'results.json',result)
    lines=['# Matched LIBERO evaluation','',result['scope'],'',
        '| Model | Policy | Native % | Completed % | Abstract % | Valid % | Raw format % | ID-normalized n | Errors |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('| '+' | '.join(str(r[k]) for k in ('model','policy','native_percent','completed_percent','abstract_percent','valid_percent','raw_format_percent','normalized_episodes','generation_errors'))+' |')
    lines+=['','## Matched training contrasts (B minus own pre-update control; 97.5% task-cluster intervals)','']
    for p,r in training.items():lines.append(f"- {p}: {100*r['delta']:+.1f} pp; wins/losses {r['wins']}/{r['losses']}; CI {[round(100*v,1) for v in r['task_cluster_interval']]}")
    lines+=['','All models use NF4/BF16, a common public prefix, identical images and calibrated executor. A uses its own learned hints; initialization/B use identical frozen paired-SigLIP hints and their own saved operation heads. Native Qwen and Joint retain their own decoders; encoder-specific token budgets (768 BPE vs 61 graph labels) are explicit. No Python planning constraints were introduced. Exact public ID normalization changes representation only, and raw format validity is reported separately.']
    (Path(root)/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    print(json.dumps(collect(a.root,read(a.root/'config.json')),indent=2))
