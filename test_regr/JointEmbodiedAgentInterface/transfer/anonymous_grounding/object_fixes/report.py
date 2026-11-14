"""CPU-only paired diagnostic report, native scores separated from abstract goals."""
from collections import defaultdict
import random
from pathlib import Path
from ...protocol import read,jsonl,indexed,freeze
from .provenance import validate_inputs


def wilson(k,n):
    z=1.96;den=1+z*z/n;p=k/n;mid=(p+z*z/(2*n))/den
    half=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/den
    return [max(0.,mid-half),min(1.,mid+half)]


def paired(old,new,ids):
    changes=[int(new[i]['success'])-int(old[i]['success']) for i in ids]
    groups=defaultdict(list)
    for i,change in zip(ids,changes):groups[i.rsplit('/',1)[0]].append(change)
    groups=list(groups.values());rng=random.Random(42);samples=[]
    for _ in range(4000):
        values=[v for group in rng.choices(groups,k=len(groups)) for v in group]
        samples.append(sum(values)/len(values))
    samples.sort()
    return {'wins':changes.count(1),'losses':changes.count(-1),'unchanged':changes.count(0),
        'delta':sum(changes)/len(changes),'paired_task_cluster_bootstrap_ci95':[samples[99],samples[3899]],
        'bootstrap_seed':42,'bootstrap_resamples':4000,'task_clusters':len(groups)}


def report(root,c):
    validate_inputs(c)
    variants=c['conditions'];plans={};execution={};summaries={}
    baseline_ids=[r['episode_id'] for r in jsonl(Path(root)/'planning/baseline/predictions.jsonl')]
    for variant in variants:
        folder=Path(root)/'planning'/variant
        plans[variant]=indexed(jsonl(folder/'predictions.jsonl'),baseline_ids)
        execution_dir=Path(c['archive_execution']) if variant=='baseline' else Path(root)/'execution'/variant
        execution[variant]=indexed(jsonl(execution_dir/'episodes.jsonl'),baseline_ids)
        for identity in baseline_ids:
            if execution[variant][identity]['prediction']!=plans[variant][identity]['prediction']:
                raise ValueError('native execution used different predictions')
            if plans[variant][identity]['observation_hash']!=plans['baseline'][identity]['observation_hash']:
                raise ValueError('public observation changed between conditions')
        summaries[variant]={'planning':read(folder/'summary.json'),'execution_source':str(execution_dir),
            'execution_reused':variant=='baseline','cohorts':{}}
        for suite in ('all','libero_object','libero_spatial'):
            ids=[i for i in baseline_ids if suite=='all' or plans[variant][i]['suite']==suite]
            selected=[execution[variant][i] for i in ids];successes=sum(bool(r['success']) for r in selected)
            summaries[variant]['cohorts'][suite]={'episodes':len(ids),'native_successes':successes,
                'native_success_rate':successes/len(ids),'native_wilson_ci95':wilson(successes,len(ids)),
                'completed_successful_plans':sum(r['success'] and r['termination_reason']=='completed'
                                                and r.get('holding_at_end') is None for r in selected),
                'paired_vs_baseline':paired(execution['baseline'],execution[variant],ids)}
    for first,second in [('baseline','contract'),('raw','raw-contract'),('aligned','aligned-contract')]:
        if any(plans[first][i]['input_hash']!=plans[second][i]['input_hash'] for i in baseline_ids):
            raise ValueError('capability condition changed actual model inputs')
    # Save representative paired changes without selecting any further prompts or weights.
    examples=[]
    for variant in variants[1:]:
        for suite in ('libero_object','libero_spatial'):
            for direction in (1,-1):
                ids=[i for i in baseline_ids if plans[variant][i]['suite']==suite and
                     int(execution[variant][i]['success'])-int(execution['baseline'][i]['success'])==direction]
                for i in ids[:2]:examples.append({'condition':variant,'episode_id':i,'change':direction,
                    'baseline':plans['baseline'][i],'candidate':plans[variant][i]})
    freeze(Path(root)/'results.json',{'conditions':summaries,'baseline_reproduction':read(Path(root)/'planning/baseline/baseline_gate.json'),
        'interpretation':'Previously inspected diagnostic test scenes; no training, no model selection, no confirmatory transfer claim.',
        'pairing':'Same frozen 100 episodes, initial states, RGB, numeric IDs, snapshot, native executor and horizon.',
        'grounding':'Frozen paired SigLIP retrieval; nearest competitors recorded; no hard-negative fine-tuning performed.'})
    freeze(Path(root)/'paired_examples.json',examples)
    lines=['# LIBERO object-transfer diagnostics','',
        'Frozen continuation epoch-2 snapshot; no training updates. Same 100 previously inspected episodes. '
        'Baseline generation and scores reproduced; baseline physical execution is the explicitly reused archive. '
        'Five intervention executions are fresh, using the unchanged 96/100-calibrated executor and 1,000-step horizon.','',
        '| Condition | Native /100 | Object /50 | Spatial /50 | Valid /100 | Grounder source Object /50 | Planner source Object /50 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for v in variants:
        s=summaries[v];co=s['cohorts'];pl=s['planning'];obj=pl['suites']['libero_object']
        lines.append(f"| {v} | {co['all']['native_successes']} | {co['libero_object']['native_successes']} | "
            f"{co['libero_spatial']['native_successes']} | {pl['overall']['valid_plans']} | "
            f"{obj['grounder_source_correct']} | {obj['planner_source_correct']} |")
    lines+=['','`raw`: learned head queries original public instruction. `contract`: executor pick/place schemas, '
        'range/static-fixture/holding preconditions, empty-hand completion; no task goal or answer template. '
        '`aligned`: instruction role phrases ranked against RGB-only object regions and context with frozen paired '
        'SigLIP features, all visible candidates; no category lookup or hidden reference inputs.','',
        'Per-cohort native confidence intervals and paired task-cluster bootstrap intervals are in results.json. '
        'Episode Wilson intervals do not account for within-task scene correlation. Nearest competing objects and '
        'margins are recorded, but no new hard-negative training has been performed. This is a diagnostic regression '
        'on inspected scenes; a further unseen-scene evaluation is required for a generalization claim.']
    (Path(root)/'RESULTS.md').write_text('\n'.join(lines)+'\n')
