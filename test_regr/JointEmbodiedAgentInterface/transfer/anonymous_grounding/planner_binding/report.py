"""Matched fresh-state results, without ranking models on external outcomes."""
from pathlib import Path
from ...protocol import read,freeze,jsonl,indexed
from ..object_fixes.report import paired,wilson


def report(root,c):
    root=Path(root);summaries={};native={};all_inputs=[]
    ids=[e['episode_id'] for e in read(c['external_manifest'])['episodes'] if e['split']=='evaluation']
    for model in ('initialization','masked','full'):
        for policy in ('none','executor'):
            folder=root/'evaluation'/model/policy;plans=indexed(jsonl(folder/'predictions.jsonl'),ids)
            outcomes=indexed(jsonl(folder/'execution/episodes.jsonl'),ids);native[model,policy]=outcomes
            if any(plans[i]['prediction']!=outcomes[i]['prediction'] for i in ids):raise ValueError('native execution used another plan')
            all_inputs.append({i:plans[i]['input_hash'] for i in ids});cohorts={}
            for suite in ('all','libero_object','libero_spatial'):
                selected=[i for i in ids if suite=='all' or plans[i]['suite']==suite];k=sum(outcomes[i]['success'] for i in selected)
                cohorts[suite]={'episodes':len(selected),'native_successes':k,'native_wilson_ci95':wilson(k,len(selected)),
                    'completed_successful_plans':sum(outcomes[i]['success'] and outcomes[i]['termination_reason']=='completed' for i in selected),
                    'valid_plans':sum(plans[i]['valid_plan'] for i in selected),'abstract_successes':sum(plans[i]['goal_success'] for i in selected),
                    'generation_errors':sum(plans[i]['generation_error'] is not None for i in selected)}
            summaries[model+':'+policy]={'cohorts':cohorts}
    if any(x!=all_inputs[0] for x in all_inputs[1:]):raise ValueError('matched evaluation inputs differ')
    comparisons={}
    for policy in ('none','executor'):
        for after,before in [('masked','initialization'),('full','initialization'),('full','masked')]:
            comparisons[after+'-vs-'+before+':'+policy]=paired(native[before,policy],native[after,policy],ids)
    freeze(root/'results.json',{'reports':summaries,'paired_native_changes':comparisons,
        'selection':read(root/'selected_snapshots.json'),'fresh_manifest_sha256':c['external_manifest_sha256'],
        'scope':'Source-only historical internal initialization; fresh LIBERO states 40--44; source validation selection; fixed executor transfer.'})
    lines=['# Per-operation binding: matched CE continuations','',
        'Four source-only epochs in each arm; both use operation grounding, anonymous ID permutations, '
        'frozen paired-SigLIP primary hints, and allowed-mass weight 0.5. Only the VLABench graph-label CE denominator differs. '
        'No target-domain training. Epoch selection is source-validation-only within each arm.','',
        '| Snapshot | Policy | Native /100 | Object /50 | Spatial /50 | Valid /100 |','|---|---|---:|---:|---:|---:|']
    for key,value in summaries.items():
        model,policy=key.split(':');s=value['cohorts'];lines.append(f"| {model} | {policy} | {s['all']['native_successes']} | {s['libero_object']['native_successes']} | {s['libero_spatial']['native_successes']} | {s['all']['valid_plans']} |")
    lines+=['','All conditions use the same fresh 100 official episodes, states 40--44. '
        'The inspected 0--4 pilot is a separate diagnostic and is not a matched numerical baseline. '
        'Native goal predicates and reference roles stayed evaluator-only. Confidence intervals and paired '
        'task-cluster differences are in results.json. EAI retention is in each no-DFA evaluation/eai.json.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
