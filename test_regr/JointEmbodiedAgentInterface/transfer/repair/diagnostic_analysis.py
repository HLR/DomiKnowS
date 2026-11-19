"""Factorial pointer interventions and historical greedy-preservation checks."""
import argparse
import json
from pathlib import Path
from ..protocol import read,freeze,episodes
from ..anonymous_grounding.planner_binding.verification_protocol import paired_interval


def rows(path):return {r['episode_id']:r for r in map(json.loads,Path(path).read_text().splitlines())}


def collect(root,c):
    ids=[r['episode_id'] for r in episodes(read(c['historical_manifest']))]
    diagnostics=Path(c.get('diagnostics_reuse_directory',root/'diagnostics'))
    plans={};outcomes={};summary=[]
    for planner in ('initialization','B'):
        for residual in ('off','B'):
            key=planner+':'+residual;folder=diagnostics/key.replace(':','-')/'evaluation'/planner/'none'
            plans[key]=rows(folder/'predictions.jsonl');outcomes[key]=rows(folder/'execution/episodes.jsonl')
            if set(plans[key])!=set(ids) or set(outcomes[key])!=set(ids):raise ValueError('diagnostic coverage differs')
            summary.append({'planner':planner,'residual':residual,'native_percent':sum(r['success'] for r in outcomes[key].values()),
                'head':'B trained head; transplant into initialization is diagnostic only'})
    archive=Path(c['historical_archive'])
    for key,old in [('initialization:off','initialization'),('B:B','B')]:
        original=rows(archive/'evaluation'/old/'none/predictions.jsonl')
        if any(plans[key][i]['prediction']!=original[i]['prediction'] or plans[key][i]['input_hash']!=original[i]['input_hash'] for i in ids):
            raise ValueError('factorial original control failed to reproduce: '+key)
    contrasts={}
    for name,left,right in [('planner drift, residual off','initialization:off','B:off'),
        ('planner drift, B residual','initialization:B','B:B'),
        ('residual at pre-B planner','initialization:off','initialization:B'),
        ('residual at B planner','B:off','B:B')]:
        contrast=paired_interval(outcomes[left],outcomes[right],ids)
        changes=[]
        for i in ids:
            a,b=plans[left][i],plans[right][i]
            def pointers(row,key):return [op.get('params',{}).get(key) for op in row['prediction'] if key in op.get('params',{})]
            changed=[role for role,key in [('source','target_entity_name'),('destination','target_container_name')]
                if pointers(a,key)!=pointers(b,key)]
            la,lb=a['labels'],b['labels'];first=next((n for n,(x,y) in enumerate(zip(la,lb)) if x!=y),
                min(len(la),len(lb)) if len(la)!=len(lb) else None)
            changes.append({'episode_id':i,'changed_roles':changed,'first_divergence':first,
                'native_before':outcomes[left][i]['success'],'native_after':outcomes[right][i]['success']})
        contrasts[name]={'paired':contrast,'episodes':changes}
    original=rows(archive/'evaluation/base/none/predictions.jsonl')
    old_dfa=rows(archive/'evaluation/base/shared-verified/execution/episodes.jsonl')
    old_none=rows(archive/'evaluation/base/none/execution/episodes.jsonl')
    folder=Path(c.get('repaired_diagnostic_reuse_directory',root/'diagnostics/base-repaired'))/'evaluation/base/shared-verified'
    repaired=rows(folder/'predictions.jsonl');native=rows(folder/'execution/episodes.jsonl')
    if any(repaired[i]['labels']!=original[i]['labels'] or repaired[i]['prediction']!=original[i]['prediction'] for i in ids):
        raise ValueError('repaired policy changed a historical compiled-legal greedy sequence')
    losses=[{'episode_id':i,'historical_no_dfa':old_none[i]['success'],'historical_dfa':old_dfa[i]['success'],
             'repaired_dfa':native[i]['success'],'prediction_preserved':True}
        for i in ids if old_none[i]['success'] and not old_dfa[i]['success']]
    result={'passed':True,'factorial':summary,'contrasts':contrasts,'repaired_native_percent':sum(r['success'] for r in native.values()),
        'historical_base_losses':losses,'greedy_sequences_preserved':100,'interpretation':'Diagnostic interventions; no claim that a head transplant is a deployable model.'}
    freeze(root/'diagnostic_results.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();collect(a.root,read(a.root/'config.json'))
