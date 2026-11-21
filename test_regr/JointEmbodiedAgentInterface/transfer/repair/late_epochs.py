"""User-requested exploratory Joint epochs; preserve the selected primary model."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

if not __package__:
    experiment=Path(sys.argv[sys.argv.index('--root')+1])
    sys.path.insert(0,str(experiment/'source/test_regr/JointEmbodiedAgentInterface'));__package__='transfer.repair'

from ..protocol import read,freeze,sha,episodes
from ..campaign import run,save,utc
from ..anonymous_grounding.planner_binding.verification_control import native_command
from .campaign import gpu_job,model_command,MODULE
from .protocol import verify


def registration(epoch, row):
    if epoch not in (3,4) or row['epoch']!=epoch:
        raise ValueError('only explicit exploratory epochs 3 and 4 are registered')
    return {'path':row['checkpoint'],'sha256':row['checkpoint_sha256'],
        'source_epoch':epoch,'source_progress_score':row['progress_score'],
        'qualified':row['eai_successes']>=47 and row['progress_score']>=45.11648148148148,
        'exploratory':True,'primary_selected_epoch':2,
        'registration':'user-requested evaluation after primary confirmation; not new checkpoint selection'}


def matched(actual, expected, identities):
    if len(identities)!=len(set(identities)) or set(actual)!=set(identities) or actual!=expected:
        raise ValueError('evaluation inputs/identities differ from the archived epoch-2 comparison')


def collect(root):
    root=Path(root);main=read(root/'config.json');rows=[];files={}
    baseline=Path(main['baseline_comparison']);source=Path(main['baseline_source_test'])
    expected_vla=read(source/'evaluation/joint-trained/none/report.json')
    expected_lib=read(baseline/'evaluation/joint-trained/none/report.json')
    expected_ids=[e['episode_id'] for e in episodes(read(baseline/'confirmation/manifest.json'))]
    for epoch in (3,4):
        folder=root/f'epoch_{epoch:03d}';c=read(folder/'config.json');verify(c)
        spec=read(folder/'frozen_selection.json')['systems']['joint-trained']
        if sha(spec['path'])!=spec['sha256'] or not spec['exploratory'] or spec['primary_selected_epoch']!=2:
            raise ValueError('exploratory checkpoint registration changed')
        eai_path=Path(c['training_parent'])/f'arms/joint/epoch_{epoch:03d}/eai.json';eai=read(eai_path)
        if eai['examples']!=88:raise ValueError('canonical EAI coverage changed')
        files[str(eai_path)]=sha(eai_path)
        for policy in ('none','shared-verified'):
            vla_path=folder/'source_test'/policy/'report.json';vla=read(vla_path)
            if vla['examples']!=450 or vla['settings']['checkpoint']!=spec:
                raise ValueError('source test checkpoint/coverage changed')
            matched(vla['input_hashes'],expected_vla['input_hashes'],expected_vla['episode_ids'])
            if read(folder/'source_test'/policy/'references.json')!=read(source/'evaluation/joint-trained/none/references.json'):
                raise ValueError('frozen source scoring references differ')
            lib_path=folder/'evaluation/joint-trained'/policy/'report.json';lib=read(lib_path)
            matched(lib['input_hashes'],expected_lib['input_hashes'],expected_ids)
            path=lib_path.parent/'execution/episodes.jsonl'
            outcomes=[json.loads(s) for s in path.read_text().splitlines()]
            plans={r['episode_id']:r for r in map(json.loads,(lib_path.parent/'predictions.jsonl').read_text().splitlines())}
            indexed={r['episode_id']:r for r in outcomes}
            if len(outcomes)!=100 or set(indexed)!=set(expected_ids) or set(plans)!=set(expected_ids):
                raise ValueError('incomplete or duplicate native execution identities')
            if any(r['steps']>1000 for r in outcomes):raise ValueError('physical horizon exceeded')
            if any(indexed[i]['prediction']!=plans[i]['prediction'] for i in expected_ids):
                raise ValueError('executor used other predictions')
            rows.append({'epoch':epoch,'exploratory':True,'source_qualified':spec['qualified'],'policy':policy,
                'eai_goal_percent':eai['goal_success']*100,'vla_Macro_0_80':vla['Macro'],
                'progress_score_0_100':vla['progress_score'],'libero_abstract_percent':lib['abstract_percent'],
                'libero_physical_percent':sum(r['success'] for r in outcomes),
                'completed_successful_plans':sum(r['success'] and r['termination_reason']=='completed' for r in outcomes),
                'vla_generation_errors':vla['generation_errors'],'libero_generation_errors':lib['generation_errors']})
            for p in (vla_path,lib_path,path):files[str(p)]=sha(p)
    result={'complete':True,'rows':rows,'primary_selected_epoch':2,
        'scope':'Post-hoc exploratory comparison on already-used strict test episodes; no reselection or independent-confirmation claim',
        'matching_inputs_and_references':True,'vlabench_examples_per_condition':450,
        'libero_episodes_per_condition':100,'physical_horizon':1000,'files':files}
    freeze(root/'table_results.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--collect',action='store_true');a=p.parse_args();root=a.root
    if a.collect:collect(root);return
    c=read(root/'config.json')
    try:
        verify(c)
        run(root,'tests',model_command(root,c,'tests','pytest',[
            'test_regr/JointEmbodiedAgentInterface/transfer/repair/test_late_epochs.py',
            'test_regr/JointEmbodiedAgentInterface/transfer/repair/test_source_test.py',
            'test_regr/JointEmbodiedAgentInterface/transfer/repair/test_input_protocol.py','-q']),{'physical_gpu':None})
        for epoch in (3,4):
            sub=root/f'epoch_{epoch:03d}';ec=read(sub/'config.json');verify(ec)
            gate=read(sub/'confirmation/calibration_reuse.json')
            executor=Path(ec['libero_base'])/'evaluation_source/test_regr/JointEmbodiedAgentInterface/generalization'
            if (gate['manifest_sha256']!=sha(sub/'confirmation/manifest.json') or
                gate['executor_source_sha256']!=sha(executor/'fixed_executor.py') or
                gate['bridge_source_sha256']!=sha(executor/'libero_campaign.py')):
                raise ValueError('qualified executor/bridge/calibration changed')
        save(root/'status.json',{'stage':'exploratory-epochs-3-and-4-evaluating','updated_utc':utc()})
        errors=[];executions=[]
        with ThreadPoolExecutor(max_workers=4) as planning,ThreadPoolExecutor(max_workers=2) as physical:
            def condition(epoch,policy,dataset):
                sub=root/f'epoch_{epoch:03d}';ec=read(sub/'config.json');name=f'late20261121-e{epoch}-{dataset}-{policy}'
                if dataset=='vlabench':
                    out=sub/'source_test'/policy
                    gpu_job(sub,ec,name,MODULE+'.source_test',['--config',sub/'config.json','--arm','joint-trained',
                        '--policy',policy,'--output',out])
                else:
                    out=sub/'evaluation/joint-trained'/policy
                    spec=read(sub/'frozen_selection.json')['systems']['joint-trained']
                    gpu_job(sub,ec,name,MODULE+'.evaluate',['--config',sub/'config.json','--arm','joint-trained',
                        '--checkpoint',spec['path'],'--manifest',sub/'confirmation/manifest.json',
                        '--policy',policy,'--output',out,'--input-protocol','image-instruction-v1'])
                    executions.append(physical.submit(run,sub,name+'-native',native_command(root,ec,name+'-native',
                        sub/'confirmation/manifest.json',sub/'confirmation/calibration_reuse.json',
                        out/'predictions.jsonl',out/'execution'),{'physical_gpu':None,'horizon':1000,'exploratory_epoch':epoch}))
            tasks=[(e,p,d) for d in ('vlabench','libero') for e in (3,4) for p in ('none','shared-verified')]
            for f in as_completed([planning.submit(condition,*task) for task in tasks]):
                try:f.result()
                except Exception as exc:errors.append(str(exc))
            for f in executions:
                try:f.result()
                except Exception as exc:errors.append(str(exc))
        if errors:raise RuntimeError('completed conditions preserved: '+'; '.join(errors))
        collect(root)
        freeze(root/'complete.json',{'table_results_sha256':sha(root/'table_results.json'),'training_launched':False})
        save(root/'status.json',{'stage':'complete-awaiting-google-table-fill','updated_utc':utc()})
    except Exception as exc:
        save(root/'status.json',{'stage':'failed','error':str(exc),'updated_utc':utc()});raise


if __name__=='__main__':main()
