"""Bounded two-arm campaign, isolated jobs, source-only selection and fixed tests."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import sys
import subprocess
import json
import time
import ast
import hashlib

if not __package__:
    root=Path(sys.argv[sys.argv.index('--root')+1])
    sys.path.insert(0,str(root/'source/test_regr/JointEmbodiedAgentInterface'));__package__='transfer.repair'
from ..protocol import read,freeze,sha
from ..campaign import run,save,utc
from ..anonymous_grounding.campaign import runtime_environment
from ..anonymous_grounding.planner_binding.epoch_campaign import unused_gpu
from ..anonymous_grounding.planner_binding.verification_control import native_command
from .protocol import verify

MODULE='test_regr.JointEmbodiedAgentInterface.transfer.repair'


def model_command(root,c,name,module,args,gpu=None,uuid=None):
    command=['docker','run','-d','--name',root.name+'-'+name,'--ipc=host','--cpus=8','--memory=64g',
        '-v','/home/auszok:/home/auszok','-v','/home/auszok/DomiKnowS:/workspace','-w',c['source']]
    for k,v in runtime_environment(Path(c['source']),c['planning_dir']).items():command+=['-e',k+'='+v]
    if gpu is not None:command+=['--gpus','device='+str(gpu),'-e','CUDA_VISIBLE_DEVICES='+uuid]
    return command+['--entrypoint','/opt/dominows/venv/bin/python',c['model_image'],'-u','-m',module,*map(str,args)]


def gpu_job(root,c,name,module,args):
    previous=root/'jobs'/f'{name}.json'
    if previous.exists():
        job=read(previous)
        if job.get('config_sha256')!=sha(root/'config.json'):raise ValueError('GPU job configuration changed')
        command=model_command(root,c,name,module,args,job['physical_gpu'],job['gpu_uuid'])
        run(root,name,command,{'physical_gpu':job['physical_gpu'],'gpu_uuid':job['gpu_uuid'],
            'config_sha256':sha(root/'config.json')})
        return
    gpu,uuid,lock=unused_gpu(root,name)
    try:run(root,name,model_command(root,c,name,module,args,gpu,uuid),
        {'physical_gpu':gpu,'gpu_uuid':uuid,'config_sha256':sha(root/'config.json')})
    finally:lock.close()


def train_args(root,c,arm,baseline=False):
    args=['--config',root/'config.json','--arm',arm,'--manifest',c['source_observations'],
          '--output',root/('native_base' if baseline else 'arms/'+arm),'--seed','42']
    if baseline:args+=['--baseline-only']
    return args


def freeze_selection(root):
    systems={'base':{'path':str(root/'native_base/initialization.pt'),'qualified':True},
        'joint-initialization':{'path':str(root/'arms/joint/initialization.pt'),'qualified':True}}
    for arm in ('native','joint'):
        selection=read(root/'arms'/arm/'selection.json')
        selected=selection['selected'] or selection['diagnostic_best']
        systems[arm+'-trained']={'path':selected['checkpoint'],'qualified':selection['qualified'],
            'source_epoch':selected['epoch'],'source_progress_score':selected['progress_score'],
            'selection_sha256':sha(root/'arms'/arm/'selection.json')}
    for spec in systems.values():spec['sha256']=sha(spec['path'])
    # No confirmation predictions are opened before this immutable record.
    freeze(root/'frozen_selection.json',{'systems':systems,'selection_data':'source validation only',
        'primary_contrasts':['native-trained minus base under none','joint-trained minus base under none'],
        'bootstrap':{'samples':10000,'seed':42,'confidence':.975,'cluster':'task, retaining five scenes'}})
    return systems


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--config',type=Path);p.add_argument('--arm',default='both');p.add_argument('--checkpoint',type=Path)
    p.add_argument('--manifest',type=Path);p.add_argument('--output',type=Path);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();root=a.root;c=read(a.config or root/'config.json')
    if a.seed!=42 or a.arm!='both' or a.checkpoint:raise ValueError('campaign scope frozen to two fresh arms')
    try:
        verify(c);save(root/'status.json',{'stage':'tests-and-confirmation-provenance','updated_utc':utc()})
        command=model_command(root,c,'tests','pytest',[
            'test_regr/JointEmbodiedAgentInterface/transfer/repair',
            'test_regr/JointEmbodiedAgentInterface/transfer/matched_baseline/test_protocol.py',
            'test_regr/JointEmbodiedAgentInterface/transfer/test_native_baseline.py','-q'])
        run(root,'tests',command,{'physical_gpu':None})
        if not read(c['confirmation_usage_audit'])['passed']:raise ValueError('confirmation prior-use audit failed')
        base=Path(c['libero_base']);prepare_script=Path(c['source'])/'test_regr/JointEmbodiedAgentInterface/transfer/repair/prepare.py'
        native=native_command(root,c,'freeze',c['historical_manifest'],c['historical_calibration'],root/'unused',root/'unused')
        # Reuse the qualified simulator environment, not its evaluation CLI.
        position=native.index(str(base/'evaluation_source/test_regr/JointEmbodiedAgentInterface/generalization/libero_campaign.py'))
        native=native[:position]+[str(prepare_script),'--config',str(root/'config.json'),'--output',str(root/'confirmation')]
        if c.get('confirmation_reuse'):
            previous=c['confirmation_reuse'];container=previous['container_id']
            while True:
                state=json.loads(subprocess.check_output(['docker','inspect',container],text=True))[0]['State']
                if not state['Running']:break
                save(root/'status.json',{'stage':'waiting-for-owned-confirmation-freeze','container_id':container,'updated_utc':utc()})
                time.sleep(5)
            if state['ExitCode']!=0:raise ValueError('owned confirmation freeze failed; no replacement cohort')
            hashes={}
            for name in ('manifest.json','split_audit.json','calibration_reuse.json','reservation.json'):
                path=Path(previous['directory'])/name;hashes[str(path)]=sha(path)
                freeze(root/'confirmation'/name,read(path))
            gate=read(root/'confirmation/calibration_reuse.json')
            if gate['manifest_sha256']!=sha(root/'confirmation/manifest.json'):
                raise ValueError('reused calibration manifest changed')
            if not read(root/'confirmation/split_audit.json')['passed']:raise ValueError('confirmation split failed')
            freeze(root/'confirmation_reuse.json',{'passed':True,'container_id':container,'files':hashes,
                'reason':'Same pre-scoring frozen observations and qualified executor; source-only EAI count correction'})
        else:run(root,'freeze',native,{'physical_gpu':None,'states':list(range(35,40)),'model_loaded':False})
        gate=read(root/'confirmation/calibration_reuse.json')
        executor=base/'evaluation_source/test_regr/JointEmbodiedAgentInterface/generalization'
        if (gate['manifest_sha256']!=sha(root/'confirmation/manifest.json') or
            gate['executor_source_sha256']!=sha(executor/'fixed_executor.py') or
            gate['bridge_source_sha256']!=sha(executor/'libero_campaign.py')):
            raise ValueError('qualified executor, bridge/reset protocol or calibration changed')
        frozen_inputs=read(c['confirmation_frozen_input_audit'])
        if not frozen_inputs['passed'] or frozen_inputs['confirmation_manifest_sha256']!=sha(root/'confirmation/manifest.json'):
            raise ValueError('settled-scene / decoded-image confirmation audit differs')
        freeze(root/'confirmation/frozen_input_audit.json',frozen_inputs)
        # Historical diagnosis can run alongside source baseline preparation.
        def diagnostics():
            if c.get('diagnostics_reuse_directory'):
                # Loss-only changes do not require rerunning historical neural
                # controls. Verify every inference implementation and every
                # source condition's immutable configuration before reuse.
                parent=Path(c['diagnostics_reuse_directory']).parent
                previous=read(parent/'config.json')
                critical=['test_regr/JointEmbodiedAgentInterface/models.py',
                    'test_regr/JointEmbodiedAgentInterface/transfer/native_baseline.py',
                    'test_regr/JointEmbodiedAgentInterface/transfer/matched_baseline/evaluate.py',
                    'test_regr/JointEmbodiedAgentInterface/transfer/matched_baseline/protocol.py',
                    'test_regr/JointEmbodiedAgentInterface/transfer/anonymous_grounding/planner_binding/binding.py',
                    'test_regr/JointEmbodiedAgentInterface/transfer/anonymous_grounding/planner_binding/agent.py',
                    'test_regr/JointEmbodiedAgentInterface/generalization/shared_constraints.py']
                for name in critical:
                    if sha(Path(c['source'])/name)!=sha(Path(previous['source'])/name):
                        raise ValueError('diagnostic inference code changed: '+name)
                def intervention_fingerprint(source):
                    path=Path(source)/'test_regr/JointEmbodiedAgentInterface/transfer/repair/diagnose.py'
                    nodes=ast.parse(path.read_text()).body
                    return {node.name:hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
                        for node in nodes if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))
                        and node.name in ('intervention','diagnostic_binding')}
                intervention_code=intervention_fingerprint(c['source'])
                if intervention_code!=intervention_fingerprint(previous['source']):
                    raise ValueError('diagnostic intervention implementation changed')
                if sha(c['historical_config'])!=sha(previous['historical_config']):raise ValueError('historical controls changed')
                # The surface lexer is independently repaired and rerun. It is
                # not used by the four no-DFA graph-head interventions.
                if c.get('repaired_diagnostic_reuse_directory'):
                    repaired=Path(c['repaired_diagnostic_reuse_directory']);owner=repaired.parents[1]
                    old=read(owner/'config.json')
                    for name in [*critical,'test_regr/JointEmbodiedAgentInterface/transfer/repair/surface.py']:
                        if sha(Path(c['source'])/name)!=sha(Path(old['source'])/name):
                            raise ValueError('repaired-policy inference source changed: '+name)
                    jobs={}
                    for name in ('diagnostic-base-repaired','diagnostic-base-repaired-native'):
                        path=owner/'jobs'/(name+'.json')
                        while not path.exists() or read(path).get('exit_code') is None:
                            if read(owner/'status.json').get('stage')=='failed':raise ValueError('repaired control supervisor failed')
                            time.sleep(5)
                        if read(path)['exit_code']!=0:raise ValueError('repaired control failed')
                        jobs[str(path)]=sha(path)
                    freeze(root/'repaired_diagnostic_reuse.json',{'passed':True,'completed_jobs':jobs,
                        'surface_sha256':sha(Path(c['source'])/'test_regr/JointEmbodiedAgentInterface/transfer/repair/surface.py')})
                else:
                    args=['--config',root/'config.json','--manifest',c['historical_manifest'],'--seed','42']
                    run(root,'surface-replay',model_command(root,c,'surface-replay',MODULE+'.diagnose',
                        args+['--arm','replay','--output',root/'diagnostics']),{'physical_gpu':None})
                    out=root/'diagnostics/base-repaired';name='diagnostic-base-repaired'
                    gpu_job(root,c,name,MODULE+'.diagnose',args+['--arm','base-repaired','--output',out])
                    folder=out/'evaluation/base/shared-verified'
                    run(root,name+'-native',native_command(root,c,name+'-native',c['historical_manifest'],
                        c['historical_calibration'],folder/'predictions.jsonl',folder/'execution'),
                        {'physical_gpu':None,'horizon':1000,'surface_repair':True})
                hashes={}
                for arm in ('initialization-off','initialization-B','B-off','B-B'):
                    for suffix in ('','-native'):
                        path=parent/'jobs'/('diagnostic-'+arm+suffix+'.json')
                        while not path.exists() or read(path).get('exit_code') is None:
                            status=read(parent/'status.json')
                            if status.get('stage')=='failed':raise ValueError('historical diagnostic supervisor failed: '+str(status))
                            time.sleep(5)
                        job=read(path)
                        if job['exit_code']!=0:raise ValueError('reused diagnostic condition failed: '+str(path))
                        hashes[str(path)]=sha(path)
                freeze(root/'diagnostic_reuse.json',{'passed':True,'inference_source_hashes':{
                    name:sha(Path(c['source'])/name) for name in critical},'completed_jobs':hashes,
                    'intervention_function_ast_sha256':intervention_code,
                    'historical_configuration_sha256':sha(c['historical_config'])})
                return
            args=['--config',root/'config.json','--manifest',c['historical_manifest'],'--seed','42']
            run(root,'surface-replay',model_command(root,c,'surface-replay',MODULE+'.diagnose',
                args+['--arm','replay','--output',root/'diagnostics']),{'physical_gpu':None})
            for arm in ('base-repaired','initialization:off','initialization:B','B:off','B:B'):
                name='diagnostic-'+arm.replace(':','-');out=root/'diagnostics'/arm.replace(':','-')
                gpu_job(root,c,name,MODULE+'.diagnose',args+['--arm',arm,'--output',out])
                model='base' if arm=='base-repaired' else arm.split(':')[0]
                policy='shared-verified' if arm=='base-repaired' else 'none'
                folder=out/'evaluation'/model/policy
                run(root,name+'-native',native_command(root,c,name+'-native',c['historical_manifest'],
                    c['historical_calibration'],folder/'predictions.jsonl',folder/'execution'),
                    {'physical_gpu':None,'horizon':1000,'diagnostic_intervention':True})
        with ThreadPoolExecutor(max_workers=2) as pool:
            diagnostic=pool.submit(diagnostics)
            if c.get('preflight_reuse_root'):
                previous=Path(c['preflight_reuse_root']);old=read(previous/'config.json')
                for name in ('recipe','model','model_image','source_observations','features','split_manifest',
                             'original_configuration','model_source','reference_root','reference_audit','decoder_hidden_size'):
                    if c[name]!=old[name]:raise ValueError('preflight input/configuration changed: '+name)
                critical=['test_regr/JointEmbodiedAgentInterface/models.py',
                    'test_regr/JointEmbodiedAgentInterface/transfer/repair/common.py',
                    'test_regr/JointEmbodiedAgentInterface/transfer/repair/objective.py',
                    'test_regr/JointEmbodiedAgentInterface/transfer/repair/alignment.py',
                    'test_regr/JointEmbodiedAgentInterface/transfer/repair/protocol.py',
                    'test_regr/JointEmbodiedAgentInterface/transfer/anonymous_grounding/train.py',
                    'test_regr/JointEmbodiedAgentInterface/transfer/source_transform.py',
                    'test_regr/JointEmbodiedAgentInterface/transfer/matched_baseline/protocol.py']
                for name in critical:
                    if sha(Path(c['source'])/name)!=sha(Path(old['source'])/name):raise ValueError('preflight model/loss/input code changed: '+name)
                def main_ast(source):
                    path=Path(source)/'test_regr/JointEmbodiedAgentInterface/transfer/repair/train.py'
                    return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main'),include_attributes=False)
                if main_ast(c['source'])!=main_ast(old['source']):raise ValueError('preflight execution path changed')
                proofs={}
                for arm in ('native','joint'):
                    for name in ('preflight.json','alignment.json'):
                        path=previous/'preflight'/arm/name;proof=read(path)
                        if not proof['passed'] or proof['optimizer_updates']!=0:raise ValueError('previous preflight not qualified')
                        freeze(root/'preflight'/arm/name,proof);proofs[str(path)]=sha(path)
                freeze(root/'preflight_reuse.json',{'passed':True,'source_proofs':proofs,
                    'model_loss_input_source_hashes':{name:sha(Path(c['source'])/name) for name in critical},
                    'reason':'Only the missing standard-library JSON import for logging changed; preflight model/loss/input path is identical'})
            else:
                for arm in ('native','joint'):
                    preflight=train_args(root,c,arm)
                    preflight[preflight.index('--output')+1]=root/'preflight'/arm
                    gpu_job(root,c,'preflight-'+arm,MODULE+'.train',preflight+['--preflight'])
            save(root/'status.json',{'stage':'frozen-native-base-source-validation','updated_utc':utc()})
            baseline=pool.submit(gpu_job,root,c,'native-base',MODULE+'.train',train_args(root,c,'native',True))
            # Freeze the untouched native initialization before either arm's
            # first update. Its independent validation can then run alongside
            # the two fixed recipes; no validation result changes training.
            while not (root/'native_base/initialization.json').exists():
                if baseline.done():baseline.result();raise ValueError('native Base initialization missing')
                time.sleep(5)
            save(root/'status.json',{'stage':'two-fresh-four-epoch-source-arms','updated_utc':utc()})
            def train(arm):gpu_job(root,c,'train-'+arm,MODULE+'.train',train_args(root,c,arm))
            # Each training job starts independently from pretrained weights.
            with ThreadPoolExecutor(max_workers=2) as training:
                for future in as_completed([training.submit(train,arm) for arm in ('native','joint')]):future.result()
            baseline.result()
            diagnostic.result()
        run(root,'diagnostic-analysis',model_command(root,c,'diagnostic-analysis',MODULE+'.diagnostic_analysis',
            ['--root',root]),{'physical_gpu':None,'diagnostic_interventions':True})
        systems=freeze_selection(root)
        save(root/'status.json',{'stage':'fixed-eight-condition-confirmation','updated_utc':utc()})
        errors=[];execution=[]
        with ThreadPoolExecutor(max_workers=2) as planning,ThreadPoolExecutor(max_workers=2) as physical:
            def condition(system,policy):
                folder=root/'evaluation'/system/policy;name='confirm-'+system+'-'+policy
                gpu_job(root,c,name,MODULE+'.evaluate',['--config',root/'config.json','--arm',system,
                    '--checkpoint',systems[system]['path'],'--manifest',root/'confirmation/manifest.json',
                    '--output',folder,'--policy',policy,'--seed','42'])
                execution.append(physical.submit(run,root,name+'-native',native_command(root,c,name+'-native',
                    root/'confirmation/manifest.json',root/'confirmation/calibration_reuse.json',
                    folder/'predictions.jsonl',folder/'execution'),{'physical_gpu':None,'horizon':1000}))
            for future in as_completed([planning.submit(condition,s,p) for s in systems for p in ('none','shared-verified')]):
                try:future.result()
                except Exception as exc:errors.append(str(exc))
            for future in execution:
                try:future.result()
                except Exception as exc:errors.append(str(exc))
        if errors:raise RuntimeError('completed conditions preserved: '+'; '.join(errors))
        run(root,'analysis',model_command(root,c,'analysis',MODULE+'.analysis',['--root',root]),{'physical_gpu':None})
        freeze(root/'complete.json',{'campaign':'two four-epoch arms and fixed confirmation only',
            'results_sha256':sha(root/'results.json'),'no_executor_repairs':True,'no_rlbench':True})
        save(root/'status.json',{'stage':'complete-awaiting-google-report-publication','updated_utc':utc()})
    except Exception as exc:
        save(root/'status.json',{'stage':'failed','error':str(exc),'updated_utc':utc()});raise


if __name__=='__main__':main()
