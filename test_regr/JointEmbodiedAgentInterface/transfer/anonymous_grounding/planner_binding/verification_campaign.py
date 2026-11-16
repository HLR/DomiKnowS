"""Validation-before-confirmation orchestration; no optimizer or training jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

if not __package__:
    root=Path(sys.argv[sys.argv.index('--root')+1])
    sys.path.insert(0,str(root/'source/test_regr/JointEmbodiedAgentInterface'))
    __package__='transfer.anonymous_grounding.planner_binding'
from ...protocol import read,freeze,sha,episodes
from ...campaign import run,save,utc
from ..campaign import runtime_environment
from .epoch_campaign import unused_gpu

MODULE='test_regr.JointEmbodiedAgentInterface.transfer.anonymous_grounding.planner_binding'
POLICIES=('none','shared-verified','executor')


def load_control(root):
    path=root/'host_source/verification_control.py'
    if not path.exists():
        from . import verification_control
        return verification_control
    spec=importlib.util.spec_from_file_location(__package__+'.verification_control',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def wait_job(root,name):
    path=root/'jobs'/f'{name}.json'
    while not path.exists() or read(path).get('exit_code') is None:time.sleep(5)
    if read(path)['exit_code']!=0:raise RuntimeError('correctness gate job failed: '+name)


def wait_control(root):
    while not (root/'control_complete.json').exists():
        path=root/'control_status.json'
        if path.exists() and read(path)['stage']=='failed':raise RuntimeError('Epoch1 reproduction or control failed: '+read(path).get('error',''))
        time.sleep(5)


def wait_reproduction(root):
    """Confirmation admission needs reproduction, not finished independent controls."""
    policies=('none','shared-verified','full-metadata')
    while True:
        status=root/'control_status.json'
        if status.exists() and read(status)['stage']=='failed':
            raise RuntimeError('control correctness failed: '+read(status).get('error',''))
        paths=[root/'control_reproduction'/(p+'.json') for p in policies]
        if all(p.exists() for p in paths):
            for policy,path in zip(policies,paths):
                gate=read(path)
                if not gate.get('passed') or gate.get('policy')!=policy:
                    raise ValueError('archived reproduction certificate differs')
            return
        time.sleep(5)


def qualify_cohort(root,cohort,config):
    """Correct inherited descriptive labels without changing a single episode."""
    folder=root/'data'/cohort;old=folder/'manifest.json';manifest=read(old)
    states=list(range(30,35) if cohort=='validation' else range(45,50))
    rows=episodes(manifest)
    if {r['state_index'] for r in rows}!=set(states):raise ValueError('cohort states differ from preregistration')
    qualified={**manifest,'fresh_states':states,'state_indices':states,'cohort':cohort}
    destination=root/'cohorts'/cohort
    path=destination/'manifest_qualified.json';freeze(path,qualified)
    gate=read(folder/'calibration_reuse.json');base=Path(config['libero_base'])
    bridge=base/'evaluation_source/test_regr/JointEmbodiedAgentInterface/generalization/libero_campaign.py'
    if gate['manifest_sha256']!=sha(old) or gate['success_rate']<.9:raise ValueError('original calibration certificate differs')
    if gate['bridge_source_sha256']!=sha(bridge) or gate['executor_source_sha256']!=sha(bridge.parent/'fixed_executor.py'):
        raise ValueError('qualified executor or bridge changed')
    if [r for r in qualified['episodes'] if r['split']=='calibration']!=[r for r in manifest['episodes'] if r['split']=='calibration']:
        raise ValueError('calibration records changed')
    freeze(destination/'calibration_qualified.json',{**gate,'manifest_sha256':sha(path),
        'metadata_only_recertification':True,'prior_manifest_sha256':sha(old)})
    freeze(destination/'manifest_qualification.json',{'passed':True,'original_manifest_sha256':sha(old),
        'qualified_manifest_sha256':sha(path),'episode_records_unchanged':qualified['episodes']==manifest['episodes'],
        'state_indices':states,'reason':'replace inherited states40--44 description; no state, image, goal or reset changes'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    control=load_control(root);base_config=read(root/'control_config.json');setup=read(root/'setup.json')
    analysis_script=Path(base_config.get('analysis_script',Path(__file__).with_name('verification_analysis.py')))
    source=Path(base_config['source']);archive=Path(setup['archive'])
    prefix=['docker','run','-d','--ipc=host','--cpus=8','--memory=64g','-v','/home/auszok:/home/auszok','-v','/home/auszok/DomiKnowS:/workspace','-w',str(source)]
    for k,v in runtime_environment(source,base_config['planning_dir']).items():prefix+=['-e',k+'='+v]
    def cpu(name,arguments):
        command=prefix+['--name',root.name+'-'+name,'--entrypoint','/opt/dominows/venv/bin/python',base_config['model_image'],'-u',*arguments]
        run(root,name,command,{'physical_gpu':None})
    def prepare(cohort):
        out=root/'cohorts'/cohort;out.mkdir(parents=True,exist_ok=True)
        train=read(Path(setup['training_root'])/'training_common.json')
        qualify_cohort(root,cohort,base_config)
        config={**train,'output':str(out),'source':str(source),'external_manifest':str(out/'manifest_qualified.json')}
        config['external_manifest_sha256']=sha(config['external_manifest'])
        path=root/(cohort+'_inputs_config.json');freeze(path,config)
        (out/'text_priors').mkdir(exist_ok=True)
        target=out/'text_priors/manifest.json'
        if not target.exists():shutil.copyfile(Path(setup['training_root'])/'text_priors/manifest.json',target)
        name='features-'+cohort;job=root/'jobs'/f'{name}.json'
        if job.exists():run(root,name,read(job)['command'],{})
        else:
            index,uuid,lock=unused_gpu(root,name)
            try:
                command=prefix+['--name',root.name+'-'+name,'--gpus','device='+str(index),'-e','CUDA_VISIBLE_DEVICES='+uuid,
                    '--entrypoint','/opt/dominows/venv/bin/python',base_config['model_image'],'-u','-m',MODULE+'.prepare_inputs','--config',str(path)]
                run(root,name,command,{'physical_gpu':index,'gpu_uuid':uuid,'target_training':False})
            finally:lock.close()
        cfg={**base_config,'output':str(out),'external_manifest':config['external_manifest'],
             'fresh_features':str(out/'fresh_features/manifest.json'),'fresh_priors':str(out/'fresh_priors.json'),
             'calibration':str(out/'calibration_qualified.json'),'cohort':cohort,
             'input_hashes':dict(base_config['input_hashes'])}
        for f in (root/'data/split_audit.json',Path(cfg['external_manifest']),Path(cfg['fresh_features']),Path(cfg['fresh_priors']),Path(cfg['calibration']),out/'inputs_complete.json'):
            cfg['input_hashes'][str(f)]=sha(f)
        if cohort=='confirmation':
            gate=read(root/'validation_gate.json')
            if not gate['passed'] or gate['model_source_sha256']!=cfg['evaluation_source_sha256']:raise ValueError('candidate changed after validation')
            if gate.get('analysis_source_sha256')!=sha(analysis_script):raise ValueError('validation gate implementation changed')
            cfg['input_hashes'][str(root/'validation_gate.json')]=sha(root/'validation_gate.json')
        config_path=root/(cohort+'_config.json');freeze(config_path,cfg)
        proof={**read(root/'cohorts/control/prepared/manifest.json'),'config_sha256':sha(config_path),
               'external_ids':[e['episode_id'] for e in episodes(read(cfg['external_manifest']))],
               'reuse':'same frozen VLABench features and text prior; LIBERO features independently frozen per cohort'}
        freeze(out/'prepared/manifest.json',proof)
        return config_path
    try:
        save(root/'status.json',{'stage':'waiting-for-data-integrity-gates','updated_utc':utc()})
        wait_job(root,'native-preparation')
        if not read(root/'data/split_audit.json')['passed']:raise ValueError('new-scene split audit failed')
        save(root/'status.json',{'stage':'preparing-policy-validation-inputs','updated_utc':utc()})
        validation=prepare('validation')
        save(root/'status.json',{'stage':'policy-validation','updated_utc':utc()})
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(control.run_model,root,validation,'validation-epoch1-'+policy,'epoch-1',policy,
                root/'validation/epoch-1'/policy,['libero'],True) for policy in POLICIES]
            errors=[]
            for f in as_completed(jobs):
                try:f.result()
                except Exception as exc:errors.append(str(exc))
            if errors:raise RuntimeError('; '.join(errors))
        cpu('validation-gate',[str(analysis_script),'validation','--root',str(root),'--config',str(validation)])
        if not read(root/'validation_gate.json')['passed']:raise ValueError('candidate qualification failed')
        wait_reproduction(root)
        save(root/'status.json',{'stage':'freezing-confirmation-inputs','updated_utc':utc()})
        confirmation=prepare('confirmation');c=read(confirmation)
        freeze(root/'confirmation_admission.json',{'passed':True,'validation_gate_sha256':sha(root/'validation_gate.json'),
            'preregistration_sha256':sha(root/'preregistration.json'),'confirmation_manifest_sha256':sha(c['external_manifest']),
            'model_source_sha256':c['evaluation_source_sha256'],'selection':'initialization versus preselected Epoch1 only'})
        save(root/'status.json',{'stage':'confirmation-planning-and-native-execution','updated_utc':utc()})
        with ThreadPoolExecutor(max_workers=2) as gpu,ThreadPoolExecutor(max_workers=2) as native:
            def model(snapshot,policy):
                out=root/'confirmation'/snapshot/policy
                control.run_model(root,confirmation,'confirm-'+snapshot+'-'+policy,snapshot,policy,out,['libero'],True)
                return snapshot,policy,out
            futures=[gpu.submit(model,snapshot,policy) for snapshot in ('initialization','epoch-1') for policy in POLICIES]
            executions=[];errors=[]
            for f in as_completed(futures):
                try:snapshot,policy,out=f.result()
                except Exception as exc:errors.append(str(exc));continue
                name='confirm-execution-'+snapshot+'-'+policy
                command=control.native_command(root,c,name,c['external_manifest'],c['calibration'],out/'libero/predictions.jsonl',out/'libero/execution')
                executions.append(native.submit(run,root,name,command,{'physical_gpu':None,'executor':'unchanged qualified bridge; horizon 1000'}))
            for f in executions:
                try:f.result()
                except Exception as exc:errors.append(str(exc))
            if errors:raise RuntimeError('; '.join(errors))
        wait_control(root)
        cpu('final-analysis',[str(analysis_script),'report','--root',str(root),'--config',str(confirmation)])
        save(root/'status.json',{'stage':'complete','updated_utc':utc()})
    except Exception as exc:
        save(root/'status.json',{'stage':'blocked-correctness','error':str(exc),'updated_utc':utc()});raise


if __name__=='__main__':main()
