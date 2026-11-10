"""Freeze RLBench resets and calibrate/evaluate predicted plans in isolation.

Run directly using rlbench_runtime.Dockerfile; no training/model dependencies.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image,ImageDraw

try:
    from .rlbench_bridge import RLBenchBridge,pose_error,CONTROL_PROFILE
    from .rlbench_executor import RLBenchExecutor
    from .rlbench_next_test import build_plan,DEFAULT_CONFIG
    from .libero_protocol import wilson
except ImportError:
    from rlbench_bridge import RLBenchBridge,pose_error,CONTROL_PROFILE
    from rlbench_executor import RLBenchExecutor
    from rlbench_next_test import build_plan,DEFAULT_CONFIG
    from libero_protocol import wilson


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,sort_keys=True,indent=2,default=lambda x:np.asarray(x).tolist())+'\n'
    if path.exists() and json.loads(path.read_text())!=json.loads(text):raise ValueError('frozen artifact differs: '+str(path))
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(text);temporary.replace(path)


def runtime_provenance():
    import importlib.metadata as metadata
    result={'rlbench_commit':subprocess.check_output(['git','-C','/opt/RLBench','rev-parse','HEAD'],text=True).strip(),
            'pyrep_commit':subprocess.check_output(['git','-C','/opt/PyRep','rev-parse','HEAD'],text=True).strip(),
            'coppelia_archive':Path('/opt/coppelia_archive.sha256').read_text().strip(),
            'packages':{n:metadata.version(n) for n in ('numpy','scipy','PyRep','rlbench','Pillow')}}
    if result['rlbench_commit']!='02720bba4c73fe02eb75df946b8791b806028a9d':raise ValueError('RLBench source drift')
    return result


def image_inputs(bridge,folder):
    from pyrep.objects.shape import Shape
    observation=bridge.task.get_observation()
    image=Image.fromarray(observation.front_rgb);numbered=image.copy();draw=ImageDraw.Draw(numbered)
    visible=[];mask=observation.front_mask
    for index,entity in enumerate(bridge.entities):
        handles=[Shape(name).get_handle() for name in entity['parts']]
        ys,xs=np.where(np.isin(mask,handles))
        if len(xs)<8:continue
        x,y=int(np.median(xs)),int(np.median(ys))
        draw.rectangle((x-2,y-2,x+16,y+14),fill='white',outline='black');draw.text((x,y),str(index),fill='black')
        visible.append(index)
    image.save(folder/'raw.png');numbered.save(folder/'numbered.png')
    return [str(folder/'raw.png'),str(folder/'numbered.png')],visible


def public_input(instruction,entities,geometry,images,visible):
    return {'instruction':instruction,'entities':[f'entity_{i:03d}' for i in range(len(entities))],
            'movable':[f'entity_{i:03d}' for i,e in enumerate(entities) if e['movable']],
            'positions':{f'entity_{i:03d}':row['center'] for i,row in enumerate(geometry)},
            'images':images,'visible_entity_ids':visible}


def prepare(args):
    plan=build_plan(json.loads(Path(args.config).read_text()))
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    provenance=runtime_provenance()
    save(output/'plan.json',plan)
    save(output/'runtime.json',provenance)
    records=[];bridge=RLBenchBridge(render=True)
    try:
        for episode in plan['episodes']:
            folder=output/episode['split']/episode['episode_id'];folder.mkdir(parents=True,exist_ok=True)
            if (folder/'record.json').exists():
                record=json.loads((folder/'record.json').read_text())
                if any(sha(p)!=h for p,h in record['files'].items()):raise ValueError('frozen reset changed')
                records.append(record);continue
            instruction=bridge.reset(episode['task'],episode['variation'],episode['seed'])
            native_state=bridge.snapshot_state()
            # Native dynamics can leave an open finger slightly beyond its
            # declared limit. Canonicalize every reset through the same restore
            # API before freezing both images and state; never filter episodes.
            save(folder/'native_state_before_canonicalization.json',native_state)
            bridge.restore_state(native_state)
            save(folder/'state.json',bridge.snapshot_state())
            geometry=[bridge.geometry(i) for i in range(len(bridge.entities))]
            images,visible=image_inputs(bridge,folder)
            # Explicit allowlist: native names, task IDs, goal objects, sensors,
            # registered graspable-object lists and custom low_dim_state stay private.
            public=public_input(instruction,bridge.entities,geometry,images,visible)
            save(folder/'generation_input.json',public)
            save(folder/'scoring_private.json',{'entities':bridge.entities,'reference':bridge.calibration_reference()})
            files={str(folder/name):sha(folder/name) for name in ('state.json','native_state_before_canonicalization.json','generation_input.json','scoring_private.json','raw.png','numbered.png')}
            record={**episode,'state':str(folder/'state.json'),'generation_input':str(folder/'generation_input.json'),
                    'scoring_private':str(folder/'scoring_private.json'),'files':files}
            save(folder/'record.json',record);records.append(record)
            print(f'[rlbench-prepare] {len(records)}/200 {episode["episode_id"]}',flush=True)
    finally:bridge.close()
    manifest={'version':2,'episodes':records,'runtime':provenance,'plan_sha256':sha(output/'plan.json'),
              'counts':plan['counts'],'reset':'canonical native-limit joint coordinates, frozen task poses/joints/colors; original native state archived; images rendered after canonicalization',
              'input_protocol':'anonymous all-task-physical-entity inventory with declared geometry and dynamic roles',
              'limitations':['Metadata-assisted planner transfer, not learned-controller transfer.',
                 'Executor uses explicitly authorized native proximity-gated attachment for the predicted entity only; no goal-selected graspable list.',
                 'RLBench rubbish distractors are static in the native asset; dynamic-role metadata can identify its only movable object.']}
    save(output/'manifest.json',manifest)


def execute(args):
    try:from .rlbench_reset_validation import validate_reset
    except ImportError:from rlbench_reset_validation import validate_reset
    from pyrep.objects.object import Object
    from pyrep.const import ObjectType
    manifest_path=Path(args.manifest);manifest=json.loads(manifest_path.read_text())
    if manifest['runtime']!=runtime_provenance():raise ValueError('simulator runtime differs from frozen resets')
    source=Path(__file__).parent
    settings={'manifest_sha256':sha(manifest_path),'executor_sha256':sha(source/'rlbench_executor.py'),
              'bridge_sha256':sha(source/'rlbench_bridge.py'),'runner_sha256':sha(__file__),
              'mode':args.mode,'horizon':1000,'grasp_protocol':'native_proximity_gated_attachment_predicted_entity_only',
              'control_profile':CONTROL_PROFILE,
              'source_sha256':{p.name:sha(p) for p in sorted([*source.glob('rlbench_*.py'),source/'libero_protocol.py'])},
              'predictions_sha256':sha(args.predictions) if args.mode=='evaluate' else None}
    plans=None
    if args.mode=='evaluate':
        gate=json.loads(Path(args.calibration).read_text())
        try:from .rlbench_qualification import qualify
        except ImportError:from rlbench_qualification import qualify
        verified=qualify([r['path'] for r in gate.get('runs',[])],manifest_path)
        if gate!=verified or not gate.get('promotion_passed'):raise ValueError('two-run executor promotion gate not passed')
        if gate['episodes']!=100 or gate['success_rate']<.9:raise ValueError('RLBench executor calibration below 90%')
        for key in ('manifest_sha256','executor_sha256','bridge_sha256','runner_sha256','horizon','grasp_protocol','control_profile','source_sha256'):
            if gate[key]!=settings[key]:raise ValueError('calibration protocol drift: '+key)
        predictions=[json.loads(s) for s in Path(args.predictions).read_text().splitlines()]
        plans={row['episode_id']:row['prediction'] for row in predictions}
        if len(plans)!=len(predictions):raise ValueError('duplicate predictions')
    split='calibration' if args.mode=='calibrate' else 'evaluation'
    records=[e for e in manifest['episodes'] if e['split']==split]
    if len(records)!=100 or len({e['episode_id'] for e in records})!=100:raise ValueError('require exactly 100 unique episodes')
    if plans is not None and set(plans)!={e['episode_id'] for e in records}:raise ValueError('prediction coverage differs')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);save(output/'settings.json',settings)
    path=output/'episodes.jsonl'
    prior=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    completed={row['episode_id']:row for row in prior}
    if len(completed)!=len(prior):raise ValueError('duplicate resumed episodes')
    if set(completed)-{e['episode_id'] for e in records}:raise ValueError('unexpected resumed episodes')
    bridge=RLBenchBridge(render=False)
    try:
        with path.open('a',buffering=1) as stream:
            for record in records:
                if record['episode_id'] in completed:continue
                if any(sha(p)!=h for p,h in record['files'].items()):raise ValueError('frozen episode changed')
                private=json.loads(Path(record['scoring_private']).read_text())
                state=json.loads(Path(record['state']).read_text())
                bridge.reset(record['task'],record['variation'],record['seed'],state=state,entities=private['entities'])
                restored={row['name']:row for row in bridge.snapshot_state()}
                robot_names={obj.get_name() for obj in bridge.env._robot.arm.get_objects_in_tree()}
                dummy_names={row['name'] for row in state if Object.get_object(row['name']).get_type()==ObjectType.DUMMY}
                reset_metrics=validate_reset(state,restored,robot_names=robot_names,
                    robot_base=bridge.env._robot.arm.get_name(),task_base=bridge.task._task.get_base().get_name(),
                    dummy_names=dummy_names,pose_error=pose_error)
                plan=private['reference'] if plans is None else plans[record['episode_id']]
                result=RLBenchExecutor(bridge,horizon=1000).execute(plan)
                result.update(episode_id=record['episode_id'],prediction=plan,success=bridge.native_success(),
                              **reset_metrics)
                stream.write(json.dumps(result)+'\n');completed[record['episode_id']]=result
                print(f'[rlbench-{args.mode}] {len(completed)}/100 success={sum(r["success"] for r in completed.values())}',flush=True)
    finally:bridge.close()
    if settings['source_sha256']!={p.name:sha(source/p.name) for p in sorted([*source.glob('rlbench_*.py'),source/'libero_protocol.py'])}:
        raise ValueError('source changed during native execution')
    successes=sum(r['success'] for r in completed.values())
    try:from .rlbench_qualification import assess
    except ImportError:from rlbench_qualification import assess
    assessment=assess(list(completed.values()),[r['episode_id'] for r in records]) if plans is None else None
    report={**settings,'episodes':len(completed),'successes':successes,'success_rate':successes/100,
            'success_ci95':wilson(successes,100),'single_run_criteria':assessment,
            'promotion_requires':'two fresh complete calibration runs'}
    save(output/'summary.json',report);print(json.dumps(report),flush=True)
    if assessment is not None and not assessment['passed']:
        raise RuntimeError('RLBench single-run promotion criteria failed; preserve qualified baseline')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','calibrate','evaluate'])
    parser.add_argument('--config',default=str(DEFAULT_CONFIG));parser.add_argument('--output',required=True)
    parser.add_argument('--manifest');parser.add_argument('--predictions');parser.add_argument('--calibration')
    args=parser.parse_args();prepare(args) if args.mode=='prepare' else execute(args)


if __name__=='__main__':main()
