"""Planning-side RLBench contracts, independent of the native simulator."""
import json
from pathlib import Path
from .libero_protocol import sha


def planning_context(public):
    names=public['entities']
    if names!=[f'entity_{i:03d}' for i in range(len(names))]:raise ValueError('anonymous entity ID ordering differs')
    if not set(public['movable']).issubset(names) or set(public['positions'])!=set(names):
        raise ValueError('observational entity coverage differs')
    return {'instruction':public['instruction']+'\nInitial observed object positions: '+json.dumps(public['positions'],sort_keys=True)
            +'\nDeclared dynamic entities: '+json.dumps(public['movable']),
            'entity_table':tuple(names),'images':tuple(public['images']),
            'generation_static_fixture_ids':[i for i,name in enumerate(names) if name not in public['movable']]}


def plan_validity(plan,public):
    held=None;violations=[]
    if not isinstance(plan,list):return {'valid_plan':False,'violations':[{'reason':'invalid_plan_schema'}]}
    for index,operation in enumerate(plan or []):
        skill=operation.get('name') if isinstance(operation,dict) else None
        if skill not in ('pick','place'):
            violations.append({'step':index,'reason':'unsupported_skill'});break
        role='target_entity_name' if skill=='pick' else 'target_container_name'
        params=operation.get('params',{})
        if not isinstance(params,dict):
            violations.append({'step':index,'reason':'invalid_pointer_or_schema'});break
        target=params.get(role)
        if set(params)!={role} or type(target) is not int or not 0<=target<len(public['entities']):
            violations.append({'step':index,'reason':'invalid_pointer_or_schema'});break
        if skill=='pick':
            if held is not None or public['entities'][target] not in public['movable']:
                violations.append({'step':index,'reason':'invalid_pick_precondition'});break
            held=target
        else:
            if held is None or target==held:
                violations.append({'step':index,'reason':'invalid_place_precondition'});break
            held=None
    if held is not None:violations.append({'reason':'object_still_held'})
    if not plan:violations.append({'reason':'empty_plan'})
    return {'valid_plan':not violations,'violations':violations}


def validate_calibration(manifest_path,calibration_path,source_directory):
    manifest_path=Path(manifest_path);source=Path(source_directory)
    gate=json.loads(Path(calibration_path).read_text())
    if gate.get('mode')!='calibrate' or gate.get('episodes')!=100 or gate.get('success_rate',0)<.9:
        raise ValueError('RLBench physical executor has not passed its 90/100 calibration gate')
    if gate['manifest_sha256']!=sha(manifest_path):raise ValueError('calibration manifest differs')
    for key,name in [('bridge_sha256','rlbench_bridge.py'),('executor_sha256','rlbench_executor.py'),('runner_sha256','rlbench_campaign.py')]:
        if gate[key]!=sha(source/name):raise ValueError('calibrated source differs: '+name)
    if (source/'rlbench_planning.py').exists():
        from .rlbench_qualification import qualify
        if not gate.get('promotion_passed'):raise ValueError('two-run contact-aware executor promotion required')
        if qualify([r['path'] for r in gate.get('runs',[])],manifest_path)!=gate:
            raise ValueError('promotion evidence changed')
        if gate.get('source_sha256')!={p.name:sha(p) for p in sorted([*source.glob('rlbench_*.py'),source/'libero_protocol.py'])}:
            raise ValueError('calibrated planning source differs')
    return gate
