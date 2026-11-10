import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_regr.JointEmbodiedAgentInterface.generalization.rlbench_bridge import RLBenchBridge
from test_regr.JointEmbodiedAgentInterface.generalization.rlbench_executor import RLBenchExecutor
from test_regr.JointEmbodiedAgentInterface.generalization.rlbench_planning import (
    matrix, stable_orientations, release_stable, thin_body, PLANNING_PROFILE,
    face_stability_required,
)
from test_regr.JointEmbodiedAgentInterface.generalization.rlbench_qualification import assess, qualify


def test_detector_is_side_effect_free_and_checks_only_predicted_object(monkeypatch):
    import sys
    queried=[]
    class Shape:
        def __init__(self,name):self.name=name
    monkeypatch.setitem(sys.modules,'pyrep.objects.shape',SimpleNamespace(Shape=Shape))
    bridge=RLBenchBridge.__new__(RLBenchBridge)
    bridge.entities=[{'movable':True,'primary_name':'wrong'},{'movable':True,'primary_name':'predicted'},{'movable':False}]
    def detect(shape):queried.append(shape.name);return shape.name=='predicted'
    class Gripper:
        _proximity_sensor=SimpleNamespace(is_detected=detect)
        def grasp(self,*args):pytest.fail('query must not attach')
    bridge.env=SimpleNamespace(_robot=SimpleNamespace(gripper=Gripper()))
    assert not bridge.detect_grasp(0) and bridge.detect_grasp(1) and not bridge.detect_grasp(2)
    assert queried==['wrong','predicted']


def test_native_release_trace_is_json_serializable(monkeypatch):
    import sys
    class Shape:
        def __init__(self,name):pass
        def get_matrix(self):return np.eye(4)
        def get_pose(self):return np.array([0.,0.,1.,0.,0.,0.,1.])
    monkeypatch.setitem(sys.modules,'pyrep.objects.shape',SimpleNamespace(Shape=Shape))
    bridge=RLBenchBridge.__new__(RLBenchBridge);bridge.entities=[{'primary_name':'body'}]
    bridge.env=SimpleNamespace(_robot=SimpleNamespace(gripper=SimpleNamespace(get_grasped_objects=lambda:[])))
    monkeypatch.setattr(bridge,'geometry',lambda _: {'dimensions':np.array([.08,.08,.03]),'low':np.array([0.,0.,.985])})
    sample=bridge.release_sample(0)
    assert json.loads(json.dumps(sample))==sample and not sample['attached']


@pytest.mark.parametrize('short',[0,1,2])
def test_stable_face_uses_geometry_axis_and_preserves_grasp_transform(short):
    dimensions=np.full(3,.08);dimensions[short]=.03
    original=matrix(np.r_[[.6,.1,1.3],Rotation.from_euler('xyz',[.5,.3,.7]).as_quat()])
    tip=matrix(np.r_[[.4,.2,1.5],Rotation.from_euler('xyz',[2.,.6,.3]).as_quat()])
    relative=np.linalg.inv(tip)@original
    rotations=list(stable_orientations(original,dimensions,[1.,0.,0.]))
    assert len(rotations)==12
    for tilt, rotation in rotations:
        axis=rotation[:,short]
        assert np.arccos(np.clip(abs(axis[2]),0,1))==pytest.approx(np.deg2rad(tilt))
        hand=rotation@relative[:3,:3].T
        assert np.allclose(hand@relative[:3,:3],rotation)
    assert thin_body({'dimensions':dimensions})
    assert not thin_body({'dimensions':[.05,.06,.07]})


def samples(*, rolling=False, tilted=False, floating=False):
    rows=[]
    for i in range(30):
        rows.append({'pose':[i*.004 if rolling else 0.,0.,1.03,0.,0.,0.,1.],
                     'bottom':1.02 if floating else 1.001,'face_tilt':np.pi/2 if tilted else 0.})
    return rows


def test_release_checks_motion_support_and_flat_face_independently_of_goal():
    assert release_stable(samples(),support=1.,require_face=True)
    assert not release_stable(samples(rolling=True),support=1.,require_face=True)
    assert not release_stable(samples(tilted=True),support=1.,require_face=True)
    assert not release_stable(samples(floating=True),support=1.,require_face=True)
    assert not release_stable(samples()[:10],support=1.)
    rotating=samples()
    for i,row in enumerate(rotating):row['pose'][3:]=Rotation.from_rotvec([0.,0.,i*.02]).as_quat().tolist()
    assert not release_stable(rotating,support=1.)
    attached=samples()
    for row in attached:row['attached']=True
    assert not release_stable(attached,support=1.)


def test_no_lift_or_attachment_when_native_detector_rejects(monkeypatch):
    bridge=SimpleNamespace(entities=[{}],geometry=lambda _: {'movable':True,'center':np.array([0.,0.,.03]),
        'low':np.zeros(3),'high':np.array([.06,.06,.06]),'yaw':0.,'horizontal_span':.06},
        base_position=lambda:np.array([-1.,0.,0.]),tip_pose=lambda:np.array([0.,0.,.04,0.,1.,0.,0.]),
        detect_grasp=lambda _:False,grasp_clearance=lambda _: .01,tick=lambda:None,
        try_grasp=lambda _:pytest.fail('must not attach outside native gate'))
    executor=RLBenchExecutor(bridge);phases=[]
    monkeypatch.setattr(executor,'move',lambda *a,**k:phases.append(executor.phase))
    monkeypatch.setattr(executor,'gripper',lambda *_:None)
    result=executor.execute([{'name':'pick','params':{'target_entity_name':0}}])
    assert 'lift' not in phases and result['holding_at_end'] is None
    assert result['steps']<=1000


def calibration_rows():
    rows=[]
    for case in range(10):
        for reset in range(10,20):
            rows.append({'episode_id':f'task/variation{case:02d}/reset{reset}',
                         'success':True,'termination_reason':'completed','holding_at_end':None})
    return rows


def test_promotion_does_not_hide_a_case_regression_or_incomplete_plan():
    rows=calibration_rows();ids=[r['episode_id'] for r in rows]
    assert assess(rows,ids)['passed']
    rows[0]['success']=False;rows[1]['success']=False
    assert not assess(rows,ids)['passed']  # 98/100 still fails per-case guard.
    rows=calibration_rows()
    for i in range(10):rows[i]['termination_reason']='horizon'
    assert not assess(rows,ids)['passed']  # Native goals cannot hide noncompletion.
    with pytest.raises(ValueError,match='coverage'):assess(rows[:-1]+[rows[0]],ids)
    with pytest.raises(ValueError,match='distinct'):qualify(['a','a'],'unused')


def test_two_run_evidence_is_recomputed_and_pins_process_and_settings(tmp_path):
    from test_regr.JointEmbodiedAgentInterface.generalization.rlbench_qualification import sha
    rows=calibration_rows()
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'episodes':[{'episode_id':r['episode_id'],'split':'calibration'} for r in rows]}))
    settings={'mode':'calibrate','manifest_sha256':sha(manifest),'source_sha256':{'planner.py':'abc'}}
    paths=[]
    for index in range(2):
        path=tmp_path/f'run{index}';path.mkdir();paths.append(path)
        (path/'settings.json').write_text(json.dumps(settings))
        (path/'summary.json').write_text(json.dumps({**settings,'episodes':100,'successes':100}))
        (path/'episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        job={'container':f'run{index}','pid':index+1,'fresh':True,'exit_code':0,'image_id':'frozen',
             'source_hashes':settings['source_sha256'],'manifest_sha256':sha(manifest)}
        path.with_name(path.name+'_job.json').write_text(json.dumps(job))
    report=qualify(paths,manifest)
    assert report['promotion_passed'] and len(report['runs'])==2
    rows[0]['success']=False;rows[1]['success']=False
    (paths[1]/'episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError,match='summary differs'):qualify(paths,manifest)
    (paths[1]/'summary.json').write_text(json.dumps({**settings,'episodes':100,'successes':98}))
    assert not qualify(paths,manifest)['promotion_passed']


def test_lookahead_keeps_predicted_ids_and_rejects_invalid_pointer(monkeypatch):
    executor=RLBenchExecutor(SimpleNamespace(entities=[{},{}]));calls=[]
    monkeypatch.setattr(executor,'pick',lambda index,destination_index=None:calls.append((index,destination_index)))
    monkeypatch.setattr(executor,'place',lambda index:None)
    plan=[{'name':'pick','params':{'target_entity_name':0}}, {'name':'place','params':{'target_container_name':1}}]
    assert executor.execute(plan)['termination_reason']=='completed' and calls==[(0,1)]
    calls.clear();plan[1]['params']['target_container_name']=True
    assert 'invalid destination' in executor.execute(plan)['termination_reason'] and not calls


@pytest.mark.parametrize('cached',[False,True])
def test_placement_executes_checked_wrist_route_without_forcing_downward_transport(monkeypatch,cached):
    desired=Rotation.from_euler('y',np.pi/2).as_quat()
    pose=np.r_[[0.,0.,1.],Rotation.from_euler('y',np.pi/3).as_quat()]
    requests=[];moves=[];rotations=[]
    def request(operation,index,destination):
        requests.append((operation,index,destination))
        if operation=='place':
            result={'placement':{'orientation':desired.tolist(),'reorientation_waypoints':[
                {'position':[0.,0.,1.],'orientation':desired.tolist()},
                {'position':[.7,0.,1.],'orientation':desired.tolist()}]}}
            if cached:
                for waypoint in result['placement']['reorientation_waypoints']:waypoint['joint_path']=[[0.],[.1]]
                result['placement']['preplace_joint_path']=[[.1],[.2]]
            return result
        return {'release':[1.,0.,1.05],'approach':[.8,0.,1.05],'orientation':desired.tolist()}
    bridge=SimpleNamespace(entities=[{},{}],tip_pose=lambda:pose.copy(),planning_request=request,
                           geometry=lambda _: {'center':pose[:3]-[0,0,.05],'low':np.array([0.,0.,.7])},
                           placement_route=lambda *a,**k:(np.array([1.,0.,1.]),np.array([.8,0.,1.])))
    executor=RLBenchExecutor(bridge);executor.held=0
    clearance_phases=[]
    def clearance(index):clearance_phases.append(executor.phase);return True
    bridge.release_clearance=clearance;bridge.support_height=lambda *a:1.
    bridge.release_sample=lambda _: {'pose':[1.,0.,1.02,0.,0.,0.,1.],'bottom':1.,'face_tilt':0.,'attached':False}
    bridge.requires_stable_face=lambda _:False;bridge.tick=lambda:None
    used=[]
    def move(position,*,orientation,**kwargs):
        used.append(kwargs.get('joint_path'))
        moves.append((executor.phase,np.asarray(orientation).copy()));pose[:3]=position;pose[3:]=orientation
    def reorient(orientation):rotations.append(orientation);pose[3:]=orientation
    monkeypatch.setattr(executor,'move',move);monkeypatch.setattr(executor,'reorient',reorient)
    monkeypatch.setattr(executor,'gripper',lambda *_:None);monkeypatch.setattr(executor,'settle',lambda *_:None)
    executor.place(1)
    assert requests==[('place',0,1),('verify_placement',0,1)] and len(rotations)==(0 if cached else 1)
    if cached:assert used[:3]==[[[0.],[.1]],[[0.],[.1]],[[.1],[.2]]]
    assert all(abs(np.dot(q,desired))>.9999 for _,q in moves)
    assert executor.held is None
    assert clearance_phases==['retreat'] and executor.steps==30


def test_source_fingerprint_includes_planning_and_qualification():
    from pathlib import Path
    from test_regr.JointEmbodiedAgentInterface.generalization import rlbench_campaign
    source=Path(rlbench_campaign.__file__).parent
    names={p.name for p in source.glob('rlbench_*.py')}
    assert {'rlbench_planning.py','rlbench_qualification.py','rlbench_bootstrap.py'}<=names


def test_payload_containment_does_not_require_outside_palm_between_shelf_planes(monkeypatch):
    import sys
    from test_regr.JointEmbodiedAgentInterface.generalization import rlbench_bridge
    class Shape:
        def __init__(self,name):pass
        def get_bounding_box(self):return [-.2,.2,-.3,.3,0.,.3]
        def get_matrix(self):
            result=np.eye(4);result[2,3]=1.;return result
        def get_mesh_data(self):return [],[],[]
    monkeypatch.setitem(sys.modules,'pyrep.objects.shape',SimpleNamespace(Shape=Shape))
    def rays(v,t,o,d):
        if d[2]>.9:return np.array([.22])
        if d[2]<-.9:return np.array([.08])
        return np.array([])
    monkeypatch.setattr(rlbench_bridge,'ray_hits',rays)
    bridge=RLBenchBridge.__new__(RLBenchBridge);bridge.env=object();bridge.entities=[{'primary_name':'shelf'}]
    monkeypatch.setattr(bridge,'assembly_points',lambda:pytest.fail('a palm outside the opening must not be forced inside its height interval'))
    body={'center':np.array([0.,0.,1.1]),'low':np.array([-.05,-.05,1.05]),'high':np.array([.05,.05,1.15])}
    target,approach=bridge.placement_route(0,body,require_fit=True)
    assert 1.006<=target[2]-.05<=1.02 and approach[0]<-.2
    # Spatial hand/fixture collision checks remain mandatory in ShadowPlanner.


@pytest.mark.parametrize('drift',[False,True])
def test_shadow_request_filters_sensors_and_detects_live_state_drift(monkeypatch,drift):
    import sys
    from enum import IntEnum
    class ObjectType(IntEnum):SHAPE=0;JOINT=1;SENSOR=2
    types={'body':ObjectType.SHAPE,'joint':ObjectType.JOINT,'goal_sensor':ObjectType.SENSOR}
    class Object:
        @staticmethod
        def get_object(name):return SimpleNamespace(get_type=lambda:types[name])
    monkeypatch.setitem(sys.modules,'pyrep.const',SimpleNamespace(ObjectType=ObjectType))
    monkeypatch.setitem(sys.modules,'pyrep.objects.object',SimpleNamespace(Object=Object))
    monkeypatch.setitem(sys.modules,'pyrep.backend',SimpleNamespace(sim=SimpleNamespace(simGetSimulationTime=lambda:1.)))
    bridge=RLBenchBridge.__new__(RLBenchBridge)
    bridge.scene=('scene',0,42);bridge.entities=[{'primary_name':'body'}];bridge.observed_support_axes={0:2}
    state=[{'name':name,'pose':[0.,0.,0.,0.,0.,0.,1.]} for name in types]
    monkeypatch.setattr(bridge,'snapshot_state',lambda:copy.deepcopy(state))
    bridge.env=SimpleNamespace(_robot=SimpleNamespace(gripper=SimpleNamespace(get_grasped_objects=lambda:[])))
    requests=[]
    def request(value):
        requests.append(value)
        if drift:state[0]['pose'][0]=.01
        return {'placement':None}
    bridge.planner=SimpleNamespace(request=request)
    if drift:
        with pytest.raises(RuntimeError,match='changed live state'):bridge.planning_request('place',0,None)
    else:
        assert bridge.planning_request('place',0,None)=={'placement':None}
        # A failed grasp may roll the body. Retrying must preserve its ORIGINAL
        # support face, rather than blessing an edge as the new stable target.
        bridge.planning_request('grasp',0,None)
        assert requests[-1]['support_axis']==2
    assert [r['name'] for r in requests[0]['state']]==['body','joint']
    assert not {'reference','goals','waypoints','success'} & requests[0].keys()


def test_observed_support_face_for_tall_box_is_preserved():
    body=np.eye(4)
    assert not face_stability_required({'dimensions':[.03,.08,.15]},2)
    assert face_stability_required({'dimensions':[.08,.08,.03]},2)
    assert face_stability_required({'dimensions':[.03,.08,.15]},0)
    for tilt,rotation in stable_orientations(body,[.04,.05,.15],[1.,0.,0.],support_axis=2):
        assert abs(rotation[2,2])==pytest.approx(np.cos(np.deg2rad(tilt)))


def test_rim_search_has_both_wrist_directions_and_horizontal_closing_axis():
    from test_regr.JointEmbodiedAgentInterface.generalization.rlbench_planning import ShadowPlanner
    bridge=SimpleNamespace(base_position=lambda:np.zeros(3),grasp_clearance=lambda q:.04)
    planner=ShadowPlanner(bridge)
    body={'center':np.array([.3,-.3,.765]),'low':np.array([.26,-.34,.75]),
          'high':np.array([.34,-.26,.78]),'horizontal_span':.08,'yaw':.2,'dimensions':[.08,.08,.14]}
    proposals=planner.grasp_proposals(body)
    radial=np.array([1.,-1.,0.])/np.sqrt(2)
    signed=[]
    for point,q,yaw in proposals[12:12+18]:
        rotation=Rotation.from_quat(q).as_matrix()
        assert abs(rotation[2,1])<1e-8, 'aligned closing axis keeps both fingers level'
        assert point[2]>=.79
        signed.append(np.dot(rotation[:,2],radial))
    assert min(signed)<-.8 and max(signed)>.8


def test_new_planner_cannot_use_legacy_single_run_gate(tmp_path):
    from test_regr.JointEmbodiedAgentInterface.generalization.rlbench_protocol import validate_calibration,sha
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    gate={'mode':'calibrate','episodes':100,'success_rate':.99,'manifest_sha256':sha(manifest)}
    for key,name in [('bridge_sha256','rlbench_bridge.py'),('executor_sha256','rlbench_executor.py'),('runner_sha256','rlbench_campaign.py')]:
        (tmp_path/name).write_text(name);gate[key]=sha(tmp_path/name)
    (tmp_path/'rlbench_planning.py').write_text('new planner')
    report=tmp_path/'calibration.json';report.write_text(json.dumps(gate))
    with pytest.raises(ValueError,match='two-run'):validate_calibration(manifest,report,tmp_path)


@pytest.mark.parametrize('obstacle',['hand','wrist','limit'])
def test_shadow_includes_gripper_base_but_rejects_wrist_collision_and_limits(monkeypatch,obstacle):
    import sys
    from test_regr.JointEmbodiedAgentInterface.generalization.rlbench_planning import ShadowPlanner
    class Shape:
        def __init__(self,name):self.name=name
        def get_name(self):return self.name
        def is_respondable(self):return True
        def check_collision(self,other):return other.get_name()==obstacle
    monkeypatch.setitem(sys.modules,'pyrep.objects.shape',SimpleNamespace(Shape=Shape))
    monkeypatch.setitem(sys.modules,'pyrep.const',SimpleNamespace(ObjectType=SimpleNamespace(SHAPE=0)))
    hand=Shape('hand');wrist=Shape('wrist');configurations=[]
    arm=SimpleNamespace(joints=[0],get_joint_intervals=lambda:([False],[[-1.,2.]]),
                        get_objects_in_tree=lambda **k:[hand,wrist],
                        set_joint_positions=lambda q,**k:configurations.append(q))
    def hand_tree(**kwargs):
        assert kwargs['exclude_base'] is False, 'gripper root must not be treated as a wrist link'
        return [hand]
    bridge=SimpleNamespace(env=SimpleNamespace(_robot=SimpleNamespace(arm=arm,gripper=SimpleNamespace(get_objects_in_tree=hand_tree))),
                           entities=[{'primary_name':'body','movable':True,'parts':['body']}],contact_entity=0,
                           path_to=lambda *a,**k:SimpleNamespace(_path_points=np.array([2. if obstacle=='limit' else .5])))
    planner=ShadowPlanner(bridge)
    if obstacle=='hand':planner.path([0,0,0],[0,0,0,1])
    else:
        with pytest.raises(ValueError,match='wrist' if obstacle=='wrist' else 'joint limit'):
            planner.path([0,0,0],[0,0,0,1])


@pytest.mark.parametrize('holding,hand_collision',[(False,False),(True,False),(False,True)])
def test_spatial_checks_allow_supported_unheld_body_but_protect_hand_and_carried_payload(monkeypatch,holding,hand_collision):
    import sys
    from test_regr.JointEmbodiedAgentInterface.generalization.rlbench_planning import ShadowPlanner
    class Shape:
        def __init__(self,name):self.name=name
        def get_name(self):return self.name
        def is_respondable(self):return True
        def check_collision(self,other):
            return other.name=='table' and (self.name=='body' or self.name=='hand' and hand_collision)
    monkeypatch.setitem(sys.modules,'pyrep.objects.shape',SimpleNamespace(Shape=Shape))
    monkeypatch.setitem(sys.modules,'pyrep.const',SimpleNamespace(ObjectType=SimpleNamespace(SHAPE=0)))
    hand=Shape('hand');arm=SimpleNamespace(joints=[0],get_joint_intervals=lambda:([False],[[-1.,2.]]),
        get_objects_in_tree=lambda **k:[hand],set_joint_positions=lambda q,**k:None)
    gripper=SimpleNamespace(get_objects_in_tree=lambda **k:[hand],get_grasped_objects=lambda:[Shape('body')] if holding else [])
    bridge=SimpleNamespace(env=SimpleNamespace(_robot=SimpleNamespace(arm=arm,gripper=gripper)),
        entities=[{'primary_name':'body','movable':True,'parts':['body']}],contact_entity=0,
        path_to=lambda *a,**k:SimpleNamespace(_path_points=np.array([.5])))
    planner=ShadowPlanner(bridge);planner.structures=[Shape('table')]
    if holding or hand_collision:
        with pytest.raises(ValueError,match='payload/structure' if holding else 'hand/structure'):
            planner.path([0,0,0],[0,0,0,1])
    else:planner.path([0,0,0],[0,0,0,1])
