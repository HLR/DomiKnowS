"""Counterexamples for physical support, grasp retry and checked-path execution."""
import itertools
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ..generalization.rlbench_geometry import world_vertices,symmetry_axis,separating_support_contact
from ..generalization.rlbench_bridge import RLBenchBridge,transformed_bounds
from ..generalization.rlbench_executor import RLBenchExecutor,ContactMotionStop
from ..generalization.rlbench_planning import ShadowPlanner,face_stability_required,aligned_shelf_rotations


@pytest.mark.parametrize('failure',[False,True])
def test_native_mesh_frees_native_buffers_not_first_triangle_index(monkeypatch,failure):
    from ..generalization.rlbench_geometry import native_mesh
    buffers=[list(range(18)),[5,3,2],[0.,0.,1.]*3];freed=[]
    class FFI:
        NULL=None
        def new(self,kind):return [None] if '**' in kind else [0]
        def cast(self,kind,pointer):
            assert any(pointer is b for b in buffers), 'must never free integer triangle index 5'
            return pointer
    def query(handle,v,nv,t,nt,n):
        assert handle==42
        v[0],nv[0],t[0],nt[0],n[0]=buffers[0],18,buffers[1],3,buffers[2]
        return -1 if failure else 1
    sim=SimpleNamespace(ffi=FFI(),lib=SimpleNamespace(simGetShapeMesh=query),
                        simReleaseBuffer=lambda p:freed.append(id(p)))
    monkeypatch.setitem(sys.modules,'pyrep.backend',SimpleNamespace(sim=sim))
    if failure:
        with pytest.raises(RuntimeError,match='mesh query failed'):native_mesh(42)
    else:
        vertices,triangles=native_mesh(42)
        assert vertices.shape==(6,3) and triangles.tolist()==[[5,3,2]]
    assert freed==[id(b) for b in buffers]


def test_local_mesh_is_copied_and_queried_once_per_reset(monkeypatch):
    from ..generalization import rlbench_geometry
    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    triangles=np.array([[0,1,2]]);calls=[]
    def query(handle):calls.append(handle);return vertices,triangles
    monkeypatch.setattr(rlbench_geometry,'native_mesh',query)
    shape=SimpleNamespace(get_handle=lambda:42)
    bridge=RLBenchBridge.__new__(RLBenchBridge)
    first=bridge.local_mesh(shape);vertices[:]=9;triangles[:]=2
    assert bridge.local_mesh(shape)[0][0].tolist()==[0.,0.,0.]
    assert calls==[42]
    bridge.mesh_cache={};bridge.local_mesh(shape)
    assert calls==[42,42]


def cylinder(radius,height):
    return np.array([[radius*np.cos(a),radius*np.sin(a),z]
                     for a in np.arange(32)*np.pi/16 for z in [-height/2,height/2]])


def test_rotated_collision_mesh_support_is_not_bounding_box_support():
    vertices=cylinder(.04,.03);transform=np.eye(4)
    transform[:3,:3]=Rotation.from_euler('xy',[.7,.6]).as_matrix()
    transform[2,3]=1-world_vertices(vertices,transform)[:,2].min()
    actual=world_vertices(vertices,transform)
    box=[-.04,.04,-.04,.04,-.015,.015]
    assert actual[:,2].min()==pytest.approx(1.)
    assert transformed_bounds(box,transform)[0,2]<.99


@pytest.mark.parametrize('radius,height',[(.04,.03),(.05,.14),(.032,.1)])
def test_round_caps_are_recognized_for_wide_tall_and_narrow_bodies(radius,height):
    assert symmetry_axis(cylinder(radius,height))==2
    assert face_stability_required({'dimensions':[2*radius,2*radius,height],'symmetry_axis':2},2)


def test_cube_square_prism_and_elliptical_cylinder_are_not_axially_symmetric():
    for size in ([.04,.04,.04],[.04,.04,.1]):
        box=np.array(list(itertools.product(*[[-a,a] for a in size])))
        assert symmetry_axis(box) is None
    ellipse=cylinder(.04,.1);ellipse[:,0]*=1.5
    assert symmetry_axis(ellipse) is None


@pytest.mark.parametrize('change',[{}, {'bottom':.747}, {'center':[.003,0.,.801]},
                                 {'tip':[0.,0.,.897]}, {'initial_bottom':.74}])
def test_existing_support_contact_only_allows_shallow_upward_separation(change):
    values=dict(initial_bottom=.7495,bottom=.751,support=.75,initial_center=np.array([0.,0.,.8]),
                center=np.array([0.,0.,.8015]),initial_tip=np.array([0.,0.,.9]),tip=np.array([0.,0.,.9015]))
    values.update(change)
    assert separating_support_contact(**values)==(not change)


@pytest.mark.parametrize('new_wall',[False,True])
def test_shadow_lift_preserves_initial_support_contact_but_rejects_new_collision(monkeypatch,new_wall):
    state={'q':0.}
    class Shape:
        def __init__(self,name):self.name=name
        def get_name(self):return self.name
        def is_respondable(self):return True
        def check_collision(self,other):
            if self.name!='body':return False
            return (other.name=='table' and state['q']<.002) or (new_wall and other.name=='wall' and state['q']>.005)
    monkeypatch.setitem(sys.modules,'pyrep.objects.shape',SimpleNamespace(Shape=Shape))
    monkeypatch.setitem(sys.modules,'pyrep.const',SimpleNamespace(ObjectType=SimpleNamespace(SHAPE=0)))
    hand=Shape('hand')
    arm=SimpleNamespace(joints=[0],get_joint_intervals=lambda:([False],[[-1.,2.]]),
        get_objects_in_tree=lambda **k:[hand],set_joint_positions=lambda q,**k:state.update(q=float(q[0])))
    bridge=SimpleNamespace(env=SimpleNamespace(_robot=SimpleNamespace(arm=arm,
        gripper=SimpleNamespace(get_objects_in_tree=lambda **k:[hand],get_grasped_objects=lambda:[Shape('body')]))),
        entities=[{'primary_name':'body','parts':['body'],'movable':True}],contact_entity=0,
        geometry=lambda _: {'center':np.array([0.,0.,.8+state['q']]),'low':np.array([0.,0.,.7495+state['q']])},
        tip_pose=lambda:np.array([0.,0.,.9+state['q'],0.,0.,0.,1.]),shape_support_height=lambda s,p:.75 if s.name=='table' else 1.2,
        path_to=lambda *a,**k:SimpleNamespace(_path_points=np.array([0.,.001,.01])))
    planner=ShadowPlanner(bridge);planner.structures=[Shape('table'),Shape('wall')]
    if new_wall:
        with pytest.raises(ValueError,match='wall'):planner.path([0,0,1],[0,0,0,1])
    else:assert planner.path([0,0,1],[0,0,0,1])==[[0.],[.001],[.01]]


def test_checked_joint_path_is_executed_without_live_ik_replanning_and_opens_during_retreat():
    pose=np.array([0.,0.,0.,0.,0.,0.,1.]);commands=[];queries=[]
    def step():pose[:3]=[.1,0,0];return True
    def checked(position,orientation,withdraw=False):
        queries.append((position.copy(),orientation.copy(),withdraw));return SimpleNamespace(step=step)
    bridge=SimpleNamespace(path_to=lambda *a,**k:pytest.fail('live IK must not choose a different path'),
        checked_motion=checked,tip_pose=lambda:pose.copy(),tick=lambda:None,
        actuate_gripper=lambda amount:commands.append(amount))
    executor=RLBenchExecutor(bridge);executor.phase='retreat'
    executor.move([.1,0,0],orientation=[0,0,0,1])
    assert queries[0][2] and commands and all(x==1. for x in commands)


def test_checked_path_constructor_never_moves_live_joints_and_rejects_wrong_start(monkeypatch):
    class Path:
        def __init__(self,arm,points):self.points=points
    monkeypatch.setitem(sys.modules,'pyrep.robots.configuration_paths.arm_configuration_path',SimpleNamespace(ArmConfigurationPath=Path))
    bridge=RLBenchBridge.__new__(RLBenchBridge)
    bridge.env=SimpleNamespace(_robot=SimpleNamespace(arm=SimpleNamespace(joints=[0],get_joint_positions=lambda:[0.])))
    assert bridge.configuration_path([[0.],[.1]]).points.tolist()==[0.,.1]
    assert bridge.configuration_path([[0.],[0.]]).step() is True
    with pytest.raises(RuntimeError,match='start differs'):bridge.configuration_path([[.1],[.2]])


def test_thin_body_search_contains_level_side_and_top_contacts_before_rolled_wrists():
    planner=ShadowPlanner(SimpleNamespace(base_position=lambda:np.array([-1.,0.,0.]),
        grasp_clearance=lambda orientation:.01))
    body={'dimensions':[.08,.08,.03],'center':np.array([0.,0.,.765]),
          'low':np.array([-.04,-.04,.75]),'high':np.array([.04,.04,.78]),
          'horizontal_span':.08,'yaw':0.}
    candidates=planner.grasp_proposals(body)
    assert np.allclose(Rotation.from_quat(candidates[0][1]).as_matrix()[:,2],[-1.,0.,0.])
    assert abs(Rotation.from_quat(candidates[1][1]).as_matrix()[2,2])<.3
    assert Rotation.from_quat(candidates[2][1]).as_matrix()[2,2]==pytest.approx(-1.)
    assert len(candidates)>12
    tall=dict(body,dimensions=[.1,.1,.14])
    assert abs(Rotation.from_quat(planner.grasp_proposals(tall)[0][1]).as_matrix()[2,2])<.9


@pytest.mark.parametrize('detected,clear,failure',[(True,True,False),(False,True,False),(True,False,False),(True,True,True)])
def test_relative_contact_screen_preserves_native_gate_and_restores_shadow_body(monkeypatch,detected,clear,failure):
    original=np.eye(4)[:3,:].copy();original[:,3]=[.2,.3,.4]
    state={'matrix':original.copy()};seen=[]
    class Shape:
        def __init__(self,name):assert name=='predicted'
        def get_matrix(self):return state['matrix'].copy()
        def set_matrix(self,value):state['matrix']=np.array(value).copy()
    monkeypatch.setitem(sys.modules,'pyrep.objects.shape',SimpleNamespace(Shape=Shape))
    def detector(index):
        assert index==0;seen.append(state['matrix'].copy())
        if failure:raise RuntimeError('query failed')
        return detected
    bridge=SimpleNamespace(entities=[{'primary_name':'predicted'}],
        tip_pose=lambda:np.array([1.,0.,0.,0.,0.,0.,1.]),
        detect_grasp=detector,release_clearance=lambda index:clear)
    planner=ShadowPlanner(bridge)
    if failure:
        with pytest.raises(RuntimeError,match='query failed'):planner.contact_feasible(0,[.5,0,0],[0,0,0,1])
    else:assert planner.contact_feasible(0,[.5,0,0],[0,0,0,1])==(detected and clear)
    assert np.allclose(seen[0][:,3],[.7,.3,.4])
    assert np.array_equal(state['matrix'],original)


def test_shelf_alignment_preserves_support_face_and_removes_quarter_turn_yaw_error():
    object_rotation=Rotation.from_euler('z',37,degrees=True).as_matrix()
    hand_rotation=Rotation.from_euler('y',90,degrees=True).as_matrix()
    relative=np.eye(4);relative[:3,:3]=hand_rotation.T@object_rotation
    inward=np.array([.7,.3,0.]);inward/=np.linalg.norm(inward)
    candidates=aligned_shelf_rotations([(0.,object_rotation)],relative,inward)
    assert len(candidates)==2
    new=candidates[0][1];forward=(new@relative[:3,:3].T)[:,2]
    assert np.allclose(forward,inward)
    assert np.allclose(new[:,2],[0,0,1])
    assert np.array_equal(candidates[-1][1],object_rotation)


def test_candidate_budget_yields_to_other_grasps_but_global_budget_still_stops(monkeypatch):
    from ..generalization import rlbench_planning
    monkeypatch.setattr(rlbench_planning.time,'monotonic',lambda:20.)
    planner=ShadowPlanner(SimpleNamespace());planner.deadline=30;planner.candidate_deadline=19
    with pytest.raises(RuntimeError,match='try another grasp'):planner.check_budget()
    planner.deadline=19
    with pytest.raises(TimeoutError,match='feasibility unverified'):planner.check_budget()


def test_shelf_staging_commands_hand_offset_and_shelf_height_not_payload_center(monkeypatch):
    hand=Rotation.from_euler('y',90,degrees=True).as_quat()
    pose=np.r_[[0.,0.,.8],hand];start=pose.copy()
    def geometry(index):
        center=pose[:3]+np.array([.1,0.,0.])
        return {'center':center,'low':center-[.04,.04,.015],'high':center+[.04,.04,.015],
                'dimensions':np.array([.08,.08,.03]),'symmetry_axis':2}
    class Shape:
        def __init__(self,name):pass
        def get_matrix(self):
            transform=np.eye(4);transform[:3,3]=geometry(0)['center'];return transform
    monkeypatch.setitem(sys.modules,'pyrep.objects.shape',SimpleNamespace(Shape=Shape))
    def restore(state):pose[:]=state
    bridge=SimpleNamespace(entities=[{'primary_name':'body'}],tip_pose=lambda:pose.copy(),geometry=geometry,
        placement_route=lambda *a,**k:(np.array([1.,0.,1.3]),np.array([.7,0.,1.3])),
        snapshot_state=lambda:pose.copy(),restore_state=restore,support_height=lambda *a:1.285,
        env=SimpleNamespace(_robot=SimpleNamespace(gripper=SimpleNamespace(release=lambda:None))))
    planner=ShadowPlanner(bridge);planner.support_axis=2
    monkeypatch.setattr(planner,'check_release',lambda _:None)
    monkeypatch.setattr(planner,'check_settling_clearance',lambda *a,**k:None)
    monkeypatch.setattr(planner,'attach_copy',lambda _:None)
    def path(position,orientation):pose[:3]=position;pose[3:]=orientation
    monkeypatch.setattr(planner,'path',path)
    result=planner.placement(0,1)
    anchor=np.array(result['reorientation_waypoints'][1]['position'])
    assert np.allclose(anchor,[.6,0.,1.3])
    assert np.array_equal(pose,start), 'all candidate FK remains in the shadow scene'


@pytest.mark.parametrize('minimum,span,q,cyclic,valid',[
    (-3.,2.,-2.,False,True),(-3.,2.,-.5,False,False),
    (1.,2.,2.5,False,True),(1.,2.,3.1,False,False),(1.,2.,100.,True,True)])
def test_native_joint_intervals_are_minimum_and_range_not_minimum_and_maximum(monkeypatch,minimum,span,q,cyclic,valid):
    class Shape:
        def __init__(self,name):pass
        def check_collision(self,other):return False
    monkeypatch.setitem(sys.modules,'pyrep.objects.shape',SimpleNamespace(Shape=Shape))
    monkeypatch.setitem(sys.modules,'pyrep.const',SimpleNamespace(ObjectType=SimpleNamespace(SHAPE=0)))
    arm=SimpleNamespace(joints=[0],get_joint_intervals=lambda:([cyclic],[[minimum,span]]),
        get_objects_in_tree=lambda **k:[],set_joint_positions=lambda *a,**k:None)
    bridge=SimpleNamespace(env=SimpleNamespace(_robot=SimpleNamespace(arm=arm,
        gripper=SimpleNamespace(get_objects_in_tree=lambda **k:[]))),
        entities=[{'primary_name':'body','parts':['body'],'movable':True}],contact_entity=0,
        path_to=lambda *a,**k:SimpleNamespace(_path_points=[q]))
    planner=ShadowPlanner(bridge);planner.structures=[]
    if valid:assert planner.path([0,0,1],[0,0,0,1])==[[q]]
    else:
        with pytest.raises(ValueError,match='joint limit'):planner.path([0,0,1],[0,0,0,1])


@pytest.mark.parametrize('always_fails',[False,True])
def test_failed_stochastic_paths_retry_same_shadow_pose_attachment_and_safety_checks(monkeypatch,always_fails):
    state={'q':0,'attached':True};starts=[];attachments=[]
    def restore(value):state.update(q=value)
    bridge=SimpleNamespace(contact_entity=7,snapshot_state=lambda:state['q'],restore_state=restore,
        env=SimpleNamespace(_robot=SimpleNamespace(gripper=SimpleNamespace(
            get_grasped_objects=lambda:[7] if state['attached'] else [],
            release=lambda:state.update(attached=False)))))
    planner=ShadowPlanner(bridge)
    def attach(index):attachments.append(index);state['attached']=True
    monkeypatch.setattr(planner,'attach_copy',attach)
    def checked(position,orientation):
        starts.append(state.copy());state['q']=100
        if always_fails or len(starts)==1:raise ValueError('unsafe sampled path')
        return [[0.],[.1]]
    monkeypatch.setattr(planner,'_path_once',checked)
    if always_fails:
        with pytest.raises(ValueError,match='unsafe sampled path'):planner.path([0,0,1],[0,0,0,1])
        assert len(starts)==3 and state=={'q':0,'attached':True}
    else:assert planner.path([0,0,1],[0,0,0,1])==[[0.],[.1]]
    assert all(s=={'q':0,'attached':True} for s in starts)
    assert all(index==7 for index in attachments)


def test_regrasp_sites_keep_observed_surface_avoid_other_objects_and_table_edges():
    body={'center':np.array([.3,0.,.85]),'dimensions':np.array([.08,.08,.03])}
    other={'low':np.array([.46,-.04,.75]),'high':np.array([.54,.04,.83])}
    def support(shape,point):
        if shape.name=='floor':return 0.
        if shape.name=='shelf' or abs(point[0])>.7 or abs(point[1])>.4:raise ValueError('no support')
        return .75
    bridge=SimpleNamespace(entities=[{'movable':True},{'movable':True}],
        geometry=lambda i:body if i==0 else other,base_position=lambda:np.zeros(3),shape_support_height=support)
    planner=ShadowPlanner(bridge)
    planner.structures=[SimpleNamespace(name=n,get_name=lambda n=n:n) for n in ['floor','table','shelf']]
    sites=planner.staging_sites(0)
    assert sites and all(s['structure']=='table' and s['height']==.75 for s in sites)
    assert not any(np.allclose(s['point'][:2],[.5,0.]) for s in sites), 'occupied site must not be selected'
    assert all(abs(s['point'][0])+.08<=.7 and abs(s['point'][1])+.08<=.4 for s in sites)


@pytest.mark.parametrize('horizon',[50,1000])
def test_regrasp_releases_and_verifies_same_entity_once_within_original_horizon(monkeypatch,horizon):
    from ..generalization.rlbench_planning import PLANNING_PROFILE
    monkeypatch.setitem(PLANNING_PROFILE['regrasp_recovery'],'minimum_remaining_steps',0)
    route={'waypoints':[],'release':[.5,0.,.8],'above':[.5,0.,.95],
           'orientation':[0.,0.,0.,1.],'support_shape':'table','support_height':.75,
           'regrasp':{'requires_contact_search':True,'grasps':[{'candidate_id':1,'point':[.5,0.,.8],'orientation':[0,0,0,1],'yaw':0.}]}}
    calls=[]
    bridge=SimpleNamespace(planning_request=lambda op,i,d:(calls.append((op,i,d)) or route),
        geometry=lambda i:{'center':np.array([.5,0.,.765])},physical_support_height=lambda name,point:.75,
        release_sample=lambda i:{'pose':[.5,0.,.765,0.,0.,0.,1.],'bottom':.75,'face_tilt':0.,'attached':False},
        release_clearance=lambda i:True,requires_stable_face=lambda i:True,tick=lambda:None)
    executor=RLBenchExecutor(bridge,horizon=horizon);picks=[];prepared=[]
    def pick(index,destination_index=None,*,allow_regrasp=True,prepared_search=None):
        prepared.append(prepared_search)
        picks.append((index,destination_index,allow_regrasp));executor.held=index
    monkeypatch.setattr(executor,'pick',pick)
    monkeypatch.setattr(executor,'move',lambda *a,**k:executor.tick())
    monkeypatch.setattr(executor,'gripper',lambda opened:executor.tick())
    if horizon==50:
        with pytest.raises(TimeoutError,match='horizon'):executor.regrasp(2,4,'infeasible direct grasp')
        assert executor.steps==50 and picks==[(2,None,False)]
    else:
        executor.regrasp(2,4,'infeasible direct grasp')
        assert picks==[(2,None,False),(2,4,False)] and calls==[('staging',2,4)]
        assert prepared==[None,route['regrasp']]
        with pytest.raises(RuntimeError):executor.regrasp(2,4,'second recovery forbidden')
        assert len(picks)==2


def test_acquisition_lift_uses_measured_contact_frame_without_nominal_pose_scraping(monkeypatch):
    nominal=np.array([0.,1.,0.,0.]);actual=(Rotation.from_euler('x',.03)*Rotation.from_quat(nominal)).as_quat()
    pose=np.r_[[.1,.2,.5],nominal];state={'z':.03};verified=[]
    def request(op,index,destination,**options):
        if op=='grasp':return {'requires_contact_search':True,'grasps':[{
            'candidate_id':0,'point':[.1,.2,.05],'orientation':nominal.tolist(),'yaw':0.}]}
        assert op=='verify_grasp'
        assert np.allclose(options['lift_position'][:2],[.105,.197])
        assert np.allclose(options['orientation'],actual);verified.append(True)
        return {'placement':{}}
    bridge=SimpleNamespace(entities=[{},{}],planning_request=request,checked_motion=lambda *a,**k:None,
        geometry=lambda i:{'movable':True,'center':np.array([.1,.2,state['z']]),'low':np.array([0.,0.,.015]),
            'high':np.array([.14,.24,.045]),'horizontal_span':.08,'yaw':0.},
        base_position=lambda:np.array([-1.,0.,0.]),tip_pose=lambda:pose.copy(),
        detect_grasp=lambda i:True,try_grasp=lambda i:True,tick=lambda:None)
    executor=RLBenchExecutor(bridge)
    def move(position,*a,**options):
        if executor.phase=='lift':
            assert np.allclose(position[:2],pose[:2]) and np.allclose(options['orientation'],actual)
            state['z']=.2
        pose[:3]=position
        if executor.phase=='grasp_approach':pose[:2]+=[.005,-.003];pose[3:]=actual
    monkeypatch.setattr(executor,'move',move);monkeypatch.setattr(executor,'gripper',lambda opened:None)
    executor.pick(0,1)
    assert verified and executor.held==0


@pytest.mark.parametrize('mode',['valid','obstacle','stale_goal','start_drift'])
def test_cached_path_rechecks_whole_robot_in_current_scene_without_new_ik(monkeypatch,mode):
    state={'q':.1 if mode=='start_drift' else .001};checked=[]
    class Shape:
        def __init__(self,name):pass
        def check_collision(self,other):return False
    monkeypatch.setitem(sys.modules,'pyrep.objects.shape',SimpleNamespace(Shape=Shape))
    monkeypatch.setitem(sys.modules,'pyrep.const',SimpleNamespace(ObjectType=SimpleNamespace(SHAPE=0)))
    arm=SimpleNamespace(joints=[0],get_joint_intervals=lambda:([False],[[-1.,2.]]),
        get_joint_positions=lambda:[state['q']],get_objects_in_tree=lambda **k:[],
        set_joint_positions=lambda values,**k:state.update(q=float(values[0])))
    def validate(exempt):
        assert exempt==[0];checked.append(state['q'])
        if mode=='obstacle' and state['q']>=.01:raise ValueError('new fixture collision')
    bridge=SimpleNamespace(env=SimpleNamespace(_robot=SimpleNamespace(arm=arm,
        gripper=SimpleNamespace(get_objects_in_tree=lambda **k:[],get_grasped_objects=lambda:[]))),
        entities=[{'primary_name':'body','parts':['body'],'movable':True}],contact_entity=0,
        tip_pose=lambda:np.array([state['q'],0.,0.,0.,0.,0.,1.]),
        local_contact_exemptions=lambda:[0],validate_arm_configuration=validate,
        path_to=lambda *a,**k:pytest.fail('cached valid path must not sample another IK route'))
    planner=ShadowPlanner(bridge);planner.structures=[];planner.proposed_joint_path=[[0.],[.01]]
    target=[.1 if mode=='stale_goal' else .01,0.,0.]
    if mode=='valid':
        points=planner._path_once(target,[0,0,0,1])
        assert points[0]==[.001] and len(checked)==len(points)
    else:
        with pytest.raises(ValueError,match={'obstacle':'fixture','stale_goal':'endpoint','start_drift':'start drift'}[mode]):
            planner._path_once(target,[0,0,0,1])


@pytest.mark.parametrize('collision',[False,True])
@pytest.mark.parametrize('original_pred_flag',[False,True])
def test_cached_native_arm_check_preserves_every_contact_flag_and_checks_other_objects(monkeypatch,collision,original_pred_flag):
    flags={'predicted':original_pred_flag,'wrong_object':True,'fixture':True};seen=[]
    class Shape:
        def __init__(self,name):self.name=name
        def is_collidable(self):return flags[self.name]
        def set_collidable(self,value):flags[self.name]=value
    monkeypatch.setitem(sys.modules,'pyrep.objects.shape',SimpleNamespace(Shape=Shape))
    def check():seen.append(flags.copy());return collision
    bridge=RLBenchBridge.__new__(RLBenchBridge);bridge.entities=[{'parts':['predicted']},{'parts':['wrong_object']}]
    bridge.env=SimpleNamespace(_robot=SimpleNamespace(arm=SimpleNamespace(check_arm_collision=check)))
    if collision:
        with pytest.raises(ValueError,match='robot/scene'):bridge.validate_arm_configuration([0])
    else:bridge.validate_arm_configuration([0])
    assert seen==[{'predicted':False,'wrong_object':True,'fixture':True}]
    assert flags=={'predicted':original_pred_flag,'wrong_object':True,'fixture':True}


@pytest.mark.parametrize('detected',[False,True])
def test_approach_stops_when_same_predicted_body_moves_and_requires_native_contact(detected):
    state={'ticks':0};held=[];queries=[]
    path=SimpleNamespace(step=lambda:False)
    def detector(index):queries.append(index);return detected
    bridge=SimpleNamespace(path_to=lambda *a,**k:pytest.fail('must execute checked motion'),
        checked_motion=lambda *a,**k:path,contact_entity=2,
        geometry=lambda i:{'center':np.array([.006 if state['ticks'] else 0.,0.,0.])},
        hold_arm=lambda p:held.append(p),detect_grasp=detector,release_clearance=lambda i:True,
        tip_pose=lambda:np.array([0.,0.,0.,0.,0.,0.,1.]),
        tick=lambda:state.update(ticks=state['ticks']+1))
    executor=RLBenchExecutor(bridge);executor.phase='grasp_approach'
    if detected:executor.move([.1,0.,0.],orientation=[0,0,0,1])
    else:
        with pytest.raises(RuntimeError,match='object moved'):executor.move([.1,0.,0.],orientation=[0,0,0,1])
    assert held==[path] and queries==[2] and executor.steps==1


@pytest.mark.parametrize('finished',[False,True])
def test_contact_stop_holds_measured_joints_without_teleporting_or_double_freeing_rml(monkeypatch,finished):
    removed=[];targets=[]
    monkeypatch.setitem(sys.modules,'pyrep.backend',SimpleNamespace(sim=SimpleNamespace(simRMLRemove=lambda h:removed.append(h))))
    bridge=RLBenchBridge.__new__(RLBenchBridge)
    bridge.env=SimpleNamespace(_robot=SimpleNamespace(arm=SimpleNamespace(
        get_joint_positions=lambda:[.1,.2],set_joint_target_positions=lambda q:targets.append(q))))
    path=SimpleNamespace(_rml_handle=42,_path_done=finished)
    bridge.hold_arm(path);bridge.hold_arm(path)
    assert removed==([] if finished else [42]) and targets==[[.1,.2],[.1,.2]]
    assert path._rml_handle is None and path._path_done


@pytest.mark.parametrize('problem',['tilted','overhang'])
def test_realized_inserted_body_must_be_stable_and_supported_before_release(monkeypatch,problem):
    body={'center':np.array([0.,0.,1.021]),'low':np.array([-.05,-.05,1.006]),
          'high':np.array([.05,.05,1.036]),'dimensions':np.array([.08,.08,.03]),'symmetry_axis':2}
    bridge=SimpleNamespace(geometry=lambda i:body,release_sample=lambda i:{'face_tilt':.6 if problem=='tilted' else 0.},
        support_height=lambda i,p:.75 if p[0]>.04 else 1.)
    planner=ShadowPlanner(bridge);planner.support_axis=2
    monkeypatch.setattr(planner,'check_release',lambda i:pytest.fail('unsafe body must not reach gripper opening'))
    with pytest.raises(ValueError,match='face-stable' if problem=='tilted' else 'overhangs'):
        planner.verify_release(0,1,None)


@pytest.mark.parametrize('actual_open',[False,True])
@pytest.mark.parametrize('settling_obstruction',[False,True])
def test_open_hand_must_clear_full_settling_sweep_and_restore_shadow_state(monkeypatch,actual_open,settling_obstruction):
    original=np.array([0.,0.,1.022,0.,0.,0.,1.]);state={'pose':original.copy(),'q':.03};queries=[]
    class Shape:
        def __init__(self,name):assert name=='body'
        def get_pose(self):return state['pose'].copy()
        def set_pose(self,value):state['pose']=np.array(value)
    monkeypatch.setitem(sys.modules,'pyrep.objects.shape',SimpleNamespace(Shape=Shape))
    def joints(q,**kwargs):
        state['q']=q[0];state['pose'][0]+=.1  # hypothetical attach-frame movement must be undone
    gripper=SimpleNamespace(get_joint_positions=lambda:[state['q']],get_joint_intervals=lambda:([False],[[0.,.04]]),set_joint_positions=joints)
    def clear(index):
        assert index==0;queries.append(state['pose'].copy())
        return state['pose'][2]>=1.02 or (not settling_obstruction and state['q']>=.039)
    bridge=SimpleNamespace(entities=[{'primary_name':'body'}],env=SimpleNamespace(_robot=SimpleNamespace(gripper=gripper)),
        geometry=lambda i:{'low':np.array([0.,0.,state['pose'][2]-.015])},release_clearance=clear)
    planner=ShadowPlanner(bridge)
    if settling_obstruction or actual_open:
        with pytest.raises(ValueError,match='settling'):planner.check_settling_clearance(0,1.,actual_open=actual_open)
    else:planner.check_settling_clearance(0,1.,actual_open=actual_open)
    assert len(queries)>1 and all(p[0]==0. for p in queries)
    assert np.array_equal(state['pose'],original) and state['q']==.03


@pytest.mark.parametrize('wrong_object,pose_jump',[(False,False),(True,False),(False,True)])
def test_native_detach_checks_predicted_identity_and_never_rewinds_world_pose(monkeypatch,wrong_object,pose_jump):
    pose=np.array([0.,0.,1.,0.,0.,0.,1.]);released=[]
    class Shape:
        def __init__(self,name):self.name=name
        def get_name(self):return self.name
        def get_pose(self):return pose.copy()
    monkeypatch.setitem(sys.modules,'pyrep.objects.shape',SimpleNamespace(Shape=Shape))
    def release():
        released.append(True)
        if pose_jump:pose[0]=.1
    bridge=RLBenchBridge.__new__(RLBenchBridge);bridge.entities=[{'primary_name':'predicted'}]
    bridge.env=SimpleNamespace(_robot=SimpleNamespace(gripper=SimpleNamespace(
        get_grasped_objects=lambda:[Shape('wrong' if wrong_object else 'predicted')],release=release)))
    if wrong_object or pose_jump:
        with pytest.raises(RuntimeError,match='predicted' if wrong_object else 'world pose'):bridge.release_native(0)
    else:bridge.release_native(0)
    assert len(released)==(0 if wrong_object else 1)
    assert pose[0]==(.1 if pose_jump else 0.)


@pytest.mark.parametrize('unsafe_after_open',[False,True])
def test_live_release_opens_then_checks_actual_fingers_then_detaches_before_withdrawal(monkeypatch,unsafe_after_open):
    q=Rotation.from_euler('y',90,degrees=True).as_quat();pose=np.r_[[.5,0.,1.05],q]
    state={'attached':True};order=[]
    sample=lambda:{'pose':[1.,0.,1.015,0.,0.,0.,1.],'bottom':1.,'face_tilt':0.,'attached':state['attached']}
    def request(op,index,destination,**options):
        assert index==0 and destination==1
        if op=='place':return {'placement':{'orientation':q.tolist(),'reorientation_waypoints':[]}}
        if op=='verify_placement':return {'release':[1.,0.,1.05],'approach':[.8,0.,1.05],'orientation':q.tolist()}
        assert op=='verify_release';order.append('verify_actual_open' if options.get('actual_open') else 'verify_closed')
        if options.get('actual_open') and unsafe_after_open:raise RuntimeError('planning worker: ValueError: fingers obstruct actual support settling')
        return {'sample':sample(),'support':1.,'retreat_joint_path':[[0.],[.1]]}
    def actuate(amount,**kwargs):assert amount==1. and state['attached'];order.append('open_fingers');return True
    def detach(index):assert index==0 and state['attached'];order.append('detach');state['attached']=False
    bridge=SimpleNamespace(entities=[{},{}],tip_pose=lambda:pose.copy(),planning_request=request,
        checked_motion=lambda *a,**k:None,actuate_fingers=actuate,release_native=detach,
        geometry=lambda i:{'center':pose[:3]-[.1,0.,0.],'low':np.array([0.,0.,.7])},
        placement_route=lambda *a,**k:(np.array([1.,0.,1.05]),np.array([.8,0.,1.05])),
        support_height=lambda *a:1.,release_sample=lambda i:sample(),release_clearance=lambda i:True,
        requires_stable_face=lambda i:True,tick=lambda:None)
    executor=RLBenchExecutor(bridge);executor.held=0
    def move(position,**options):
        if executor.phase=='retreat':assert not state['attached'];order.append('withdraw')
        pose[:3]=position;pose[3:]=options['orientation']
    monkeypatch.setattr(executor,'move',move)
    monkeypatch.setattr(executor,'gripper',lambda opened:order.append('legacy_gripper'))
    if unsafe_after_open:
        with pytest.raises(RuntimeError,match='actual support'):executor.place(1,allow_recovery=False)
        assert state['attached'] and executor.held==0 and 'detach' not in order and 'withdraw' not in order
    else:
        executor.place(1,allow_recovery=False)
        assert order[:5]==['verify_closed','open_fingers','verify_actual_open','detach','withdraw']
        assert executor.held is None


@pytest.mark.parametrize('mode',['contact','timeout','horizon'])
def test_slow_finger_closure_stops_on_contact_or_failure_without_squeezing_full_range(mode):
    state={'ticks':0};held=[];commands=[];queries=[]
    def distance(index):
        queries.append(index)
        return .0004 if mode=='contact' and state['ticks']>=3 else .02
    bridge=SimpleNamespace(hold_fingers=lambda:held.append(state['ticks']),finger_clearance=distance,
        actuate_fingers=lambda amount,**kw:(commands.append((amount,kw['velocity'])) or False),
        tick=lambda:state.update(ticks=state['ticks']+1))
    executor=RLBenchExecutor(bridge,horizon=2 if mode=='horizon' else 1000)
    if mode=='contact':executor.close_contact(5)
    else:
        with pytest.raises(TimeoutError):executor.close_contact(5)
    assert held==[3 if mode=='contact' else 2 if mode=='horizon' else 80]
    assert set(queries)=={5} and all(c==(0.,.02) for c in commands)


@pytest.mark.parametrize('during_contact',[False,True])
def test_contact_target_follows_same_body_translation_and_rotation_before_or_during_approach(monkeypatch,during_contact):
    source=np.array([0.,0.,.03,0.,0.,0.,1.])
    changed=np.r_[[.01,.02,.03],Rotation.from_euler('z',90,degrees=True).as_quat()]
    state={'pose':source.copy() if during_contact else changed.copy()};queries=[];moves=[]
    def sample(index):queries.append(index);return {'pose':state['pose'].copy()}
    executor=RLBenchExecutor(SimpleNamespace(release_sample=sample))
    def move(point,*a,**options):
        moves.append((np.array(point),np.array(options['orientation'])))
        if during_contact and len(moves)==1:
            state['pose']=changed.copy();raise ContactMotionStop('same body moved')
    monkeypatch.setattr(executor,'move',move)
    point,q=executor.approach_contact(2,np.array([.1,0.,.05]),[0,1,0,0],0.,tolerance=.006,stall_after=20,body_pose=source)
    assert np.allclose(point,[.01,.12,.05])
    expected=(Rotation.from_euler('z',90,degrees=True)*Rotation.from_quat([0,1,0,0])).as_quat()
    assert abs(np.dot(q,expected))==pytest.approx(1.) and set(queries)=={2}
    assert len(moves)==(2 if during_contact else 1)


def test_contact_following_has_finite_budget_without_changing_entity(monkeypatch):
    queries=[];moves=[]
    bridge=SimpleNamespace(release_sample=lambda i:(queries.append(i) or {'pose':[0,0,0,0,0,0,1]}))
    executor=RLBenchExecutor(bridge)
    def move(*a,**k):moves.append(True);raise ContactMotionStop('body moved again')
    monkeypatch.setattr(executor,'move',move)
    with pytest.raises(RuntimeError,match='budget exhausted'):
        executor.approach_contact(3,np.zeros(3),[0,0,0,1],0.,tolerance=.006,stall_after=20)
    assert len(moves)==13 and set(queries)=={3}


def test_finger_hold_stops_velocity_and_commands_measured_positions_without_live_rewind():
    calls=[];bridge=RLBenchBridge.__new__(RLBenchBridge)
    gripper=SimpleNamespace(joints=[0,1],get_joint_positions=lambda:[.01,.02],
        set_joint_target_velocities=lambda values:calls.append(('velocity',values)),
        set_joint_target_positions=lambda values:calls.append(('position',values)))
    bridge.env=SimpleNamespace(_robot=SimpleNamespace(gripper=gripper));bridge.hold_fingers()
    assert calls==[('velocity',[0.,0.]),('position',[.01,.02])]


def test_regrasp_reserves_physical_steps_without_resetting_or_extending_horizon(monkeypatch):
    executor=RLBenchExecutor(SimpleNamespace());executor.steps=401
    monkeypatch.setattr(executor,'pick',lambda *a,**k:pytest.fail('insufficient remaining physical budget'))
    with pytest.raises(RuntimeError,match='no route'):executor.regrasp(0,1,'no route')
    assert executor.steps==401 and executor.horizon==1000 and not executor.regrasp_used


@pytest.mark.parametrize('onward_valid',[False,True])
def test_empty_staging_site_is_not_committed_without_verified_onward_grasp_and_destination(monkeypatch,onward_valid):
    import copy
    start_body=np.array([.1,0.,.8,0.,0.,0.,1.]);start_tip=np.r_[[0.,0.,.8],Rotation.from_euler('y',90,degrees=True).as_quat()]
    state={'body':start_body.copy(),'tip':start_tip.copy(),'q':[.01],'attached':True};calls=[]
    class Shape:
        def __init__(self,name):assert name=='body'
        def get_matrix(self):return matrix(state['body'])
        def get_pose(self):return state['body'].copy()
        def set_pose(self,value):state['body']=np.asarray(value)
    from ..generalization.rlbench_planning import matrix
    monkeypatch.setitem(sys.modules,'pyrep.objects.shape',SimpleNamespace(Shape=Shape))
    def restore(value):state.clear();state.update(copy.deepcopy(value))
    def geometry(index):
        assert index==0;c=state['body'][:3]
        return {'center':c.copy(),'low':c-[.04,.04,.015],'high':c+[.04,.04,.015],
                'dimensions':np.array([.08,.08,.03]),'symmetry_axis':2}
    gripper=SimpleNamespace(release=lambda:state.update(attached=False),get_joint_intervals=lambda:([False],[[0.,.04]]),
        set_joint_positions=lambda q,**k:state.update(q=list(q)))
    bridge=SimpleNamespace(entities=[{'primary_name':'body'},{'primary_name':'other'},{'primary_name':'predicted_destination'}],
        geometry=geometry,tip_pose=lambda:state['tip'].copy(),snapshot_state=lambda:copy.deepcopy(state),restore_state=restore,
        env=SimpleNamespace(_robot=SimpleNamespace(gripper=gripper)))
    planner=ShadowPlanner(bridge);planner.support_axis=2
    monkeypatch.setattr(planner,'staging_sites',lambda index:[{'point':[.5,0.,.8],'height':.75,'structure':'table'}])
    monkeypatch.setattr(planner,'attach_copy',lambda index:state.update(attached=True))
    monkeypatch.setattr(planner,'check_release',lambda index:None)
    def path(position,orientation):state['tip']=np.r_[position,orientation];return [[0.],[.1]]
    monkeypatch.setattr(planner,'path',path)
    def future(index,destination,settings):
        calls.append((index,destination));assert settings['force_search']
        return {'grasps':[{'candidate_id':9}] if onward_valid else [],'requires_contact_search':True}
    monkeypatch.setattr(planner,'search_grasps',future)
    if onward_valid:assert planner.staging(0,2)['regrasp']['grasps']==[{'candidate_id':9}]
    else:
        with pytest.raises(RuntimeError,match='no checked regrasp'):planner.staging(0,2)
    assert calls and all(c==(0,2) for c in calls)
    assert np.array_equal(state['body'],start_body) and np.array_equal(state['tip'],start_tip) and state['attached']


def test_measured_bad_grasp_is_retried_before_lifting_same_predicted_body(monkeypatch):
    calls=[];poses=np.array([0.,0.,.5,0.,1.,0.,0.]);state={'z':.03,'candidate':0}
    def request(operation,index,destination,**kwargs):
        assert index==0 and destination==1
        calls.append((operation,kwargs))
        if operation=='grasp':
            state['candidate']=1 if kwargs.get('excluded_candidates') else 0
            return {'requires_contact_search':True,'grasps':[{'candidate_id':state['candidate'],
                'point':[0.,0.,.05],'orientation':[0.,1.,0.,0.],'yaw':0.}]}
        assert operation=='verify_grasp'
        if state['candidate']==0:raise RuntimeError('release finger clearance')
        return {'placement':{}}
    bridge=SimpleNamespace(entities=[{},{}],planning_request=request,checked_motion=lambda *a,**k:None,
        geometry=lambda _: {'movable':True,'center':np.array([0.,0.,state['z']]),'low':np.array([0.,0.,.015]),
                            'high':np.array([0.,0.,.045]),'yaw':0.,'horizontal_span':.08},
        base_position=lambda:np.array([-1.,0.,0.]),tip_pose=lambda:poses.copy(),
        tick=lambda:None,try_grasp=lambda _:True,detect_grasp=lambda _:True)
    executor=RLBenchExecutor(bridge);lifted=[]
    def move(position,*a,**k):
        poses[:3]=position
        if executor.phase=='lift':lifted.append(state['candidate']);state['z']=.2
    monkeypatch.setattr(executor,'move',move);monkeypatch.setattr(executor,'gripper',lambda _:None)
    executor.pick(0,destination_index=1)
    assert lifted==[1] and executor.held==0
    assert calls[2][1]['excluded_candidates']==[0]
