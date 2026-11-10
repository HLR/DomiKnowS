import copy
import numpy as np
import pytest

from ..generalization.rlbench_bridge import transformed_bounds, ray_hits,pose_error
from ..generalization.rlbench_executor import RLBenchExecutor, downward_quaternion, rotation_step,side_entry_quaternion,tilted_downward_quaternion,lateral_grasp_quaternion
from ..generalization.rlbench_campaign import public_input, save
from ..generalization.rlbench_bridge import RLBenchBridge


def test_world_bounds_include_rotation_and_translation():
    matrix=np.array([[0,-1,0,10],[1,0,0,20],[0,0,1,30],[0,0,0,1]])
    box=transformed_bounds([-1,1,-2,2,-3,3],matrix)
    assert np.allclose(box,[[8,19,27],[12,21,33]])


def test_mesh_rays_distinguish_wall_from_opening_without_goal_sensor():
    # The same physical wall blocks a ray through its face but not a ray
    # parallel to it or outside its extent; triangle winding is irrelevant.
    vertices=np.array([[1,-1,-1],[1,1,-1],[1,1,1],[1,-1,1]])
    triangles=[[0,1,2],[0,2,3]]
    assert np.allclose(ray_hits(vertices,triangles,[0,.2,0],[1,0,0]),[1])
    assert np.allclose(ray_hits(vertices,triangles,[0,.2,0],[0,0,1]),[])
    assert np.allclose(ray_hits(vertices,triangles,[0,2,0],[1,0,0]),[])
    assert np.allclose(ray_hits(vertices,triangles,[2,.2,0],[-1,0,0]),[1])


def test_xyzw_quaternion_steps_are_normalized_and_take_short_rotation():
    target=downward_quaternion(.7)
    assert np.isclose(np.linalg.norm(target),1)
    assert np.allclose(rotation_step(target,-target),target)
    start=downward_quaternion(0)
    step=rotation_step(start,downward_quaternion(np.pi))
    assert 0<2*np.arccos(abs(np.dot(step,start)))<=.1
    with pytest.raises(ValueError,match='zero'):rotation_step([0,0,0,0],target)


def test_side_entry_points_fingers_into_observed_opening():
    from scipy.spatial.transform import Rotation
    inward=np.array([1.,2.,0.]);inward=inward/np.linalg.norm(inward)
    matrix=Rotation.from_quat(side_entry_quaternion(inward)).as_matrix()
    assert np.allclose(matrix[:,2],inward)
    assert np.allclose(matrix[:,1],[0,0,1])


def test_tilted_grasp_points_toward_observed_object_without_changing_id():
    from scipy.spatial.transform import Rotation
    matrix=Rotation.from_quat(tilted_downward_quaternion(0,[1,0,0])).as_matrix()
    assert matrix[0,2]>0 and matrix[2,2]<-.9
    assert np.allclose(tilted_downward_quaternion(0,[0,0,0]),downward_quaternion(0))


def test_pose_reset_check_accepts_antipodal_quaternions_but_rejects_displacement():
    pose=np.array([1.,2.,3.,0.,0.,0.,1.]);same=pose.copy();same[3:]*=-1
    assert pose_error(pose,same)==0
    same[0]+=.01
    assert pose_error(pose,same)==pytest.approx(.01)


def test_container_route_uses_world_gravity_when_mesh_frame_is_rotated(monkeypatch):
    import sys
    from types import SimpleNamespace
    class Shape:
        def __init__(self,name):pass
        def get_bounding_box(self):return [-1,1,-2,2,-3,3]
        def get_matrix(self):return np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,10],[0,0,0,1.]])
        def get_mesh_data(self):return [],[],[]
    monkeypatch.setitem(sys.modules,'pyrep.objects.shape',SimpleNamespace(Shape=Shape))
    bridge=RLBenchBridge.__new__(RLBenchBridge);bridge.entities=[{'primary_name':'physical-container'}]
    target,approach=bridge.placement_route(0,{'high':[1,1,1],'low':[0,0,0]})
    assert target[2]==pytest.approx(12.525)
    assert approach[2]>target[2] and np.allclose(target[:2],approach[:2])


def test_public_inputs_do_not_expose_reference_names_or_native_goals():
    entities=[{'root_name':'target','movable':True,'reference':'secret'},
              {'root_name':'distractor','movable':False,'success_pose':[1,2,3]}]
    geometry=[{'center':[0,0,0]},{'center':[1,1,1]}]
    expected=public_input('put item in bin',entities,geometry,['raw','ids'],[0,1])
    changed=copy.deepcopy(entities);changed[0].update(root_name='different',reference='other target')
    assert public_input('put item in bin',changed,geometry,['raw','ids'],[0,1])==expected
    assert expected['entities']==['entity_000','entity_001']
    assert expected['movable']==['entity_000']
    assert not any(name in str(expected) for name in ('distractor','secret','success_pose'))


def test_executor_keeps_wrong_object_plan_and_never_asks_for_goal(monkeypatch):
    class Bridge:
        entities=[{},{}]
        def native_success(self):pytest.fail('executor cannot read goals')
    executor=RLBenchExecutor(Bridge());targets=[]
    def pick(index,destination_index=None):targets.append(index);raise RuntimeError('slip')
    monkeypatch.setattr(executor,'pick',pick)
    result=executor.execute([{'name':'pick','params':{'target_entity_name':1}}])
    assert targets==[1] and result['termination_reason']=='slip'
    result=executor.execute([{'name':'open','params':{'target_entity_name':0}}])
    assert result['termination_reason']=='unsupported skill'
    result=executor.execute([{'name':'pick','params':{'target_entity_name':True}}])
    assert result['termination_reason']=='invalid entity pointer'


def test_gripper_polarity_and_simulator_step_budget():
    class Bridge:
        entities=[]
        commands=[]
        def actuate_gripper(self,amount):self.commands.append(amount);return True
        def tick(self):pass
    bridge=Bridge();executor=RLBenchExecutor(bridge,horizon=2)
    executor.gripper(True);executor.gripper(False)
    assert bridge.commands==[1.,0.] and executor.steps==2
    with pytest.raises(TimeoutError,match='horizon'):executor.tick()


def test_native_attachment_only_checks_the_predicted_entity_and_preserves_pose(monkeypatch):
    import sys
    from types import SimpleNamespace
    poses={'wanted':np.array([1,2,3,0,0,0,1.]),'wrong':np.array([4,5,6,0,0,0,1.])}
    class Shape:
        def __init__(self,name):self.name=name
        def get_pose(self):return poses[self.name].copy()
    monkeypatch.setitem(sys.modules,'pyrep.objects.shape',SimpleNamespace(Shape=Shape))
    attempted=[]
    def grasp(shape):attempted.append(shape.name);return shape.name=='wrong'
    bridge=RLBenchBridge.__new__(RLBenchBridge)
    bridge.entities=[{'primary_name':'wanted','movable':True},{'primary_name':'wrong','movable':True}]
    bridge.env=SimpleNamespace(_robot=SimpleNamespace(gripper=SimpleNamespace(grasp=grasp)))
    assert bridge.try_grasp(1) and attempted==['wrong']
    assert not bridge.try_grasp(0) and attempted==['wrong','wanted']
    def teleport(shape):poses[shape.name][0]+=1;return True
    bridge.env._robot.gripper.grasp=teleport
    with pytest.raises(RuntimeError,match='world pose'):bridge.try_grasp(1)


def test_reset_does_not_step_physics_between_joints_or_reapply_robot_link_poses(monkeypatch):
    import sys
    from types import SimpleNamespace
    applied=[];joint_updates=[]
    class Obj:
        def __init__(self,name):self.name=name
        def get_name(self):return self.name
        def set_pose(self,pose,**kwargs):applied.append(self.name)
        def set_joint_position(self,value,disable_dynamics):
            assert disable_dynamics is False, 'native True would advance physics between coordinates'
            joint_updates.append((self.name,value))
        def set_joint_target_position(self,value):pass
        def set_joint_target_velocity(self,value):pass
    monkeypatch.setitem(sys.modules,'pyrep.objects.object',SimpleNamespace(Object=SimpleNamespace(get_object=Obj)))
    arm=SimpleNamespace(get_objects_in_tree=lambda:[Obj('joint1'),Obj('tip')],get_name=lambda:'robot_base',
                        get_joint_positions=lambda:[.2],set_joint_target_positions=lambda q:None)
    bridge=RLBenchBridge.__new__(RLBenchBridge);bridge.env=SimpleNamespace(_robot=SimpleNamespace(arm=arm))
    bridge.restore_state([{'name':'task_object','pose':[0]*7},{'name':'robot_base','pose':[0]*7},
                          {'name':'joint1','pose':[0]*7,'joint':.2},{'name':'tip','pose':[0]*7}])
    assert applied==['task_object','robot_base'] and joint_updates==[('joint1',.2)]


def test_contact_planning_exempts_only_predicted_body_and_restores_flags_on_failure(monkeypatch):
    import sys
    from types import SimpleNamespace
    flags={'predicted':True,'wall':True}
    class Shape:
        def __init__(self,name):self.name=name
        def is_collidable(self):return flags[self.name]
        def set_collidable(self,value):flags[self.name]=value
    monkeypatch.setitem(sys.modules,'pyrep.objects.shape',SimpleNamespace(Shape=Shape))
    def path(*args,**kwargs):
        assert flags=={'predicted':False,'wall':True}
        assert kwargs['ignore_collisions'] is False
        raise RuntimeError('no feasible path')
    bridge=RLBenchBridge.__new__(RLBenchBridge)
    bridge.entities=[{'parts':['predicted']},{'parts':['wall']}];bridge.contact_entity=0
    bridge.env=SimpleNamespace(_robot=SimpleNamespace(arm=SimpleNamespace(get_path=path)))
    with pytest.raises(RuntimeError,match='geometry-only'):bridge.path_to([0,0,0],[0,0,0,1],contact=True)
    assert flags=={'predicted':True,'wall':True}


def test_contact_planning_allows_local_movable_contact_but_protects_fixtures(monkeypatch):
    import sys
    from types import SimpleNamespace
    names=['predicted','nearby','far','wall']
    flags=dict.fromkeys(names,True)
    class Shape:
        def __init__(self,name):self.name=name
        def is_collidable(self):return flags[self.name]
        def set_collidable(self,value):flags[self.name]=value
        def set_respondable(self,value):pytest.fail('physical collision response must remain enabled')
    monkeypatch.setitem(sys.modules,'pyrep.objects.shape',SimpleNamespace(Shape=Shape))
    calls=[]
    def path(*args,**kwargs):
        calls.append(dict(flags))
        assert flags=={'predicted':False,'nearby':len(calls)==1,'far':True,'wall':True}
        assert kwargs['ignore_collisions'] is False
        raise RuntimeError('native planner failure')
    bridge=RLBenchBridge.__new__(RLBenchBridge);bridge.contact_entity=0
    bridge.entities=[{'parts':[name],'movable':name!='wall'} for name in names]
    boxes=[([0,0,0],[.1,.1,.1]),([.12,0,0],[.2,.1,.1]),([1,0,0],[1.1,.1,.1]),([0,0,0],[.1,.1,.1])]
    monkeypatch.setattr(bridge,'geometry',lambda i:{'low':np.array(boxes[i][0]),'high':np.array(boxes[i][1])})
    bridge.env=SimpleNamespace(_robot=SimpleNamespace(arm=SimpleNamespace(get_path=path)))
    with pytest.raises(RuntimeError,match='native planner failure'):bridge.path_to([0,0,1],[0,0,0,1],contact=True)
    assert all(flags.values())
    assert len(calls)==2
    assert bridge.planning_contacts[0]['kinematic_contact_entity_ids']==[0]
    assert bridge.planning_contacts[1]['kinematic_contact_entity_ids']==[0,1]
    assert not bridge.planning_contacts[0]['local_contact_fallback']
    assert bridge.planning_contacts[1]['local_contact_fallback']


def test_contact_planning_retains_nearby_obstacles_when_strict_path_exists(monkeypatch):
    import sys
    from types import SimpleNamespace
    flags={'predicted':True,'neighbor':True}
    class Shape:
        def __init__(self,name):self.name=name
        def is_collidable(self):return flags[self.name]
        def set_collidable(self,value):flags[self.name]=value
    monkeypatch.setitem(sys.modules,'pyrep.objects.shape',SimpleNamespace(Shape=Shape))
    result=object()
    def path(*args,**kwargs):
        assert flags=={'predicted':False,'neighbor':True}
        return result
    bridge=RLBenchBridge.__new__(RLBenchBridge);bridge.contact_entity=0
    bridge.entities=[{'parts':[name],'movable':True} for name in flags]
    monkeypatch.setattr(bridge,'geometry',lambda i:{'low':np.zeros(3),'high':np.ones(3)})
    bridge.env=SimpleNamespace(_robot=SimpleNamespace(arm=SimpleNamespace(get_path=path)))
    assert bridge.path_to([0,0,1],[0,0,0,1],contact=True) is result
    assert len(bridge.planning_contacts)==1 and not bridge.planning_contacts[0]['local_contact_fallback']
    assert all(flags.values())


def test_frozen_settings_reject_changed_bridge_or_predictions(tmp_path):
    p=tmp_path/'settings.json';save(p,{'bridge':'a','predictions':'x'})
    save(p,{'bridge':'a','predictions':'x'})
    with pytest.raises(ValueError,match='differs'):save(p,{'bridge':'b','predictions':'x'})
    with pytest.raises(ValueError,match='differs'):save(p,{'bridge':'a','predictions':'y'})


def test_failed_physical_pinch_settles_before_retry_geometry(monkeypatch):
    class Bridge:
        entities=[{}]
        z=0.;fall_ticks=None;calls=0
        pose=np.array([0.,0.,.5,0.,1.,0.,0.])
        def geometry(self,index):
            return {'movable':True,'center':np.array([0.,0.,self.z]),'low':[0.,0.,self.z-.02],
                    'high':[0.,0.,self.z+.02],'yaw':0.,'horizontal_span':.04}
        def base_position(self):return np.array([-1.,0.,0.])
        def tip_pose(self):return self.pose.copy()
        def set_contact_entity(self,index):assert index==0
        def try_grasp(self,index):self.calls+=1;return self.calls>2
        def tick(self):
            if self.fall_ticks is not None:
                self.fall_ticks-=1
                if self.fall_ticks<=0:self.z=0.;self.fall_ticks=None
    bridge=Bridge();executor=RLBenchExecutor(bridge);approaches=[]
    def move(position,*args,**kwargs):
        bridge.pose[:3]=position
        if executor.phase=='pregrasp':approaches.append(np.asarray(position).copy())
        if executor.phase=='grasp_approach' and bridge.calls<2:bridge.z=.2
        if executor.phase=='lift':bridge.z=.2
    def gripper(opened):
        if opened and bridge.z>0:bridge.fall_ticks=20
    monkeypatch.setattr(executor,'move',move);monkeypatch.setattr(executor,'gripper',gripper)
    executor.pick(0)
    assert executor.held==0 and len(approaches)==3
    assert approaches[1][2]==pytest.approx(approaches[0][2])


def test_pick_tries_equivalent_finger_axis_with_opposite_wrist_heading(monkeypatch):
    class Bridge:
        entities=[{}]
        z=0.
        pose=np.array([0.,0.,1.,0.,1.,0.,0.])
        def geometry(self,index):
            return {'movable':True,'center':np.array([0.,0.,self.z]),'low':[0.,0.,self.z-.02],
                    'high':[0.,0.,self.z+.02],'yaw':0.,'horizontal_span':.04}
        def base_position(self):return np.array([-1.,0.,0.])
        def tip_pose(self):return self.pose.copy()
        def set_contact_entity(self,index):assert index==0
        def try_grasp(self,index):assert index==0;return True
        def tick(self):pass
    bridge=Bridge();executor=RLBenchExecutor(bridge);attempts=[]
    flipped=downward_quaternion(np.pi)
    def move(position,*args,orientation=None,**kwargs):
        if executor.phase=='pregrasp':attempts.append(np.asarray(orientation).copy())
        if abs(np.dot(orientation,flipped))<.999:
            raise RuntimeError('this arm configuration cannot reach that wrist heading')
        bridge.pose[:3]=position;bridge.pose[3:]=orientation
        if executor.phase=='lift':bridge.z=.1
    monkeypatch.setattr(executor,'move',move);monkeypatch.setattr(executor,'gripper',lambda opened:None)
    executor.pick(0)
    assert executor.held==0 and len(attempts)==5
    assert all(event['entity_id']==0 for event in executor.events)
    from scipy.spatial.transform import Rotation
    first_axis=Rotation.from_quat(attempts[0]).as_matrix()[:,1]
    final_axis=Rotation.from_quat(attempts[-1]).as_matrix()[:,1]
    assert np.allclose(first_axis,-final_axis)


def test_lateral_grasp_keeps_finger_closing_axis_horizontal():
    from scipy.spatial.transform import Rotation
    radial=np.array([2.,1.,0.]);radial/=np.linalg.norm(radial)
    for roll in [0.,np.pi]:
        matrix=Rotation.from_quat(lateral_grasp_quaternion(radial,roll)).as_matrix()
        assert np.allclose(matrix[:,2],radial)
        assert abs(matrix[2,1])<1e-9
        assert np.linalg.det(matrix)==pytest.approx(1.)


def test_support_clearance_uses_rotated_hand_geometry_without_changing_scene(monkeypatch):
    from scipy.spatial.transform import Rotation
    import itertools
    bridge=RLBenchBridge.__new__(RLBenchBridge)
    pose=np.array([10.,20.,30.,0.,0.,0.,1.])
    points=np.array(list(itertools.product([-.1,.1],[-.02,.02],[-.03,.03])))
    monkeypatch.setattr(bridge,'tip_pose',lambda:pose.copy())
    monkeypatch.setattr(bridge,'assembly_points',lambda:points+pose[:3])
    for orientation in [[0.,0.,0.,1.],Rotation.from_euler('y',np.pi/2).as_quat()]:
        clearance=bridge.grasp_clearance(orientation)
        world=points@Rotation.from_quat(orientation).as_matrix().T+[0,0,clearance]
        assert world[:,2].min()==pytest.approx(.006)
    assert np.allclose(pose,[10.,20.,30.,0.,0.,0.,1.])


def test_rejected_descent_does_not_lift_an_already_clear_hover(monkeypatch):
    class Bridge:
        entities=[{}]
        pose=np.array([0.,0.,.5,0.,1.,0.,0.])
        def geometry(self,index):
            return {'movable':True,'center':np.array([0.,0.,.05]),'low':[0.,0.,0.],
                    'high':[0.,0.,.1],'yaw':0.,'horizontal_span':.04}
        def base_position(self):return np.array([-1.,0.,0.])
        def tip_pose(self):return self.pose.copy()
        def set_contact_entity(self,index):assert index==0
        def grasp_clearance(self,orientation):return .02
        def tick(self):pass
    bridge=Bridge();executor=RLBenchExecutor(bridge);unnecessary_lifts=[]
    def move(position,*args,**kwargs):
        if executor.phase=='pregrasp':bridge.pose[:3]=position;return
        if position[2]>bridge.pose[2]:unnecessary_lifts.append(position)
        raise RuntimeError('descent has no collision-free path')
    monkeypatch.setattr(executor,'move',move);monkeypatch.setattr(executor,'gripper',lambda opened:None)
    result=executor.execute([{'name':'pick','params':{'target_entity_name':0}}])
    assert result['termination_reason']=='grasp failed to lift the predicted object'
    assert not unnecessary_lifts and executor.steps<executor.horizon


@pytest.mark.parametrize('failure',[TimeoutError,RuntimeError])
def test_reorientation_stays_on_local_orientation_branch_and_keeps_anchor(monkeypatch,failure):
    class Bridge:
        pose=np.r_[[1.,2.,3.],downward_quaternion(0.)]
        def tip_pose(self):return self.pose.copy()
        def tick(self):pass
    bridge=Bridge();executor=RLBenchExecutor(bridge);anchors=[]
    direct=[]
    def move(position,*,orientation,stall_after=None,tolerance=.008):
        angle=2*np.arccos(np.clip(abs(np.dot(bridge.pose[3:],orientation)),0.,1.))
        if stall_after is not None:
            direct.append(orientation);raise failure('direct rotation failed')
        assert angle<=.201, 'large IK orientation jumps can choose a distant joint branch'
        anchors.append(np.asarray(position).copy());bridge.pose[:3]=position;bridge.pose[3:]=orientation
        executor.tick()
    monkeypatch.setattr(executor,'move',move)
    target=side_entry_quaternion([1.,0.,0.]);executor.reorient(target)
    assert len(direct)==1 and len(anchors)>1 and all(np.allclose(p,[1.,2.,3.]) for p in anchors)
    assert 2*np.arccos(np.clip(abs(np.dot(bridge.pose[3:],target)),0.,1.))<.04
    executor=RLBenchExecutor(bridge,horizon=1);bridge.pose[3:]=downward_quaternion(0.)
    def bounded_move(position,*,orientation,stall_after=None,tolerance=.008):
        if stall_after is not None:raise failure('direct rotation failed')
        bridge.pose[3:]=orientation;executor.tick()
    monkeypatch.setattr(executor,'move',bounded_move)
    with pytest.raises(TimeoutError,match='horizon'):executor.reorient(target)


def test_completed_but_stalled_trajectory_returns_control_before_motion_budget_expires():
    from types import SimpleNamespace
    bridge=SimpleNamespace(tick=lambda:None,tip_pose=lambda:np.r_[[0.,0.,0.],downward_quaternion(0.)],
                           path_to=lambda *args,**kwargs:SimpleNamespace(step=lambda:True))
    executor=RLBenchExecutor(bridge)
    with pytest.raises(TimeoutError,match='stalled'):
        executor.move([1.,0.,0.],orientation=downward_quaternion(0.),stall_after=3)
    assert executor.steps==4 and executor.steps<executor.horizon


def test_slow_but_progressing_tracking_is_not_mistaken_for_a_stall():
    from types import SimpleNamespace
    position=np.zeros(3)
    def tick():position[0]=min(.05,position[0]+.0002)
    bridge=SimpleNamespace(tick=tick,tip_pose=lambda:np.r_[position,downward_quaternion(0.)],
                           path_to=lambda *args,**kwargs:SimpleNamespace(step=lambda:True))
    executor=RLBenchExecutor(bridge);executor.move([.05,0.,0.],orientation=downward_quaternion(0.))
    assert 40<executor.steps<400


def test_hover_clearance_uses_nearby_movable_height_and_preserves_input(monkeypatch):
    bridge=RLBenchBridge.__new__(RLBenchBridge)
    bridge.entities=[{'movable':True},{'movable':True},{'movable':True},{'movable':False}]
    boxes=[([0,0,0],[.1,.1,.05]),([.12,0,0],[.2,.1,.3]),([1,0,0],[1.1,.1,2.]),([0,0,0],[.1,.1,10.])]
    monkeypatch.setattr(bridge,'geometry',lambda i:{'low':np.array(boxes[i][0]),'high':np.array(boxes[i][1])})
    monkeypatch.setattr(bridge,'grasp_clearance',lambda orientation:.03)
    proposed=np.array([.05,.05,.19]);raised=bridge.grasp_standoff(0,[0,0,0,1],proposed)
    assert np.allclose(raised,[.05,.05,.35]) and np.allclose(proposed,[.05,.05,.19])


@pytest.mark.parametrize('blocked',[False,True])
def test_higher_pregrasp_is_only_used_after_a_rejected_approach(monkeypatch,blocked):
    class Bridge:
        entities=[{}]
        z=.05;recoveries=0
        pose=np.r_[[0.,0.,.5],downward_quaternion(0.)]
        def geometry(self,index):return {'movable':True,'center':np.array([0.,0.,self.z]),
                                        'low':[0,0,0],'high':[.1,.1,.1],'yaw':0.,'horizontal_span':.04}
        def base_position(self):return np.array([-1.,0.,0.])
        def tip_pose(self):return self.pose.copy()
        def tick(self):pass
        def try_grasp(self,index):assert index==0;return True
        def grasp_standoff(self,index,orientation,proposed):
            assert blocked and index==0;self.recoveries+=1;return np.array([0.,0.,.5])
    bridge=Bridge();executor=RLBenchExecutor(bridge)
    def move(position,*args,**kwargs):
        if blocked and executor.phase=='pregrasp' and position[2]<.5:raise RuntimeError('hover intersects clutter')
        bridge.pose[:3]=position
        if executor.phase=='lift':bridge.z=.2
    monkeypatch.setattr(executor,'move',move);monkeypatch.setattr(executor,'gripper',lambda opened:None)
    executor.pick(0)
    assert executor.held==0 and bridge.recoveries==int(blocked)


def test_side_entry_releases_near_physical_support_instead_of_midair(monkeypatch):
    import sys,itertools
    from types import SimpleNamespace
    vertices=np.array(list(itertools.product([-.2,.2],[-.3,.3],[0.,.3])))
    triangles=[]
    for axis,value in [(0,.2),(1,-.3),(1,.3),(2,0.),(2,.3)]:
        face=[i for i,v in enumerate(vertices) if v[axis]==value]
        # Product ordering gives a rectangular grid: 0,1,3,2 is its perimeter.
        a,b,c,d=[face[i] for i in [0,1,3,2]]
        triangles.extend([[a,b,c],[a,c,d]])
    class Shape:
        def __init__(self,name):pass
        def get_bounding_box(self):return [-.2,.2,-.3,.3,0.,.3]
        def get_matrix(self):
            matrix=np.eye(4);matrix[2,3]=1.;return matrix
        def get_mesh_data(self):return vertices,triangles,[]
    monkeypatch.setitem(sys.modules,'pyrep.objects.shape',SimpleNamespace(Shape=Shape))
    bridge=RLBenchBridge.__new__(RLBenchBridge);bridge.env=object();bridge.entities=[{'primary_name':'fixture'}]
    points=np.array([[0.,0.,1.05],[0.,0.,1.20]])
    monkeypatch.setattr(bridge,'assembly_points',lambda:points.copy())
    held={'center':[0.,0.,1.1],'low':[-.05,-.05,1.05],'high':[.05,.05,1.15]}
    target,approach=bridge.placement_route(0,held,require_fit=True)
    translated=points+target-np.asarray(held['center'])
    assert 1.006<=translated[:,2].min()<=1.02
    assert translated[:,2].max()<1.294
    assert -.2<target[0]-.05 and target[0]+.05<.2
    assert approach[0]<-.2


def test_wide_flat_clearance_proposal_with_an_attachment_capable_test_sensor(monkeypatch):
    # Synthetic 50 mm detection isolates the clearance proposal. Native Panda
    # probes separately test acceptance/rejection; its gate may reject this pose.
    class Bridge:
        entities=[{}]
        lift=0.
        pose=np.r_[[0.,0.,.5],downward_quaternion(0.)]
        def geometry(self,index):return {'movable':True,'center':np.array([0.,0.,.0165+self.lift]),
            'low':[0.,0.,self.lift],'high':[.087,.087,.033+self.lift],'yaw':0.,'horizontal_span':.087}
        def base_position(self):return np.array([-.5,0.,0.])
        def tip_pose(self):return self.pose.copy()
        def tick(self):pass
        def try_grasp(self,index):
            assert index==0
            sensor_distance=self.pose[2]+.046-(.033+self.lift)
            return 0.<=sensor_distance<=.05
    bridge=Bridge();executor=RLBenchExecutor(bridge);contacts=[]
    def move(position,*args,**kwargs):
        bridge.pose[:3]=position
        if executor.phase=='grasp_approach':contacts.append(np.asarray(position).copy())
        if executor.phase=='lift':bridge.lift=.14
    monkeypatch.setattr(executor,'move',move);monkeypatch.setattr(executor,'gripper',lambda opened:None)
    executor.pick(0)
    assert executor.held==0 and len(contacts)==1
    tip_height=contacts[0][2]
    assert tip_height+.0005>.033, 'wide body must not intersect the open finger ends'
    assert tip_height+.046-.033<=.05, 'native attachment must still pass its proximity gate'


def test_placement_refreshes_realized_orientation_and_held_offset_after_staging(monkeypatch):
    from scipy.spatial.transform import Rotation
    nominal=side_entry_quaternion([1.,0.,0.])
    realized=(Rotation.from_rotvec([0.,0.,.1])*Rotation.from_quat(nominal)).as_quat()
    staged=(Rotation.from_rotvec([0.,0.,.15])*Rotation.from_quat(nominal)).as_quat()
    class Bridge:
        entities=[{},{}]
        pose=np.r_[[0.,0.,1.],downward_quaternion(0.)]
        offset=.05
        def tip_pose(self):return self.pose.copy()
        def geometry(self,index):
            center=self.pose[:3]-[0,0,self.offset]
            return {'center':center,'low':center-.01,'high':center+.01}
        def placement_route(self,index,held,require_fit=False):
            return np.array([1.,0.,1.]),np.array([.8,0.,1.])
        def entry_orientation(self,inward):return nominal.copy()
    bridge=Bridge();executor=RLBenchExecutor(bridge);executor.held=0;release=[]
    def reorient(orientation):bridge.pose[3:]=realized
    def move(position,*args,orientation=None,**kwargs):
        if executor.phase=='preplace':
            assert np.allclose(orientation,realized) and kwargs['tolerance']==.025
            bridge.pose[:3]=np.asarray(position)-[.02,0,0]
            bridge.pose[3:]=staged;bridge.offset=.08
        elif executor.phase=='insert_or_lower':
            assert np.allclose(orientation,staged)
            assert kwargs.get('tolerance',.008)==.008
            bridge.pose[:3]=position;release.append(bridge.geometry(0)['center'].copy())
        else:bridge.pose[:3]=position
    monkeypatch.setattr(executor,'move',move);monkeypatch.setattr(executor,'reorient',reorient)
    monkeypatch.setattr(executor,'gripper',lambda opened:None);monkeypatch.setattr(executor,'settle',lambda count:None)
    executor.place(1)
    assert len(release)==1 and np.allclose(release[0],[1.,0.,1.])
