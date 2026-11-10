"""Goal-blind geometry search in a separate native simulator process.

Only the worker performs hypothetical FK, parenting and scene restoration.
The live process sends physical state; no reference, waypoint or success sensor
is part of this interface. Native attachment is still checked live at contact.
"""
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

PLANNING_PROFILE = {
    'version': 3, 'thin_ratio': .5, 'wide_opening_fraction': .94,
    'rim_fraction': .85, 'grasp_depth': .018,
    'tilts_degrees': [0, 15, 30], 'seek_step': .001, 'seek_limit': .010,
    'stable_tilt_degrees': 15, 'settle_window': 15,
    'settle_translation': .002, 'settle_rotation_degrees': 2,
    'support_gap': .012, 'worker_timeout_seconds': 660, 'search_budget_seconds':600,
    'thin_front_margins': [.04,.025,.01], 'thin_release_offsets':[.001,.005,.01],
    'ordinary_release_offsets':[.01,.025,.04],
    'lateral_rolls_degrees':[45,90,135,-45,-90,-135],
    'rim_penetrations':[.002,.008], 'top_grasp_tilts_degrees':[75,-75,60,-60,45,-45,30,-30,15,-15,0],
    'shadow_effort':{'trials':60,'max_configs':2,'trials_per_goal':1},
    'runtime_motion_fallback_effort':{'trials':300,'max_configs':8,'trials_per_goal':3},
    'validated_grasps_per_request':1,
    'flat_body_policy':'thin_geometry_with_observed_short_axis_support',
    'structural_fit':'payload_containment_and_spatial_hand_payload_mesh_collision',
    'release_separation':'native_detach_then_physical_withdrawal_then_clearance_and_support',
    'reorientation_modes':['rotate_before_transport','transport_then_rotate'],
    'motion_execution':'checked_shadow_joint_path_from_measured_state',
    'grasp_attempts':3,'planned_grasp_tolerance':.006,'support_geometry':'physical_mesh',
    'settle_limit':120,'existing_support_contact_tolerance':.003,
    'mesh_api':'local_safe_ffi_mesh_buffers_no_upstream_normals_binding',
    'payload_floor_clearance':.003,'landing_tolerance':.008,
    'release_dwell_steps':30,'grasp_order':'thin_interleaved_level_sides_and_top',
    'top_contact_offsets':[.002,-.006,-.018],
    'contact_screen':'rigid_relative_native_detector_before_ik',
    'destination_budget_per_grasp_seconds':12,
    'shelf_yaw_candidates':'continuous_hand_forward_alignment_then_quarter_turns',
    'staging_frame':'tip_from_rotated_payload_center_and_measured_grasp',
    'thin_lateral_rolls_degrees':[0,180,15,-15,30,-30],
    'thin_lateral_headings':['opposite_radial','left','right','radial'],
    'joint_interval_convention':'native_minimum_plus_range',
    'checked_path_attempts':3,
    'planned_contact_stall_ticks':20,
    'regrasp_recovery':{'maximum':1,'radial_distances':[.5,.6],
                        'azimuth_offsets_degrees':[0,30,-30,60,-60],
                        'occupied_margin':.04,'lift_clearance':.15,'direct_search_seconds':120,'minimum_remaining_steps':600,
                        'site_lookahead_seconds':45},
    'acquisition_lift_frame':'measured_vertical_after_contact_not_nominal_grasp_pose',
    'candidate_trajectory_execution':'reuse_checked_joint_paths_revalidate_measured_scene_before_each_segment',
    'contact_object_motion_threshold':.005,
    'release_readiness':{'hold_settle_steps':10,'maximum_drop':.02,'maximum_corrections':1,'recovery_reserve_steps':200},
    'native_release_timing':'open_fingers_under_native_attachment_then_check_actual_clearance_then_detach_before_withdrawal',
    'settling_finger_clearance':.002,
    'finger_contact_control':{'close_velocity':.02,'open_velocity':.04,'stop_distance':.0005,'maximum_ticks':80},
    'contact_frame_replans':12,
}


def matrix(pose):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def thin_body(geometry):
    dimensions = np.asarray(geometry['dimensions'])
    return bool(dimensions.min() / np.partition(dimensions, 1)[1] <= PLANNING_PROFILE['thin_ratio'])


def face_stability_required(geometry,support_axis):
    # An upright carton can have one thin dimension without resting on that
    # face. Do not impose a flat-can policy on every thin bounding box.
    cap=geometry.get('symmetry_axis')
    return ((thin_body(geometry) and support_axis==int(np.argmin(geometry['dimensions']))) or
            (cap is not None and cap==support_axis))


def stable_orientations(object_matrix, dimensions, inward, support_axis=None):
    """Face-stable rotations first; no assumption about a model's local Z."""
    short = int(np.argmin(dimensions)) if support_axis is None else support_axis
    axis = object_matrix[:3, short].copy()
    if axis[2] < 0:
        axis = -axis
    cross = np.cross(axis, [0., 0., 1.])
    angle = np.arccos(np.clip(axis[2], -1., 1.))
    align = Rotation.identity() if np.linalg.norm(cross) < 1e-8 else Rotation.from_rotvec(cross / np.linalg.norm(cross) * angle)
    flat = align * Rotation.from_matrix(object_matrix[:3, :3])
    inward = np.asarray(inward, dtype=float).copy(); inward[2] = 0
    if np.linalg.norm(inward) < 1e-8:
        inward = np.array([1., 0., 0.])
    tilt_axis = np.cross([0., 0., 1.], -inward / np.linalg.norm(inward))
    for tilt in PLANNING_PROFILE['tilts_degrees']:
        for yaw in (0., np.pi/2, -np.pi/2, np.pi):
            rotation = (Rotation.from_rotvec(tilt_axis*np.deg2rad(tilt)) *
                        Rotation.from_rotvec([0., 0., yaw]) * flat)
            yield tilt, rotation.as_matrix()


def release_stable(samples, *, support, require_face=False):
    """Check observed support/motion, independently of native task success."""
    if len(samples) < PLANNING_PROFILE['settle_window']:
        return False
    window = samples[-PLANNING_PROFILE['settle_window']:]
    poses = np.asarray([s['pose'] for s in window])
    translation = float(np.max(np.linalg.norm(poses[:, :3]-poses[-1, :3], axis=1)))
    rotations = Rotation.from_quat(poses[:, 3:])
    angular = float(np.max((rotations[-1].inv()*rotations).magnitude()))
    gap = abs(window[-1]['bottom']-support)
    return bool(translation <= PLANNING_PROFILE['settle_translation'] and
                not any(s.get('attached',False) for s in window) and
                angular <= np.deg2rad(PLANNING_PROFILE['settle_rotation_degrees']) and
                gap <= PLANNING_PROFILE['support_gap'] and
                (not require_face or window[-1]['face_tilt'] <= np.deg2rad(PLANNING_PROFILE['stable_tilt_degrees'])))


def aligned_shelf_rotations(rotations,relative,inward):
    """Use the opening direction to remove arbitrary quarter-turn yaw error."""
    inward=np.asarray(inward,dtype=float)
    result=[];seen=set()
    for tilt,rotation in rotations:
        forward=(rotation@relative[:3,:3].T)[:,2]
        if min(np.linalg.norm(forward[:2]),np.linalg.norm(inward[:2]))<1e-6:continue
        angle=np.arctan2(inward[1],inward[0])-np.arctan2(forward[1],forward[0])
        aligned=Rotation.from_rotvec([0.,0.,angle]).as_matrix()@rotation
        key=(tilt,tuple(aligned.round(6).ravel()))
        if key not in seen:seen.add(key);result.append((tilt,aligned))
    return result+rotations


class PlanningClient:
    def __init__(self):
        parent, child = socket.socketpair()
        self.socket = parent
        parent.settimeout(PLANNING_PROFILE['worker_timeout_seconds'])
        self.process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(child.fileno())],
                                        pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL,
                                        stdout=sys.stderr, stderr=sys.stderr,start_new_session=True)
        child.close()
        self.stream = parent.makefile('rwb')

    def request(self, value):
        try:
            self.stream.write((json.dumps(value)+'\n').encode()); self.stream.flush()
            line = self.stream.readline()
        except OSError as exc:
            raise RuntimeError('planning worker communication failed: '+str(exc)) from exc
        if not line:
            raise RuntimeError('planning worker exited')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError('planning worker: '+result['error'])
        return result

    def close(self):
        import os,signal
        try:self.stream.close()
        except OSError:pass
        self.socket.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:os.killpg(self.process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(self.process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                self.process.wait(timeout=5)
        # The native process can crash before its finally block, leaving Xvfb
        # alive after the group leader exited. Reap only this owned group.
        try:os.killpg(self.process.pid,signal.SIGTERM)
        except ProcessLookupError:pass


class ShadowPlanner:
    def __init__(self, bridge):
        self.bridge = bridge
        self.bridge.planning_effort=PLANNING_PROFILE['shadow_effort']

    def sync(self, request):
        self.bridge.env._robot.gripper.release()
        self.bridge.reset(*request['scene'], state=request['state'], entities=request['entities'])
        self.bridge.set_contact_entity(request['object'])
        self.support_axis=request.get('support_axis')
        self.bridge.observed_support_axes[request['object']]=self.support_axis
        self.bridge.placement_options=request.get('placement_options',{})
        self.bridge.linear_motion=request.get('withdraw',False)
        from pyrep.objects.object import Object
        from pyrep.const import ObjectType
        robot={o.get_name() for o in self.bridge.env._robot.arm.get_objects_in_tree(exclude_base=False)}
        movable={name for entity in self.bridge.entities if entity['movable'] for name in entity['parts']}
        self.structures=[o for o in Object._get_objects_in_tree(object_type=ObjectType.SHAPE)
                         if o.get_name() not in robot|movable and o.is_respondable() and not o.is_dynamic()]

    def attach_copy(self, index):
        # Hypothetical attachment ONLY in the worker, never live acquisition.
        from pyrep.objects.shape import Shape
        shape = Shape(self.bridge.entities[index]['primary_name'])
        gripper = self.bridge.env._robot.gripper
        gripper._old_parents = [shape.get_parent()]
        gripper._grasped_objects = [shape]
        shape.set_parent(gripper._attach_point, keep_in_place=True)

    def path(self, position, orientation):
        # Native IK/RRT is stochastic: a newly sampled path can hit a wall even
        # after candidate search found a safe route. Retry from the SAME shadow
        # state, never from the failed FK configuration and never unchecked.
        if not hasattr(self.bridge,'snapshot_state'):
            return self._path_once(position,orientation)
        state=self.bridge.snapshot_state()
        attached=bool(self.bridge.env._robot.gripper.get_grasped_objects())
        failure=None;effort=self.bridge.planning_effort
        try:
            for attempt in range(PLANNING_PROFILE['checked_path_attempts']):
                self.check_budget()
                if attempt and getattr(self,'current_operation',None)=='motion':
                    self.bridge.planning_effort=PLANNING_PROFILE['runtime_motion_fallback_effort']
                try:return self._path_once(position,orientation)
                except (ValueError,RuntimeError) as exc:
                    failure=exc
                    if getattr(self,'proposed_joint_path',None) is not None:
                        print('[shadow-cache] rejected='+str(exc),file=sys.stderr,flush=True)
                self.bridge.env._robot.gripper.release()
                self.bridge.restore_state(state)
                if attached:self.attach_copy(self.bridge.contact_entity)
                if getattr(self,'current_operation',None)=='motion':self.proposed_joint_path=None
                # Candidate search should try another pose, rather than repeat
                # an unreachable IK endpoint three times. Only runtime motion
                # escalates the sampling effort for a previously accepted pose.
                if isinstance(failure,RuntimeError) and getattr(self,'current_operation',None)!='motion':break
        finally:self.bridge.planning_effort=effort
        raise failure

    def _path_once(self, position, orientation):
        """Check all path configurations, including payload/arm collisions."""
        from pyrep.objects.shape import Shape
        self.check_budget()
        arm = self.bridge.env._robot.arm
        initial=self.bridge.geometry(self.bridge.contact_entity) if hasattr(self.bridge,'geometry') else None
        initial_tip=self.bridge.tip_pose()[:3].copy() if initial is not None else None
        initial_contacts={}
        if initial is not None and getattr(self.bridge.env._robot.gripper,'get_grasped_objects',lambda:[])():
            payload=Shape(self.bridge.entities[self.bridge.contact_entity]['primary_name'])
            for structure in getattr(self,'structures',[]):
                # Native collision toggles at micron-scale FK roundoff when a
                # body is already resting on a surface. Establish support from
                # measured geometry before planning, not penetration alone.
                try:
                    support=self.bridge.shape_support_height(structure,initial['center'])
                    if abs(initial['low'][2]-support)<=PLANNING_PROFILE['existing_support_contact_tolerance']:
                        initial_contacts[structure.get_name()]=support
                except ValueError:pass
        pose=self.bridge.tip_pose() if hasattr(self.bridge,'tip_pose') else None
        same_pose=(pose is not None and np.linalg.norm(np.asarray(position)-pose[:3])<1e-5 and
                   (Rotation.from_quat(pose[3:]).inv()*Rotation.from_quat(orientation)).magnitude()<1e-4)
        cached=getattr(self,'proposed_joint_path',None)
        if cached is not None:
            points=np.asarray(cached,dtype=float)
            current=np.asarray(arm.get_joint_positions())
            if points.ndim!=2 or points.shape[1]!=len(arm.joints) or not len(points) or not np.isfinite(points).all():
                raise ValueError('invalid cached joint path')
            delta=points[0]-current
            if np.max(np.abs(delta))>.03:raise ValueError('cached path start drift; replan required')
            count=max(1,int(np.ceil(np.max(np.abs(delta))/.005)))
            prefix=np.linspace(current,points[0],count+1)[:-1]
            points=np.concatenate([prefix,points])
            exempt=self.bridge.local_contact_exemptions()
        elif same_pose:points=np.asarray(arm.get_joint_positions()).reshape(1,-1)
        else:
            path = self.bridge.path_to(position, orientation, contact=True)
            points = np.asarray(path._path_points).reshape(-1, len(arm.joints))
        cyclic, intervals = arm.get_joint_intervals()
        intervals = np.asarray(intervals)
        payload = Shape(self.bridge.entities[self.bridge.contact_entity]['primary_name'])
        hand_names = {s.get_name() for s in self.bridge.env._robot.gripper.get_objects_in_tree(exclude_base=False)}
        from pyrep.const import ObjectType
        arm_shapes = [s for s in arm.get_objects_in_tree(object_type=ObjectType.SHAPE)
                      if s.is_respondable() and s.get_name() not in hand_names]
        for configuration_number,configuration in enumerate(points):
            noncyclic = ~np.asarray(cyclic, dtype=bool)
            if np.any(configuration[noncyclic] < intervals[noncyclic, 0]-1e-5) or np.any(configuration[noncyclic] > intervals[noncyclic, 0]+intervals[noncyclic, 1]+1e-5):
                raise ValueError('joint limit: configuration=%d q=%s minimum=%s maximum=%s' % (
                    configuration_number,configuration.tolist(),intervals[:,0].tolist(),(intervals[:,0]+intervals[:,1]).tolist()))
            arm.set_joint_positions(configuration, disable_dynamics=False)
            if cached is not None:self.bridge.validate_arm_configuration(exempt)
            for part in arm_shapes:
                if payload.check_collision(part):raise ValueError('payload/wrist or arm collision: '+part.get_name())
            # Native path generation checks robot/fixtures. The held payload
            # may not belong to that collision collection, so check it too.
            structures=getattr(self,'structures',None)
            if structures is None:
                structures=[Shape(name) for entity in self.bridge.entities if not entity['movable'] for name in entity['parts']]
            hand=[s for s in self.bridge.env._robot.gripper.get_objects_in_tree(object_type=ObjectType.SHAPE,exclude_base=False)
                  if s.is_respondable() and s.get_name() not in self.bridge.entities[self.bridge.contact_entity]['parts']]
            for structure in structures:
                holding=bool(getattr(self.bridge.env._robot.gripper,'get_grasped_objects',lambda:[])())
                if holding and payload.check_collision(structure):
                    try:from .rlbench_geometry import separating_support_contact
                    except ImportError:from rlbench_geometry import separating_support_contact
                    support=initial_contacts.get(structure.get_name())
                    current=self.bridge.geometry(self.bridge.contact_entity) if support is not None else None
                    allowed=support is not None and separating_support_contact(
                        initial_bottom=initial['low'][2],bottom=current['low'][2],support=support,
                        initial_center=initial['center'],center=current['center'],initial_tip=initial_tip,
                        tip=self.bridge.tip_pose()[:3],tolerance=PLANNING_PROFILE['existing_support_contact_tolerance'])
                    if not allowed:
                        detail=''
                        if hasattr(self.bridge,'geometry'):
                            measured=self.bridge.geometry(self.bridge.contact_entity)
                            detail=' configuration=%d/%d bottom=%.6f initial_bottom=%.6f support=%s center_delta=%s tip_delta=%s' % (configuration_number,len(points),measured['low'][2],initial['low'][2],support,(measured['center']-initial['center']).tolist(),(self.bridge.tip_pose()[:3]-initial_tip).tolist())
                        raise ValueError('payload/structure collision: '+structure.get_name()+detail)
                if any(part.check_collision(structure) for part in hand):raise ValueError('hand/structure collision: '+structure.get_name())
        if cached is not None:
            actual=self.bridge.tip_pose()
            if (np.linalg.norm(actual[:3]-position)>getattr(self,'target_tolerance',.008) or
                    abs(np.dot(actual[3:],orientation))<.995):
                raise ValueError('cached path endpoint changed; replan required')
        return points.tolist()

    def check_budget(self):
        if time.monotonic()>getattr(self,'deadline',float('inf')):
            raise TimeoutError('geometry search budget exhausted; feasibility unverified')
        if time.monotonic()>getattr(self,'candidate_deadline',float('inf')):
            raise RuntimeError('candidate destination budget exhausted; try another grasp')
        if time.monotonic()>getattr(self,'site_deadline',float('inf')):
            raise RuntimeError('staging site has no verified onward grasp within its budget')

    def check_release(self,index):
        from pyrep.const import ObjectType
        from pyrep.objects.shape import Shape
        gripper=self.bridge.env._robot.gripper
        body=Shape(self.bridge.entities[index]['primary_name']);pose=np.asarray(body.get_pose()).copy()
        _,intervals=gripper.get_joint_intervals()
        closed=gripper.get_joint_positions()
        try:
            bounds=np.asarray(intervals)
            gripper.set_joint_positions(bounds[:,0]+bounds[:,1],disable_dynamics=False)
            body.set_pose(pose.tolist())
            if not self.bridge.release_clearance(index):raise ValueError('release finger clearance')
            for part in gripper.get_objects_in_tree(object_type=ObjectType.SHAPE,exclude_base=False):
                if not part.is_respondable() or part.get_name() in self.bridge.entities[index]['parts']:continue
                structures=getattr(self,'structures',[Shape(name) for entity in self.bridge.entities if not entity['movable'] for name in entity['parts']])
                if any(part.check_collision(structure) for structure in structures):
                    raise ValueError('open hand/structure collision at release')
        finally:
            gripper.set_joint_positions(closed,disable_dynamics=False);body.set_pose(pose.tolist())

    def check_settling_clearance(self,index,support,*,actual_open=False):
        """Open fingers must clear the entire gravity-settling envelope."""
        from pyrep.objects.shape import Shape
        body=Shape(self.bridge.entities[index]['primary_name']);pose=np.asarray(body.get_pose()).copy()
        gripper=self.bridge.env._robot.gripper;closed=gripper.get_joint_positions()
        _,intervals=gripper.get_joint_intervals();bounds=np.asarray(intervals)
        gap=max(0.,self.bridge.geometry(index)['low'][2]-support)
        try:
            if not actual_open:gripper.set_joint_positions(bounds[:,0]+bounds[:,1],disable_dynamics=False)
            for distance in np.linspace(0.,gap,max(1,int(np.ceil(gap/.001)))+1):
                hypothetical=pose.copy();hypothetical[2]-=distance;body.set_pose(hypothetical.tolist())
                clear=self.bridge.release_clearance(index)
                if clear and hasattr(self.bridge,'finger_clearance'):
                    clear=self.bridge.finger_clearance(index)>=PLANNING_PROFILE['settling_finger_clearance']
                if not clear:
                    raise ValueError('open fingers obstruct settling onto support: gap=%.6f distance=%.6f actual_open=%s' % (gap,distance,actual_open))
        finally:
            gripper.set_joint_positions(closed,disable_dynamics=False);body.set_pose(pose.tolist())

    def verify_placement(self,index,destination):
        body=self.bridge.geometry(index)
        sample=self.bridge.release_sample(index)
        if face_stability_required(body,self.support_axis) and sample['face_tilt']>np.deg2rad(PLANNING_PROFILE['stable_tilt_degrees']):
            raise ValueError('actual held face is not stable for shelf release')
        target,approach=self.bridge.placement_route(destination,body,require_fit=True)
        pose=self.bridge.tip_pose();offset=pose[:3]-body['center']
        release=target+offset;above=approach+offset
        insertion=self.path(release,pose[3:]);self.check_release(index)
        self.bridge.env._robot.gripper.release()
        retreat=self.path(above,pose[3:])
        return {'release':release.tolist(),'approach':above.tolist(),'orientation':pose[3:].tolist(),
                'insertion_joint_path':insertion,'retreat_joint_path':retreat}

    def verify_release(self,index,destination,retreat_position,*,actual_open=False):
        """Check the REALIZED inserted body before releasing the live gripper."""
        body=self.bridge.geometry(index);sample=self.bridge.release_sample(index)
        if face_stability_required(body,self.support_axis) and sample['face_tilt']>np.deg2rad(PLANNING_PROFILE['stable_tilt_degrees']):
            raise ValueError('actual inserted body is not face-stable; do not release')
        support=self.bridge.support_height(destination,body['center'])
        if not support-.003<=body['low'][2]<=support+PLANNING_PROFILE['release_readiness']['maximum_drop']:
            raise ValueError('actual inserted body is not near the physical support')
        # A corner over the opening edge can drop outside during withdrawal.
        for x in (body['low'][0],body['high'][0]):
            for y in (body['low'][1],body['high'][1]):
                height=self.bridge.support_height(destination,np.array([x,y,body['center'][2]]))
                if abs(height-support)>.002:raise ValueError('actual inserted footprint overhangs support')
        self.check_release(index)
        self.check_settling_clearance(index,support,actual_open=actual_open)
        if retreat_position is None:return {'sample':sample,'support':support}
        gripper=self.bridge.env._robot.gripper
        gripper.release();_,intervals=gripper.get_joint_intervals();bounds=np.asarray(intervals)
        gripper.set_joint_positions(bounds[:,0]+bounds[:,1],disable_dynamics=False)
        orientation=self.bridge.tip_pose()[3:]
        retreat=self.path(retreat_position,orientation)
        return {'sample':sample,'support':support,'retreat_joint_path':retreat}

    def grasp_proposals(self,body):
        try:from .rlbench_executor import downward_quaternion, tilted_downward_quaternion, lateral_grasp_quaternion
        except ImportError:from rlbench_executor import downward_quaternion, tilted_downward_quaternion, lateral_grasp_quaternion
        proposals=[]
        radial=np.asarray(body['center'])-self.bridge.base_position();radial[2]=0.;radial/=np.linalg.norm(radial)
        rolls=(PLANNING_PROFILE['thin_lateral_rolls_degrees'] if thin_body(body) else [])+PLANNING_PROFILE['lateral_rolls_degrees']
        left=np.array([-radial[1],radial[0],0.])
        headings=[-radial,left,-left,radial] if thin_body(body) else [radial]
        for degrees in rolls:
            for heading in headings:
                orientation=lateral_grasp_quaternion(heading,np.deg2rad(degrees))
                for penetration in PLANNING_PROFILE['rim_penetrations']:
                    point=np.asarray(body['center']).copy()-heading*(body['horizontal_span']/2-penetration)
                    point[2]=max(point[2],body['low'][2]+self.bridge.grasp_clearance(orientation))
                    proposals.append((point,orientation,body['yaw']))
        radial_yaw=float(np.arctan2(radial[1],radial[0]))
        for yaw in (radial_yaw,radial_yaw+np.pi,body['yaw'],body['yaw']+np.pi,body['yaw']+np.pi/2,body['yaw']-np.pi/2):
            for degrees in PLANNING_PROFILE['top_grasp_tilts_degrees']:
                for side in ((1.,-1.) if degrees<0 else (-1.,1.)):
                    point=np.asarray(body['center']).copy()
                    point+=Rotation.from_quat(downward_quaternion(yaw)).as_matrix()[:,0]*side*body['horizontal_span']/2*PLANNING_PROFILE['rim_fraction']
                    point[2]=body['high'][2]-PLANNING_PROFILE['grasp_depth']
                    orientation=tilted_downward_quaternion(yaw,point-self.bridge.base_position(),np.deg2rad(degrees))
                    point[2]=max(point[2],body['low'][2]+self.bridge.grasp_clearance(orientation))
                    proposals.append((point,orientation,yaw))
        if thin_body(body):
            # Keep both shallow top and level side contacts in play for a low
            # body; heavily rolled wrists can collide with its support plane.
            lateral_count=len(rolls)*len(headings)*len(PLANNING_PROFILE['rim_penetrations'])
            top=proposals[lateral_count:]
            tangent=[]
            for offset in PLANNING_PROFILE['top_contact_offsets']:
                for yaw in (radial_yaw,radial_yaw+np.pi,body['yaw'],body['yaw']+np.pi):
                    for degrees in (0.,-15.,15.,-30.,30.):
                        point=np.asarray(body['center']).copy()
                        point[2]=body['high'][2]+offset
                        q=tilted_downward_quaternion(yaw,point-self.bridge.base_position(),np.deg2rad(degrees))
                        point[2]=max(point[2],body['low'][2]+self.bridge.grasp_clearance(q))
                        tangent.append((point,q,yaw))
            lateral=proposals[:len(proposals)-len(top)]
            # Do not let a large family of top approaches starve lateral/rim
            # candidates when its grasp transform cannot fit a shelf.
            import itertools
            proposals=[candidate for group in itertools.zip_longest(lateral,top,tangent)
                       for candidate in group if candidate is not None]
        return proposals

    def contact_feasible(self,index,position,orientation):
        """Screen relative hand/body contact in this copy, before expensive IK.

        Move only the hypothetical payload into the *current* hand frame. The
        rigid transform preserves its relative pose to the proposed hand and
        uses the actual native detector and actual fingers, not an AABB gate.
        Structure/reachability checks still follow on the real candidate FK.
        """
        from pyrep.objects.shape import Shape
        body=Shape(self.bridge.entities[index]['primary_name'])
        original=np.asarray(body.get_matrix()).copy()
        homogeneous=np.eye(4);homogeneous[:3,:]=original[:3,:]
        transformed=matrix(self.bridge.tip_pose())@np.linalg.inv(matrix(np.r_[position,orientation]))@homogeneous
        try:
            body.set_matrix(transformed[:3,:])
            return self.bridge.detect_grasp(index) and self.bridge.release_clearance(index)
        finally:body.set_matrix(original)

    def staging_sites(self,index):
        """Clear physical support sites; no task goal, waypoint, or entity alias."""
        body=self.bridge.geometry(index);base=self.bridge.base_position()
        radial=body['center']-base;radial[2]=0
        if np.linalg.norm(radial)<1e-6:return []
        observed=[]
        for structure in self.structures:
            try:observed.append((self.bridge.shape_support_height(structure,body['center']),structure.get_name()))
            except ValueError:pass
        if not observed:return []
        original_height,original_support=max(observed)
        heading=np.arctan2(radial[1],radial[0]);sites=[]
        profile=PLANNING_PROFILE['regrasp_recovery']
        for distance in profile['radial_distances']:
            for degrees in profile['azimuth_offsets_degrees']:
                angle=heading+np.deg2rad(degrees)
                point=np.array([base[0]+distance*np.cos(angle),base[1]+distance*np.sin(angle),body['center'][2]])
                radius=float(np.max(body['dimensions'])/2)+profile['occupied_margin']
                occupied=False
                for other,row in enumerate(self.bridge.entities):
                    if other==index or not row['movable']:continue
                    bounds=self.bridge.geometry(other)
                    if np.all(point[:2]+radius>bounds['low'][:2]) and np.all(point[:2]-radius<bounds['high'][:2]):
                        occupied=True;break
                if occupied:continue
                supports=[]
                for structure in self.structures:
                    if structure.get_name()!=original_support:continue
                    try:
                        height=self.bridge.shape_support_height(structure,point)
                        if abs(height-original_height)>.002:continue
                        corners=[point+np.array([x,y,0.]) for x in (-radius,radius) for y in (-radius,radius)]
                        if any(abs(self.bridge.shape_support_height(structure,c)-height)>.002 for c in corners):continue
                        supports.append((height,structure.get_name()))
                    except ValueError:continue
                if supports:
                    height,name=max(supports)
                    sites.append({'point':point.tolist(),'height':height,'structure':name})
        return sites

    def staging(self,index,destination):
        """Plan a release on a clear support so the same body can be regrasped."""
        from pyrep.objects.shape import Shape
        bridge=self.bridge;body=bridge.geometry(index);state=bridge.snapshot_state()
        tip=matrix(bridge.tip_pose());obj=np.asarray(Shape(bridge.entities[index]['primary_name']).get_matrix())
        relative=np.linalg.inv(tip)@obj;center_in_hand=np.linalg.inv(tip)@np.r_[body['center'],1.]
        rotations=list(stable_orientations(obj,body['dimensions'],[1.,0.,0.],self.support_axis))
        failures=[]
        for site in self.staging_sites(index):
            for tilt,rotation,mode in [(t,r,m) for t,r in rotations for m in PLANNING_PROFILE['reorientation_modes']]:
                self.check_budget()
                if tilt!=0:continue
                bridge.env._robot.gripper.release();bridge.restore_state(state);self.attach_copy(index)
                hand=rotation@relative[:3,:3].T;q=Rotation.from_matrix(hand).as_quat()
                import itertools
                corners=np.array(list(itertools.product(*zip(body['low'],body['high']))))
                rotated=(corners-body['center'])@obj[:3,:3]@rotation.T
                target=np.asarray(site['point']).copy();target[2]=site['height']-rotated[:,2].min()+PLANNING_PROFILE['payload_floor_clearance']
                release=target-hand@center_in_hand[:3];above=release+[0,0,PLANNING_PROFILE['regrasp_recovery']['lift_clearance']]
                if mode=='rotate_before_transport':
                    waypoints=[{'position':bridge.tip_pose()[:3].tolist(),'orientation':q.tolist()},
                               {'position':above.tolist(),'orientation':q.tolist()}]
                else:
                    waypoints=[{'position':above.tolist(),'orientation':bridge.tip_pose()[3:].tolist()},
                               {'position':above.tolist(),'orientation':q.tolist()}]
                try:
                    for waypoint in waypoints:waypoint['joint_path']=self.path(waypoint['position'],waypoint['orientation'])
                    lower=self.path(release,q);self.check_release(index)
                    bridge.env._robot.gripper.release()
                    _,intervals=bridge.env._robot.gripper.get_joint_intervals();bounds=np.asarray(intervals)
                    bridge.env._robot.gripper.set_joint_positions(bounds[:,0]+bounds[:,1],disable_dynamics=False)
                    retreat=self.path(above,q)
                    # Check the NEXT grasp and destination in this copy before
                    # physically committing to a merely empty staging site.
                    resting=Shape(bridge.entities[index]['primary_name'])
                    pose=np.asarray(resting.get_pose()).copy()
                    pose[2]-=bridge.geometry(index)['low'][2]-site['height']
                    resting.set_pose(pose.tolist())
                    self.site_deadline=time.monotonic()+PLANNING_PROFILE['regrasp_recovery']['site_lookahead_seconds']
                    try:
                        onward=self.search_grasps(index,destination,{'force_search':True})
                        if not onward['grasps']:raise ValueError('staging site has no verified onward grasp')
                    finally:self.site_deadline=float('inf')
                    return {'waypoints':waypoints,'release':release.tolist(),'above':above.tolist(),
                            'orientation':q.tolist(),'support_shape':site['structure'],'support_height':site['height'],
                            'lower_joint_path':lower,'retreat_joint_path':retreat,'regrasp':onward,'failures':failures}
                except (RuntimeError,ValueError) as exc:failures.append(str(exc))
                finally:
                    bridge.env._robot.gripper.release();bridge.restore_state(state);self.attach_copy(index)
        raise RuntimeError('no checked regrasp staging site: '+repr(failures[-3:]))

    def placement(self, index, destination):
        from pyrep.objects.shape import Shape
        bridge = self.bridge
        obj = Shape(bridge.entities[index]['primary_name'])
        body = bridge.geometry(index)
        target, approach = bridge.placement_route(destination, body)
        inward = target-approach
        if abs(inward[2]) >= .3*np.linalg.norm(inward):
            return None
        tip_initial = matrix(bridge.tip_pose())
        object_initial = np.asarray(obj.get_matrix())
        relative = np.linalg.inv(tip_initial) @ object_initial
        original_state = bridge.snapshot_state()
        dimensions = body['dimensions']
        flat=face_stability_required(body,self.support_axis)
        round_support=body.get('symmetry_axis') is not None and body.get('symmetry_axis')==self.support_axis
        failures = []
        rotations=list(stable_orientations(object_initial, dimensions, inward,
                self.support_axis))
        if flat or round_support:
            rotations=aligned_shelf_rotations(rotations,relative,inward)
            direction=inward/np.linalg.norm(inward)
            rotations.sort(key=lambda item:(item[0],-float(np.dot((item[1]@relative[:3,:3].T)[:,2],direction))))
        if not flat and not round_support:
            entries=[(0.,Rotation.from_quat(q).as_matrix()@relative[:3,:3]) for q in bridge.entry_orientations(inward)]
            entries.sort(key=lambda item:float(np.abs(item[1][2,:])@dimensions))
            rotations=entries+rotations
        choices=[(tilt,rotation,{'front_margin':margin,'release_offset':offset},mode)
                 for tilt,rotation in rotations
                 for margin in (PLANNING_PROFILE['thin_front_margins'] if flat else [.04])
                 for offset in (PLANNING_PROFILE['thin_release_offsets'] if flat else PLANNING_PROFILE['ordinary_release_offsets'])
                 for mode in PLANNING_PROFILE['reorientation_modes']]
        rejected_prefixes=set()
        for tilt, object_rotation, options, mode in choices:
            try:self.check_budget()
            except RuntimeError as exc:
                raise RuntimeError(str(exc)+'; recent placement failures: '+repr(failures[-3:])) from exc
            prefix=(tilt,object_rotation.tobytes(),mode)
            if prefix in rejected_prefixes:continue
            if (flat or round_support) and tilt > PLANNING_PROFILE['stable_tilt_degrees']:
                continue  # Do not accept an unsupported edge-resting can.
            bridge.restore_state(original_state)
            bridge.placement_options=options
            orientation_matrix = object_rotation @ relative[:3, :3].T
            orientation = Rotation.from_matrix(orientation_matrix).as_quat()
            # The route is expressed for the payload CENTER, not the hand tip.
            # Stage at the candidate's actual shelf height and measured grasp
            # offset; an uncorrected center waypoint can put the wrist inside
            # the opening or below the shelf before reorientation.
            import itertools
            corners=np.array(list(itertools.product(*zip(body['low'],body['high']))))
            rotated=(corners-body['center'])@object_initial[:3,:3]@object_rotation.T
            hypothetical=dict(body,low=body['center']+rotated.min(axis=0),
                              high=body['center']+rotated.max(axis=0))
            try:
                _,candidate_approach=bridge.placement_route(destination,hypothetical,require_fit=True)
            except ValueError as exc:
                failures.append('candidate envelope: '+str(exc));continue
            center_in_hand=np.linalg.inv(tip_initial)@np.r_[body['center'],1.]
            anchor=np.asarray(candidate_approach)-orientation_matrix@center_in_hand[:3]
            anchor[2]=max(anchor[2],bridge.tip_pose()[2])
            stage='anchor'
            try:
                if mode=='rotate_before_transport':
                    waypoints=[{'position':bridge.tip_pose()[:3].tolist(),'orientation':orientation.tolist()},
                               {'position':anchor.tolist(),'orientation':orientation.tolist()}]
                else:
                    waypoints=[{'position':anchor.tolist(),'orientation':bridge.tip_pose()[3:].tolist()},
                               {'position':anchor.tolist(),'orientation':orientation.tolist()}]
                for waypoint in waypoints:
                    stage='wrist_orientation' if np.linalg.norm(np.asarray(waypoint['position'])-bridge.tip_pose()[:3])<.025 else 'anchor'
                    waypoint['joint_path']=self.path(waypoint['position'],waypoint['orientation'])
                held = bridge.geometry(index)
                target, approach = bridge.placement_route(destination, held, require_fit=True)
                offset = bridge.tip_pose()[:3]-held['center']
                above = approach+offset; release = target+offset
                stage='preplace'
                preplace=self.path(above, orientation)
                stage='insertion'
                insertion=self.path(release, orientation)
                stage='release_clearance'
                self.check_release(index)
                self.check_settling_clearance(index,bridge.support_height(destination,bridge.geometry(index)['center']))
                # Check retreat with the payload left physically at release.
                bridge.env._robot.gripper.release()
                stage='retreat'
                retreat=self.path(above, orientation)
                return {'orientation':orientation.tolist(), 'tilt_degrees':tilt,
                        'approach':above.tolist(), 'release':release.tolist(),
                        'placement_options':options,
                        'reorientation_waypoints':waypoints,
                        'preplace_joint_path':preplace,'insertion_joint_path':insertion,'retreat_joint_path':retreat,
                        'failures':failures}
            except (RuntimeError, ValueError) as exc:
                failures.append(stage+': '+str(exc))
                if stage in ('anchor','wrist_orientation'):rejected_prefixes.add(prefix)
            finally:
                # Reattach at original transform for the next candidate.
                bridge.env._robot.gripper.release()
                bridge.restore_state(original_state)
                self.attach_copy(index)
        raise RuntimeError('no stable shelf placement: '+str(failures[-4:]))

    def run(self, request):
        self.sync(request)
        self.current_operation=request['operation']
        self.proposed_joint_path=None
        budget=PLANNING_PROFILE['search_budget_seconds']
        if request['operation']=='grasp' and request['destination'] is not None:
            budget=min(budget,PLANNING_PROFILE['regrasp_recovery']['direct_search_seconds'])
        self.deadline=time.monotonic()+budget
        index = request['object']; destination = request['destination']
        if request['operation']=='motion':
            if request.get('attached'):self.attach_copy(index)
            self.proposed_joint_path=request.get('joint_path');self.target_tolerance=request.get('target_tolerance',.008)
            return {'joint_path':self.path(request['position'],request['orientation'])}
        if request['operation']=='verify_grasp':
            if not request.get('attached'):raise ValueError('actual grasp is not attached')
            self.attach_copy(index)
            self.path(request['lift_position'],request['orientation'])
            return {'placement':self.placement(index,destination) if destination is not None else None}
        if request['operation']=='staging':
            if not request.get('attached'):raise ValueError('regrasp staging requires actual attachment')
            self.attach_copy(index)
            if destination is None:raise ValueError('staging requires predicted destination lookahead')
            return self.staging(index,destination)
        if request['operation'] in ('place','verify_placement','verify_release'):
            if not request.get('attached'):raise ValueError('placement requires actual attachment')
            self.attach_copy(index)
            if request['operation']=='verify_placement':return self.verify_placement(index,destination)
            if request['operation']=='verify_release':return self.verify_release(index,destination,request.get('retreat_position'),actual_open=request.get('actual_open',False))
            return {'placement':self.placement(index, destination)}
        return self.search_grasps(index,destination,request)

    def search_grasps(self,index,destination,request):
        body = self.bridge.geometry(index)
        opening=self.bridge.finger_opening()
        if body['horizontal_span'] < opening*PLANNING_PROFILE['wide_opening_fraction'] and body.get('symmetry_axis') is None and not request.get('force_search'):
            return {'grasps':[], 'requires_contact_search':False,'finger_opening':opening}
        state = self.bridge.snapshot_state()
        from pyrep.objects.shape import Shape
        source_body_pose=np.asarray(Shape(self.bridge.entities[index]['primary_name']).get_pose()).tolist()
        candidates = []; failures = []
        for candidate_id,(point,orientation,yaw) in enumerate(self.grasp_proposals(body)):
            if candidate_id in request.get('excluded_candidates',[]):continue
            self.check_budget()
            self.bridge.env._robot.gripper.release(); self.bridge.restore_state(state)
            if not self.contact_feasible(index,point,orientation):
                failures.append('relative contact: native detector or open-finger clearance rejected')
                continue
            above = point+[0.,0.,.14]
            stage='pregrasp'
            try:
                try:self.path(above,orientation)
                except (RuntimeError,ValueError):
                    self.bridge.restore_state(state)
                    safer=self.bridge.grasp_standoff(index,orientation,above)
                    if safer[2]<=above[2]+.001:raise
                    self.path(safer,orientation);above=safer
                stage='contact_approach'
                self.path(point, orientation)
                # Actual open-finger geometry and actual native sensor,
                # queried in this copy; live checks remain mandatory.
                if not self.bridge.detect_grasp(index):
                    raise ValueError('predicted body outside native detector')
                if not self.bridge.release_clearance(index):
                    raise ValueError('open fingers intersect predicted body')
                self.attach_copy(index)
                measured=self.bridge.tip_pose()
                point=measured[:3].copy();orientation=measured[3:].copy()
                above=point+[0.,0.,max(.14,above[2]-point[2])]
                stage='lift'
                self.path(above, orientation)
                stage='destination'
                self.candidate_deadline=time.monotonic()+PLANNING_PROFILE['destination_budget_per_grasp_seconds']
                try:placement = self.placement(index, destination) if destination is not None else None
                finally:self.candidate_deadline=float('inf')
                candidates.append({'point':point.tolist(),'orientation':orientation.tolist(),
                                   'yaw':float(yaw),'placement':placement,'candidate_id':candidate_id,'above':above.tolist(),
                                   'body_pose':source_body_pose})
                print('[shadow-grasp] validated candidate',file=sys.stderr,flush=True)
                if len(candidates) >= PLANNING_PROFILE['validated_grasps_per_request']:
                    return {'grasps':candidates, 'failures':failures,'requires_contact_search':True,'finger_opening':opening}
            except (RuntimeError, ValueError) as exc:
                failures.append(stage+': '+str(exc))
                print('[shadow-grasp] candidate=%d rejected=%s: %s' % (len(failures),stage,exc),file=sys.stderr,flush=True)
        return {'grasps':candidates, 'failures':failures,'requires_contact_search':True,'finger_opening':opening}


def worker(fd):
    import faulthandler
    import os
    import time
    import select
    faulthandler.enable()
    faulthandler.dump_traceback_later(60,file=sys.stderr,repeat=True)
    environment=dict(os.environ)
    environment.pop('LD_LIBRARY_PATH',None);environment.pop('QT_QPA_PLATFORM_PLUGIN_PATH',None)
    reader,writer=os.pipe()
    try:
        server=subprocess.Popen(['Xvfb','-displayfd',str(writer),'-ac','-screen','0','1280x1024x24'],
                                env=environment,pass_fds=(writer,))
        os.close(writer);writer=None
        if not select.select([reader],[],[],10)[0]:raise RuntimeError('planning X server startup timeout')
        number=os.read(reader,128).decode().strip()
        if not number.isdigit():raise RuntimeError('planning X server failed')
        display=':'+number
    finally:
        os.close(reader)
        if writer is not None:os.close(writer)
    os.environ['DISPLAY']=display
    print('[shadow] launching native scene',file=sys.stderr,flush=True)
    from rlbench_bridge import RLBenchBridge
    connection = socket.socket(fileno=fd)
    stream = connection.makefile('rwb')
    bridge = RLBenchBridge(render=False)
    faulthandler.cancel_dump_traceback_later()
    print('[shadow] native scene ready',file=sys.stderr,flush=True)
    try:
        planner = ShadowPlanner(bridge)
        for line in stream:
            import faulthandler
            faulthandler.dump_traceback_later(120,file=sys.stderr)
            try:
                print('[shadow] request received',file=sys.stderr,flush=True)
                result = planner.run(json.loads(line))
            except Exception as exc:
                result = {'error':type(exc).__name__+': '+str(exc)}
            finally:
                faulthandler.cancel_dump_traceback_later()
            stream.write((json.dumps(result)+'\n').encode()); stream.flush()
    finally:
        bridge.close(); stream.close(); connection.close()
        server.terminate();server.wait(timeout=5)


if __name__ == '__main__':
    worker(int(sys.argv[2]))
