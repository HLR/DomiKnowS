"""Goal-blind, bounded Panda manipulation through the RLBench bridge.

No task, variation, success sensor, demonstration or target reference is accepted.
"""
import numpy as np
try:from .rlbench_planning import PLANNING_PROFILE, release_stable, thin_body, matrix
except ImportError:from rlbench_planning import PLANNING_PROFILE, release_stable, thin_body, matrix


class ContactMotionStop(RuntimeError):
    """The same predicted body moved; update the contact frame before continuing."""


def downward_quaternion(yaw):
    # xyzw: Rz(yaw) @ Ry(pi); rotations are world-frame, gripper opening is 1.
    return np.array([-np.sin(yaw/2), np.cos(yaw/2), 0., 0.])


def rotation_step(current,target,max_angle=.1):
    current=np.asarray(current,dtype=float);target=np.asarray(target,dtype=float)
    if current.shape!=(4,) or target.shape!=(4,) or not np.isfinite(np.r_[current,target]).all():
        raise ValueError('invalid xyzw quaternion')
    if min(np.linalg.norm(current),np.linalg.norm(target))<1e-9:raise ValueError('zero quaternion')
    current=current/np.linalg.norm(current);target=target/np.linalg.norm(target)
    dot=np.dot(current,target)
    if dot<0:target=-target;dot=-dot
    angle=2*np.arccos(np.clip(dot,-1,1));fraction=min(1.,max_angle/max(angle,1e-9))
    result=(1-fraction)*current+fraction*target
    return result/np.linalg.norm(result)


def side_entry_quaternion(inward):
    from scipy.spatial.transform import Rotation
    z=np.asarray(inward,dtype=float);z=z/np.linalg.norm(z)
    y=np.array([0.,0.,1.]);x=np.cross(y,z);x=x/np.linalg.norm(x);y=np.cross(z,x)
    return Rotation.from_matrix(np.column_stack((x,y,z))).as_quat()


def tilted_downward_quaternion(yaw,radial,angle=np.pi/12):
    from scipy.spatial.transform import Rotation
    direction=np.r_[np.asarray(radial)[:2],0.]
    if np.linalg.norm(direction)<1e-9:return downward_quaternion(yaw)
    axis=np.cross([0.,0.,-1.],direction/np.linalg.norm(direction))
    return (Rotation.from_rotvec(axis*angle)*Rotation.from_quat(downward_quaternion(yaw))).as_quat()


def lateral_grasp_quaternion(radial,roll=0.):
    """Approach toward the body with a horizontal finger-closing axis."""
    from scipy.spatial.transform import Rotation
    z=np.r_[np.asarray(radial,dtype=float)[:2],0.]
    if np.linalg.norm(z)<1e-9:raise ValueError('undefined lateral approach direction')
    z/=np.linalg.norm(z);y=np.cross([0.,0.,1.],z);x=np.cross(y,z)
    return Rotation.from_matrix(np.column_stack((x,y,z))@Rotation.from_rotvec([0.,0.,roll]).as_matrix()).as_quat()


class RLBenchExecutor:
    def __init__(self, bridge, *, horizon=1000):
        self.bridge = bridge
        self.horizon = horizon
        self.steps = 0
        self.held = None
        self.events = []
        self.grasp_yaw = 0.
        self.phase='start'
        self.last_command=None
        self.regrasp_used=False

    def tick(self):
        if self.steps >= self.horizon:raise TimeoutError('executor horizon exhausted')
        self.bridge.tick()
        self.steps += 1

    def settle(self, count):
        for _ in range(count):self.tick()

    def gripper(self, opened):
        for _ in range(80):
            done = self.bridge.actuate_gripper(1. if opened else 0.)
            self.tick()
            if done:return
        raise TimeoutError('gripper actuation timed out')

    def open_fingers_held(self):
        try:
            for _ in range(80):
                done=self.bridge.actuate_fingers(1.,velocity=PLANNING_PROFILE['finger_contact_control']['open_velocity'])
                self.tick()
                if done:return
            raise TimeoutError('finger opening timed out before native release')
        finally:
            if hasattr(self.bridge,'hold_fingers'):self.bridge.hold_fingers()

    def close_contact(self,index):
        if not hasattr(self.bridge,'hold_fingers') or not hasattr(self.bridge,'finger_clearance'):
            return self.gripper(False)
        profile=PLANNING_PROFILE['finger_contact_control']
        try:
            for _ in range(profile['maximum_ticks']):
                done=self.bridge.actuate_fingers(0.,velocity=profile['close_velocity'])
                self.tick()
                distance=self.bridge.finger_clearance(index)
                if done or distance<=profile['stop_distance']:
                    self.events.append({'skill':'finger_contact_stop','entity_id':index,'distance':distance})
                    return
            raise TimeoutError('contact-aware finger closing timed out')
        finally:self.bridge.hold_fingers()

    def move(self, position, yaw=0., limit=120, tolerance=.008, orientation=None,contact=False,stall_after=None,joint_path=None):
        target = np.asarray(position, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():raise ValueError('invalid world target')
        quaternion = downward_quaternion(yaw) if orientation is None else np.asarray(orientation)
        self.last_command={'phase':self.phase,'position':target.tolist(),'quaternion_xyzw':quaternion.tolist()}
        if hasattr(self.bridge,'path_to'):
            # Exempt only intended contact with the predicted/held body during
            # path planning; simulator collision response is never disabled.
            if hasattr(self.bridge,'checked_motion'):
                options={'joint_path':joint_path,'tolerance':tolerance} if joint_path is not None else {}
                path=self.bridge.checked_motion(target,quaternion,withdraw=self.phase=='retreat',**options)
            else:path=self.bridge.path_to(target,quaternion,contact=contact or self.held is not None)
            done=False;best_error=float('inf');stagnant=0
            monitor=(self.phase in ('grasp_approach','contact_seek') and hasattr(self.bridge,'hold_arm')
                     and hasattr(self.bridge,'detect_grasp') and getattr(self.bridge,'contact_entity',None) is not None)
            contact_origin=self.bridge.geometry(self.bridge.contact_entity)['center'].copy() if monitor else None
            for _ in range(max(limit,400)):
                if not done:done=path.step()
                if self.phase=='retreat' and hasattr(self.bridge,'actuate_gripper'):
                    self.bridge.actuate_gripper(1.)
                self.tick()
                if monitor and np.linalg.norm(self.bridge.geometry(self.bridge.contact_entity)['center']-contact_origin)>PLANNING_PROFILE['contact_object_motion_threshold']:
                    self.bridge.hold_arm(path)
                    detected=(self.bridge.detect_grasp(self.bridge.contact_entity) and
                              self.bridge.release_clearance(self.bridge.contact_entity))
                    self.events.append({'skill':'contact_motion_stop','entity_id':self.bridge.contact_entity,'native_detected':detected})
                    if detected:return
                    raise ContactMotionStop('predicted object moved during approach; replan from measured pose')
                pose=self.bridge.tip_pose()
                if done and np.linalg.norm(pose[:3]-target)<tolerance and abs(np.dot(pose[3:],quaternion))>.995:
                    self.settle(3);return
                if done and stall_after is not None:
                    error=np.linalg.norm(pose[:3]-target)+.1*2*np.arccos(np.clip(abs(np.dot(pose[3:],quaternion)),0.,1.))
                    if error<best_error-.001:best_error=error;stagnant=0
                    else:stagnant+=1
                    if stagnant>=stall_after:
                        if hasattr(self.bridge,'hold_arm'):self.bridge.hold_arm(path)
                        raise TimeoutError('Panda completed trajectory stalled during tracking')
            if hasattr(self.bridge,'hold_arm'):self.bridge.hold_arm(path)
            raise TimeoutError('Panda path tracking did not converge: trajectory_done='+str(done)
                               +' actual_pose='+str(self.bridge.tip_pose().tolist()))
        for _ in range(limit):
            current = np.asarray(self.bridge.tip_pose()[:3])
            delta = target-current
            waypoint = current + delta*min(1., .015/max(np.linalg.norm(delta), 1e-9))
            self.bridge.set_tip_target(waypoint, rotation_step(self.bridge.tip_pose()[3:],quaternion))
            self.tick()
            pose = self.bridge.tip_pose()
            if np.linalg.norm(pose[:3]-target) < tolerance and abs(np.dot(pose[3:],quaternion)) > .995:
                self.settle(3)
                return
        raise TimeoutError('IK waypoint did not converge: target='+str(target.tolist())+
                           ' actual='+str(self.bridge.tip_pose().tolist()))

    def reorient(self,orientation):
        # Keep IK on a continuous local wrist branch instead of choosing one
        # distant joint configuration for a large in-place orientation change.
        anchor=self.bridge.tip_pose()[:3].copy()
        try:
            self.move(anchor,orientation=orientation,stall_after=20,tolerance=.025)
            return
        except (TimeoutError,RuntimeError) as exc:
            if self.steps>=self.horizon:raise
            self.events.append({'skill':'reorientation_recovery','failure':str(exc)})
        anchor=self.bridge.tip_pose()[:3].copy()
        for _ in range(40):
            current=self.bridge.tip_pose()[3:]
            if 2*np.arccos(np.clip(abs(np.dot(current,orientation)),0.,1.))<.04:return
            # Small corrective poses can settle slowly under contact. Retain
            # their bounded tracking allowance; the global horizon still applies.
            self.move(anchor,orientation=rotation_step(current,orientation,.2),stall_after=None,tolerance=.025)
        raise RuntimeError('incremental wrist rotation did not converge')

    def approach_contact(self,index,point,orientation,yaw,*,tolerance,stall_after,body_pose=None):
        if not hasattr(self.bridge,'release_sample'):
            self.move(point,yaw,tolerance=tolerance,contact=True,orientation=orientation,stall_after=stall_after)
            return point,orientation
        from scipy.spatial.transform import Rotation
        body=matrix(self.bridge.release_sample(index)['pose'])
        source=body if body_pose is None else matrix(body_pose)
        relative=np.linalg.inv(source)@matrix(np.r_[point,orientation])
        target=body@relative;point=target[:3,3].copy();orientation=Rotation.from_matrix(target[:3,:3]).as_quat()
        for refinement in range(PLANNING_PROFILE['contact_frame_replans']+1):
            try:
                self.move(point,yaw,tolerance=tolerance,contact=True,orientation=orientation,stall_after=stall_after)
                return point,orientation
            except ContactMotionStop:
                if refinement>=PLANNING_PROFILE['contact_frame_replans']:
                    raise RuntimeError('contact-frame replanning budget exhausted')
                body=matrix(self.bridge.release_sample(index)['pose']);target=body@relative
                point=target[:3,3].copy();orientation=Rotation.from_matrix(target[:3,:3]).as_quat()
                self.events.append({'skill':'contact_frame_replan','entity_id':index,'refinement':refinement+1,
                                    'position':point.tolist(),'orientation':orientation.tolist()})

    def regrasp(self,index,destination,failure):
        if (self.regrasp_used or destination is None or
                self.horizon-self.steps<PLANNING_PROFILE['regrasp_recovery']['minimum_remaining_steps']):
            raise RuntimeError(failure)
        self.regrasp_used=True
        self.events.append({'skill':'regrasp_recovery','entity_id':index,'failure':failure})
        self.pick(index,None,allow_regrasp=False)
        route=self.bridge.planning_request('staging',index,destination)
        self.events.append({'skill':'regrasp_staging_search','entity_id':index,'route':route})
        for waypoint in route['waypoints']:
            self.phase='regrasp_staging';self.move(waypoint['position'],orientation=waypoint['orientation'],tolerance=.025,stall_after=20,
                                                 joint_path=waypoint.get('joint_path'))
        self.phase='regrasp_lower'
        self.move(route['release'],orientation=route['orientation'],tolerance=PLANNING_PROFILE['landing_tolerance'],stall_after=20,
                  joint_path=route.get('lower_joint_path'))
        support=self.bridge.physical_support_height(route['support_shape'],self.bridge.geometry(index)['center'])
        if abs(support-route['support_height'])>.002:raise RuntimeError('regrasp support geometry changed')
        self.gripper(True);self.held=None
        if self.bridge.release_sample(index)['attached']:raise RuntimeError('regrasp native attachment remained')
        self.settle(PLANNING_PROFILE['release_dwell_steps'])
        self.phase='retreat';self.move(route['above'],orientation=route['orientation'],stall_after=20,
                                       joint_path=route.get('retreat_joint_path'))
        self.gripper(True)
        if not self.bridge.release_clearance(index):raise RuntimeError('regrasp finger separation failed')
        self.phase='regrasp_verify_support';samples=[]
        for _ in range(PLANNING_PROFILE['settle_limit']):
            self.tick();samples.append(self.bridge.release_sample(index))
            if len(samples)>=30 and release_stable(samples,support=support,require_face=self.bridge.requires_stable_face(index)):break
        else:raise RuntimeError('regrasp staging failed physical support/stability verification')
        self.events.append({'skill':'regrasp_staging','entity_id':index,'samples':samples,'support':support,'success':True})
        options={'prepared_search':route['regrasp']} if route.get('regrasp') is not None else {}
        self.pick(index,destination,allow_regrasp=False,**options)

    def pick(self, index, destination_index=None, *, allow_regrasp=True,prepared_search=None):
        geometry = self.bridge.geometry(index)
        if not geometry['movable']:raise ValueError('cannot pick a static fixture')
        initial_z = geometry['center'][2]
        if hasattr(self.bridge,'set_contact_entity'):self.bridge.set_contact_entity(index)
        proposals=[]
        if hasattr(self.bridge,'planning_request'):
            try:search=prepared_search if prepared_search is not None else self.bridge.planning_request('grasp',index,destination_index)
            except RuntimeError as exc:
                if allow_regrasp and destination_index is not None and str(exc).startswith('planning worker:'):
                    return self.regrasp(index,destination_index,str(exc))
                raise
            proposals=search['grasps']
            self.events.append({'skill':'grasp_search','entity_id':index,'search':search})
            if search['requires_contact_search'] and not proposals:
                failure='no contact-aware grasp with stable destination placement'
                if allow_regrasp and destination_index is not None:return self.regrasp(index,destination_index,failure)
                raise RuntimeError(failure)
        # Fixed same-body wrist/height alternatives, independent of task goals.
        alternatives=((geometry['yaw'],0.,0.),
                (geometry['yaw'],0.,np.pi/6),(geometry['yaw']+np.pi/2,0.,np.pi/6),
                (geometry['yaw'],-.01,np.pi/12),
                # Finger symmetry does not imply arm/wrist reachability symmetry.
                # Keep equivalent closing axes with the opposite wrist heading.
                (geometry['yaw']+np.pi,0.,0.),(geometry['yaw']+np.pi,0.,np.pi/6),
                (0.,0.,None),(np.pi,0.,None))
        choices=[(p['yaw'],0.,0.,p) for p in proposals] if proposals else [(y,h,t,None) for y,h,t in alternatives]
        excluded=[]
        for attempt,(yaw, height,tilt,proposal) in enumerate(choices):
            try:
                box = self.bridge.geometry(index)
                point = np.asarray(box['center']) + [0., 0., height]
                # A body wider than Panda's finger opening uses native proximal
                # top contact rather than an impossible straddling descent.
                # A wide body must stay below the open fingers so that native
                # release actually frees it. Deep straddling can keep a flat
                # body physically wedged even after native detachment.
                top_offset=.002 if box.get('horizontal_span',0.)>.075 else -.025
                point[2]=max(point[2],box['high'][2]+top_offset+height)
                radial=point-self.bridge.base_position()
                if tilt is None:
                    orientation=lateral_grasp_quaternion(radial,roll=yaw)
                    # Keep the physical hand above the observed supporting
                    # surface, rather than demanding an impossible table grasp.
                    clearance=self.bridge.grasp_clearance(orientation)
                    point[2]=max(point[2],box['low'][2]+clearance)
                else:
                    orientation=tilted_downward_quaternion(yaw,radial,angle=tilt) if tilt else downward_quaternion(yaw)
                if proposal is not None:
                    point=np.asarray(proposal['point']);orientation=np.asarray(proposal['orientation'])
                above = point + [0., 0., .14]
                if proposal is not None and 'above' in proposal:above=np.asarray(proposal['above'])
                self.phase='pregrasp'
                try:self.move(above, yaw,orientation=orientation)
                except RuntimeError:
                    if not hasattr(self.bridge,'grasp_standoff'):raise
                    safer=self.bridge.grasp_standoff(index,orientation,above)
                    if safer[2]<=above[2]+.001:raise
                    self.events.append({'skill':'pregrasp_standoff_recovery','entity_id':index,
                                        'original':above.tolist(),'raised':safer.tolist()})
                    self.move(safer,yaw,orientation=orientation)
                self.gripper(True)
                self.phase='grasp_approach'
                try:point,orientation=self.approach_contact(index,point,orientation,yaw,
                              tolerance=PLANNING_PROFILE['planned_grasp_tolerance'] if proposal else .018,
                              stall_after=PLANNING_PROFILE['planned_contact_stall_ticks'] if proposal else None,
                              body_pose=proposal.get('body_pose') if proposal else None)
                except TimeoutError:
                    # Physical contact may prevent exact Cartesian convergence.
                    # Continue only if the native proximity gate detects this
                    # same predicted body; never accept a nearby substitute.
                    detector=getattr(self.bridge,'detect_grasp',self.bridge.try_grasp)
                    if self.steps>=self.horizon or not detector(index):raise
                detector=getattr(self.bridge,'detect_grasp',self.bridge.try_grasp)
                detected=detector(index)
                if not detected and hasattr(self.bridge,'detect_grasp'):
                    # Bounded contact search, never an unconditional lift.
                    origin=self.bridge.tip_pose()[:3].copy()
                    from scipy.spatial.transform import Rotation
                    direction=Rotation.from_quat(orientation).as_matrix()[:,2]
                    for distance in np.arange(PLANNING_PROFILE['seek_step'],PLANNING_PROFILE['seek_limit']+1e-6,PLANNING_PROFILE['seek_step']):
                        candidate=origin+direction*distance
                        if candidate[2]<box['low'][2]+self.bridge.grasp_clearance(orientation):break
                        self.phase='contact_seek'
                        self.move(candidate,orientation=orientation,contact=True,stall_after=10)
                        if detector(index):detected=True;break
                if not detected:raise RuntimeError('native detector rejected predicted object before lift')
                if hasattr(self.bridge,'checked_motion') and hasattr(self.bridge,'release_clearance') and not self.bridge.release_clearance(index):
                    raise RuntimeError('actual open fingers intersect predicted body before attachment')
                if not self.bridge.try_grasp(index):raise RuntimeError('native attachment rejected predicted object')
                self.close_contact(index)
                if hasattr(self.bridge,'detect_grasp') and not self.bridge.detect_grasp(index):
                    raise RuntimeError('native detector lost predicted body during finger closure')
                if hasattr(self.bridge,'checked_motion'):
                    measured=self.bridge.tip_pose()
                    orientation=measured[3:].copy()
                    above=measured[:3]+[0.,0.,max(.14,above[2]-measured[2])]
                if destination_index is not None and hasattr(self.bridge,'checked_motion'):
                    self.phase='verify_actual_grasp'
                    checked=self.bridge.planning_request('verify_grasp',index,destination_index,
                                                        lift_position=above.tolist(),orientation=np.asarray(orientation).tolist())
                    self.events.append({'skill':'actual_grasp_recheck','entity_id':index,'search':checked})
                self.phase='lift'
                self.move(above, yaw,contact=True,orientation=orientation)
                self.settle(10)
                if detected and self.bridge.geometry(index)['center'][2] > initial_z+.05:
                    self.held = index
                    self.grasp_yaw = yaw
                    self.events.append({'skill':'pick','entity_id':index,'success':True})
                    return
            except (RuntimeError,ValueError,TimeoutError) as exc:
                import sys
                print('[executor-grasp] entity=%d attempt=%d phase=%s rejected=%s' % (index,attempt,self.phase,exc),file=sys.stderr,flush=True)
                self.events.append({'skill':'pick_attempt','entity_id':index,'attempt':attempt,'failure':str(exc)})
                if self.steps>=self.horizon:raise
            self.gripper(True)
            # A failed contact approach may leave the hand touching the body.
            # Retreat before requesting another collision-checked approach.
            if self.bridge.tip_pose()[2]<self.bridge.geometry(index)['high'][2]+.06:
                recovery=self.bridge.tip_pose()[:3].copy();recovery[2]+=.14
                try:self.move(recovery,orientation=self.bridge.tip_pose()[3:],contact=True)
                except (RuntimeError,ValueError,TimeoutError):
                    if self.steps>=self.horizon:raise
            # A physical pinch can lift the body even when the native detector
            # did not attach it. After releasing, let it settle before recomputing
            # the next approach; otherwise retries target its transient air pose.
            self.settle(20)
            if proposal is not None and hasattr(self.bridge,'planning_request') and attempt+1<PLANNING_PROFILE['grasp_attempts']:
                excluded.append(proposal['candidate_id'])
                try:search=self.bridge.planning_request('grasp',index,destination_index,excluded_candidates=excluded,force_search=True)
                except RuntimeError as exc:
                    if allow_regrasp and destination_index is not None and str(exc).startswith('planning worker:'):
                        return self.regrasp(index,destination_index,str(exc))
                    raise
                self.events.append({'skill':'grasp_retry_search','entity_id':index,'search':search})
                if search['grasps']:
                    p=search['grasps'][0];choices.append((p['yaw'],0.,0.,p))
        raise RuntimeError('grasp failed to lift the predicted object')

    def place(self, index, *, allow_recovery=True):
        if self.held is None or self.held == index:raise ValueError('place requires a different held object')
        held = self.bridge.geometry(self.held)
        target,approach = self.bridge.placement_route(index, held)
        inward=np.asarray(target)-np.asarray(approach)
        orientation=downward_quaternion(self.grasp_yaw)
        side_entry=abs(inward[2]) < .3*np.linalg.norm(inward)
        placement_search=None
        verified={}
        release_prepared=False
        if hasattr(self.bridge,'planning_request'):
            search=self.bridge.planning_request('place',self.held,index)
            self.events.append({'skill':'placement_search','entity_id':index,'search':search})
            if search['placement'] is not None:
                placement_search=search['placement'];orientation=np.asarray(placement_search['orientation'])
        if side_entry:
            # Enter a side-opening receptacle with the palm outside. The object
            # may rotate to fit; orientation follows observed opening geometry.
            if not hasattr(self.bridge,'planning_request'):
                orientation=self.bridge.entry_orientation(inward) if hasattr(self.bridge,'entry_orientation') else side_entry_quaternion(inward)
            # Reorient in the clear area before the opening, rather than at an
            # arbitrary pick location near a wrist singularity or workspace edge.
            if placement_search is not None:
                for waypoint in placement_search['reorientation_waypoints']:
                    position=np.asarray(waypoint['position']);quaternion=np.asarray(waypoint['orientation'])
                    if waypoint.get('joint_path') is not None:
                        self.phase='checked_reorientation';self.move(position,orientation=quaternion,tolerance=.025,
                                                                    joint_path=waypoint['joint_path'],stall_after=20)
                    elif np.linalg.norm(position-self.bridge.tip_pose()[:3])<.025:
                        self.phase='reorient_for_opening';self.reorient(quaternion)
                    else:
                        self.phase='reorientation_anchor'
                        self.move(position,orientation=quaternion,tolerance=.025)
            else:
                anchor=np.asarray(approach).copy()
                anchor[2]=max(self.bridge.tip_pose()[2],self.bridge.geometry(index)['low'][2]-.10)
                self.phase='reorientation_anchor'
                self.move(anchor,self.grasp_yaw,tolerance=.025)
                self.phase='reorient_for_opening'
                self.reorient(orientation)
            # Fit and command the same realized orientation. The accepted native
            # pose need not equal the nominal IK target exactly.
            orientation=self.bridge.tip_pose()[3:].copy()
            held=self.bridge.geometry(self.held)
            target,approach=self.bridge.placement_route(index,held,require_fit=True)
        offset=self.bridge.tip_pose()[:3]-np.asarray(held['center'])
        release=np.asarray(target)+offset;above=np.asarray(approach)+offset
        self.phase='preplace'
        self.move(above,orientation=orientation,tolerance=.025,
                  joint_path=placement_search.get('preplace_joint_path') if placement_search else None)
        # Refresh after staging: physics/contact can change the held transform.
        # Final placement still uses the original strict Cartesian tolerance.
        orientation=self.bridge.tip_pose()[3:].copy()
        held=self.bridge.geometry(self.held)
        target,_=self.bridge.placement_route(index,held,require_fit=side_entry)
        release=np.asarray(target)+self.bridge.tip_pose()[:3]-np.asarray(held['center'])
        if side_entry and hasattr(self.bridge,'planning_request'):
            verified=self.bridge.planning_request('verify_placement',self.held,index)
            release=np.asarray(verified['release']);above=np.asarray(verified['approach'])
            orientation=np.asarray(verified['orientation'])
            self.events.append({'skill':'actual_placement_recheck','entity_id':index,'search':verified})
        self.phase='insert_or_lower'
        if hasattr(self.bridge,'checked_motion'):
            self.move(release,orientation=orientation,contact=True,tolerance=PLANNING_PROFILE['landing_tolerance'],stall_after=20,
                      joint_path=verified.get('insertion_joint_path') if side_entry else None)
        else:self.move(release,orientation=orientation,contact=True)
        if side_entry and hasattr(self.bridge,'checked_motion') and hasattr(self.bridge,'planning_request'):
            self.phase='verify_actual_release'
            self.settle(PLANNING_PROFILE['release_readiness']['hold_settle_steps'])
            opened=False
            try:
                ready=self.bridge.planning_request('verify_release',self.held,index,retreat_position=np.asarray(above).tolist())
                if hasattr(self.bridge,'actuate_fingers'):
                    self.phase='open_before_native_release';opened=True;self.open_fingers_held()
                    actual=self.bridge.planning_request('verify_release',self.held,index,actual_open=True)
                    self.events.append({'skill':'open_finger_release_readiness','entity_id':self.held,
                                        'sample':actual['sample'],'support':actual['support']})
                    self.bridge.release_native(self.held);release_prepared=True
            except RuntimeError as exc:
                if (not allow_recovery or not str(exc).startswith('planning worker:') or
                        self.horizon-self.steps<PLANNING_PROFILE['release_readiness']['recovery_reserve_steps']):raise
                self.events.append({'skill':'release_correction','entity_id':self.held,'destination_id':index,'failure':str(exc)})
                if opened:self.close_contact(self.held)
                self.phase='release_correction_outward'
                path=verified.get('insertion_joint_path')
                self.move(above,orientation=self.bridge.tip_pose()[3:],tolerance=.025,stall_after=20,
                          joint_path=list(reversed(path)) if path is not None else None)
                return self.place(index,allow_recovery=False)
            self.events.append({'skill':'release_readiness','entity_id':self.held,'sample':ready['sample'],'support':ready['support']})
            verified['retreat_joint_path']=ready['retreat_joint_path']
        released=self.held
        if not release_prepared:self.gripper(True)
        self.held=None
        # Attachment is gone; physical support must be established separately.
        support=None;samples=[]
        if hasattr(self.bridge,'release_sample'):
            support=self.bridge.support_height(index,self.bridge.geometry(released)['center'])
            self.phase='release_separation'
            if self.bridge.release_sample(released)['attached']:
                raise RuntimeError('native attachment remained after release')
            # Let gravity seat the detached body before the open hand withdraws.
            # Every physical tick remains charged to the same episode horizon.
            self.settle(PLANNING_PROFILE['release_dwell_steps'])
        self.phase='retreat'
        self.move(above,orientation=orientation,contact=True,
                  joint_path=verified.get('retreat_joint_path') if side_entry and hasattr(self.bridge,'planning_request') else None)
        if hasattr(self.bridge,'checked_motion'):self.gripper(True)
        if support is not None and not self.bridge.release_clearance(released):
            raise RuntimeError('released object remained in contact with fingers after withdrawal')
        if support is None:self.settle(30)
        else:
            self.phase='verify_support'
            require_face=side_entry and self.bridge.requires_stable_face(released)
            for _ in range(PLANNING_PROFILE['settle_limit']):
                self.tick();samples.append(self.bridge.release_sample(released))
                if len(samples)>=30 and release_stable(samples,support=support,require_face=require_face):break
            else:
                self.events.append({'skill':'release_verification','entity_id':released,'support':support,'samples':samples,'success':False})
                raise RuntimeError('released object failed physical support/stability verification')
            self.events.append({'skill':'release_verification','entity_id':released,'support':support,'samples':samples,'success':True})
        self.events.append({'skill':'place','entity_id':index,'success':True,'physical_release_verified':support is not None})

    def execute(self, plan):
        reason = 'completed'
        try:
            if not plan:raise ValueError('empty plan')
            for operation_number,operation in enumerate(plan):
                if not isinstance(operation,dict):raise ValueError('invalid operation')
                skill = operation.get('name')
                if skill not in ('pick','place'):raise ValueError('unsupported skill')
                role = 'target_entity_name' if skill == 'pick' else 'target_container_name'
                params = operation.get('params', {})
                if not isinstance(params,dict):raise ValueError('invalid entity pointer')
                index = params.get(role)
                if not isinstance(params,dict) or set(params) != {role} or type(index) is not int or not 0 <= index < len(self.bridge.entities):
                    raise ValueError('invalid entity pointer')
                if skill=='pick':
                    following=plan[operation_number+1] if operation_number+1<len(plan) else {}
                    next_params=following.get('params',{}) if isinstance(following,dict) else {}
                    destination=(next_params.get('target_container_name') if isinstance(next_params,dict) and isinstance(following,dict) and following.get('name')=='place' else None)
                    if destination is not None and (type(destination) is not int or not 0<=destination<len(self.bridge.entities)):
                        raise ValueError('invalid destination lookahead')
                    self.pick(index,destination_index=destination)
                else:self.place(index)
        except (RuntimeError, ValueError, TimeoutError) as exc:
            reason = str(exc)
        return {'steps':self.steps,'termination_reason':reason,'events':self.events,
                'phase':self.phase,
                'checked_motions':list(getattr(self.bridge,'checked_motion_records',())),
                'planning_contact_exemptions':list(getattr(self.bridge,'planning_contacts',())),
                'last_command':self.last_command,
                'holding_at_end':self.held,'grasp_protocol':'native_proximity_gated_attachment_predicted_entity_only'}
