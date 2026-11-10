"""Native RLBench observations/reset/physics bridge; simulator imports are lazy."""
import importlib
import itertools
import random
import numpy as np
try:from .rlbench_planning import PLANNING_PROFILE
except ImportError:from rlbench_planning import PLANNING_PROFILE

CONTROL_PROFILE={'arm_max_velocity':1.0,'arm_max_acceleration':4.0,
                 'tracking_stall_ticks':None,
                 'wide_grasp':'top_surface_plus_2mm',
                 'side_entry_release':'lowest_feasible_height_plus_1cm',
                 'staging_position_tolerance':.025,
                 'release_position_tolerance':.008,
                 'recovery_tracking_stall_ticks':None,
                 'pregrasp':'strict_then_geometry_standoff',
                 'reorientation':'direct_then_stall_recovery_0.2_rad_waypoints',
                 'contact_planning':'strict_then_local_movable_contact_with_structural_collision_checks',
                 'contact_neighbor_gap':.06,
                 'grasp':'native_proximity_gated_attachment_predicted_entity_only'}
CONTROL_PROFILE.update(geometry_search=PLANNING_PROFILE,
                       acquisition='native_detector_before_attachment_and_lift',
                       placement='shadow_joint_grasp_stable_face_and_wrist_search')


def transformed_bounds(box, matrix):
    points = np.array(list(itertools.product(*[(box[2*i],box[2*i+1]) for i in range(3)])))
    world = points @ np.asarray(matrix)[:3,:3].T + np.asarray(matrix)[:3,3]
    return np.array([world.min(axis=0),world.max(axis=0)])


def pose_error(expected,actual):
    expected=np.asarray(expected);actual=np.asarray(actual)
    return max(float(np.max(np.abs(expected[:3]-actual[:3]))),
               float(min(np.max(np.abs(expected[3:]-actual[3:])),np.max(np.abs(expected[3:]+actual[3:])))))


def ray_hits(vertices, triangles, origin, direction):
    """Positive Moller–Trumbore ray distances; geometry only, no task sensors."""
    tri = np.asarray(vertices).reshape(-1,3)[np.asarray(triangles,dtype=int).reshape(-1,3)]
    edge1,edge2 = tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
    h = np.cross(np.broadcast_to(direction,edge2.shape),edge2)
    determinant = np.einsum('ij,ij->i',edge1,h)
    good = np.abs(determinant)>1e-9
    inverse = np.zeros_like(determinant);inverse[good]=1/determinant[good]
    s = np.asarray(origin)-tri[:,0];u=inverse*np.einsum('ij,ij->i',s,h)
    q=np.cross(s,edge1);v=inverse*np.einsum('j,ij->i',direction,q)
    distance=inverse*np.einsum('ij,ij->i',edge2,q)
    return distance[good & (u>=0) & (v>=0) & (u+v<=1) & (distance>1e-5)]


class RLBenchBridge:
    def __init__(self, *, render=False):
        from rlbench.environment import Environment
        from rlbench.observation_config import ObservationConfig
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
        from rlbench.action_modes.gripper_action_modes import Discrete
        obs=ObservationConfig();obs.set_all(False)
        obs.front_camera.rgb=render;obs.front_camera.mask=render
        obs.front_camera.masks_as_one_channel=True;obs.front_camera.image_size=(384,384)
        self.env=Environment(MoveArmThenGripper(EndEffectorPoseViaIK(),Discrete(attach_grasped_objects=False)),
                             obs_config=obs,headless=True,attach_grasped_objects=False,
                             arm_max_velocity=CONTROL_PROFILE['arm_max_velocity'],
                             arm_max_acceleration=CONTROL_PROFILE['arm_max_acceleration'])
        self.env.launch()
        self.task=None;self.entities=[]
        self.contact_entity=None
        self.planning_contacts=[]

    def close(self):
        if getattr(self,'planner',None) is not None:self.planner.close()
        self.env.shutdown()

    def reset(self, task_name, variation, seed, *, state=None, entities=None):
        if task_name not in ('put_rubbish_in_bin','put_groceries_in_cupboard'):raise ValueError('unsupported RLBench task')
        random.seed(seed);np.random.seed(seed)
        self.mesh_cache={}
        self.symmetry_cache={}
        self.scene=(task_name,variation,seed)
        self.planning_contacts=[];self.contact_entity=None
        self.checked_motion_records=[]
        self.observed_support_axes={}
        self.placement_options={}
        cls=getattr(importlib.import_module('rlbench.tasks.'+task_name),''.join(w.title() for w in task_name.split('_')))
        if self.task is None or self.task.get_name()!=task_name:self.task=self.env.get_task(cls)
        self.task.set_variation(variation)
        descriptions,_=self.task.reset()
        if state is not None:self.restore_state(state)
        self.entities=self.inventory() if entities is None else entities
        if state is None:
            for _ in range(20):self.tick()
            self.entities=self.inventory()
        return descriptions[0]

    def tick(self):
        self.env._pyrep.step()
        self.task._task.step()

    def tip_pose(self):return np.asarray(self.env._robot.arm.get_tip().get_pose())

    def set_tip_target(self, position, quaternion):
        arm=self.env._robot.arm
        try:q=arm.solve_ik_via_jacobian(position,quaternion=quaternion)
        except Exception:
            try:q=arm.solve_ik_via_sampling(position,quaternion=quaternion,ignore_collisions=False,
                                           trials=100,max_configs=1,max_time_ms=10)[0]
            except Exception as exc:raise RuntimeError('Panda IK failed: '+str(exc)) from exc
        current=np.asarray(arm.get_joint_positions())
        q=current+np.clip(np.asarray(q)-current,-.08,.08)
        arm.set_joint_target_positions(q)

    def actuate_gripper(self, amount):
        if amount>.5:self.env._robot.gripper.release()
        return self.env._robot.gripper.actuate(amount,velocity=.2)

    def actuate_fingers(self,amount,*,velocity=.2):
        return self.env._robot.gripper.actuate(amount,velocity=velocity)

    def hold_fingers(self):
        gripper=self.env._robot.gripper
        gripper.set_joint_target_velocities([0.]*len(gripper.joints))
        gripper.set_joint_target_positions(gripper.get_joint_positions())

    def release_native(self,index):
        from pyrep.objects.shape import Shape
        gripper=self.env._robot.gripper;objects=gripper.get_grasped_objects()
        expected=self.entities[index]['primary_name']
        if [obj.get_name() for obj in objects]!=[expected]:raise RuntimeError('native release requires only predicted attachment')
        body=Shape(expected);before=np.asarray(body.get_pose())
        gripper.release()
        if pose_error(before,body.get_pose())>1e-6:raise RuntimeError('native release changed world pose')

    def hold_arm(self,path=None):
        """Stop commanding a stale contact trajectory; never set live poses."""
        from pyrep.backend import sim
        if path is not None and getattr(path,'_rml_handle',None) is not None:
            if not getattr(path,'_path_done',False):sim.simRMLRemove(path._rml_handle)
            path._rml_handle=None;path._path_done=True
        arm=self.env._robot.arm
        arm.set_joint_target_positions(arm.get_joint_positions())

    def try_grasp(self,index):
        from pyrep.objects.shape import Shape
        if not self.entities[index]['movable']:return False
        # Standard PyRep proximity-gated attachment, explicitly user-authorized.
        # Only the predicted entity is eligible; never enumerate task targets.
        shape=Shape(self.entities[index]['primary_name'])
        before=np.asarray(shape.get_pose())
        detected=self.env._robot.gripper.grasp(shape)
        if pose_error(before,shape.get_pose())>1e-6:raise RuntimeError('native grasp changed world pose')
        return detected

    def detect_grasp(self,index):
        """Read the native predicted-object gate without attachment or ticks."""
        from pyrep.objects.shape import Shape
        if not self.entities[index]['movable']:return False
        return bool(self.env._robot.gripper._proximity_sensor.is_detected(
            Shape(self.entities[index]['primary_name'])))

    def release_clearance(self,index):
        from pyrep.objects.shape import Shape
        from pyrep.const import ObjectType
        body=Shape(self.entities[index]['primary_name'])
        fingers=[s for s in self.env._robot.gripper.get_objects_in_tree(object_type=ObjectType.SHAPE)
                 if s.is_respondable() and 'finger' in s.get_name().lower()]
        if not fingers:raise RuntimeError('physical finger geometry unavailable')
        return not any(body.check_collision(finger) for finger in fingers)

    def finger_clearance(self,index):
        from pyrep.objects.shape import Shape
        from pyrep.const import ObjectType
        body=Shape(self.entities[index]['primary_name'])
        fingers=[s for s in self.env._robot.gripper.get_objects_in_tree(object_type=ObjectType.SHAPE)
                 if s.is_respondable() and 'finger' in s.get_name().lower()]
        if not fingers:raise RuntimeError('physical finger geometry unavailable')
        values=[0. if body.check_collision(finger) else float(body.check_distance(finger)) for finger in fingers]
        if not np.isfinite(values).all() or min(values)<0:raise RuntimeError('native finger distance unavailable')
        return min(values)

    def finger_opening(self):
        """Measured inner finger clearance along the actual tip closing axis."""
        from pyrep.const import ObjectType
        from scipy.spatial.transform import Rotation
        pose=self.tip_pose();rotation=Rotation.from_quat(pose[3:]).as_matrix()
        left=[];right=[]
        for shape in self.env._robot.gripper.get_objects_in_tree(object_type=ObjectType.SHAPE):
            if not shape.is_respondable() or 'finger' not in shape.get_name().lower():continue
            box=shape.get_bounding_box();transform=shape.get_matrix()
            local=np.array(list(itertools.product(*[(box[2*i],box[2*i+1]) for i in range(3)])))
            ys=((local@transform[:3,:3].T+transform[:3,3]-pose[:3])@rotation)[:,1]
            (left if ys.mean()<0 else right).extend(ys)
        if not left or not right:raise RuntimeError('missing opposing finger geometry')
        return float(min(right)-max(left))

    def planning_request(self,operation,index,destination,**options):
        from pyrep.backend import sim
        from pyrep.const import ObjectType
        from pyrep.objects.object import Object
        try:from .rlbench_planning import PlanningClient
        except ImportError:from rlbench_planning import PlanningClient
        # Do not transmit sensors, native goals, references or waypoints.
        before=self.snapshot_state();time=sim.simGetSimulationTime()
        attachments=[o.get_name() for o in self.env._robot.gripper.get_grasped_objects()]
        physical=[row for row in before if Object.get_object(row['name']).get_type() in (ObjectType.SHAPE,ObjectType.JOINT)]
        if operation=='grasp' and index not in self.observed_support_axes:
            from pyrep.objects.shape import Shape
            rotation=np.asarray(Shape(self.entities[index]['primary_name']).get_matrix())[:3,:3]
            self.observed_support_axes[index]=int(np.argmax(np.abs(rotation[2,:])))
        if getattr(self,'planner',None) is None:self.planner=PlanningClient()
        try:
            result=self.planner.request({'operation':operation,'object':index,'destination':destination,
                                         'scene':self.scene,'state':physical,'entities':self.entities,
                                         'support_axis':self.observed_support_axes.get(index),
                                         'placement_options':getattr(self,'placement_options',{}),
                                         'attached':self.entities[index]['primary_name'] in attachments,
                                         **options})
            if operation=='place' and result.get('placement') is not None:
                self.placement_options=result['placement']['placement_options']
            return result
        except RuntimeError:
            # A timed-out stream cannot be reused; never consume a stale reply
            # for a later episode or leave its native search running.
            self.planner.close();self.planner=None
            raise
        finally:
            after=self.snapshot_state()
            if (before!=after or time!=sim.simGetSimulationTime() or
                attachments!=[o.get_name() for o in self.env._robot.gripper.get_grasped_objects()]):
                raise RuntimeError('shadow planning changed live state')

    def release_sample(self,index):
        from pyrep.objects.shape import Shape
        body=self.geometry(index);shape=Shape(self.entities[index]['primary_name'])
        support_axis=getattr(self,'observed_support_axes',{}).get(index,int(np.argmin(body['dimensions'])))
        axis=np.asarray(shape.get_matrix())[:3,support_axis]
        return {'pose':np.asarray(shape.get_pose()).tolist(),'bottom':float(body['low'][2]),
                'attached':any(o.get_name()==self.entities[index]['primary_name'] for o in self.env._robot.gripper.get_grasped_objects()),
                'face_tilt':float(np.arccos(np.clip(abs(axis[2]),0.,1.)))}

    def requires_stable_face(self,index):
        try:from .rlbench_planning import face_stability_required
        except ImportError:from rlbench_planning import face_stability_required
        body=self.geometry(index)
        return face_stability_required(body,self.observed_support_axes.get(index))

    def checked_motion(self,position,orientation,*,withdraw=False,joint_path=None,tolerance=.008):
        if self.contact_entity is None:raise ValueError('motion requires predicted contact entity')
        result=self.planning_request('motion',self.contact_entity,None,
                                     position=np.asarray(position).tolist(),orientation=np.asarray(orientation).tolist(),withdraw=withdraw,
                                     joint_path=joint_path,target_tolerance=tolerance)
        import hashlib,json
        path=self.configuration_path(result['joint_path'])
        self.checked_motion_records.append({'joint_path_sha256':hashlib.sha256(json.dumps(result['joint_path']).encode()).hexdigest(),
            'configurations':len(result['joint_path']),'start':result['joint_path'][0],'goal':result['joint_path'][-1],
            'target_position':np.asarray(position).tolist(),'target_quaternion':np.asarray(orientation).tolist(),'linear_withdrawal':withdraw})
        return path

    def validate_arm_configuration(self,exempt):
        """Native whole-arm/self/scene collision check in the planning copy."""
        from pyrep.objects.shape import Shape
        flags=[]
        try:
            for i in exempt:
                for name in self.entities[i]['parts']:
                    shape=Shape(name);flags.append((shape,shape.is_collidable()));shape.set_collidable(False)
            if self.env._robot.arm.check_arm_collision():raise ValueError('cached path robot/scene collision')
        finally:
            for shape,value in flags:shape.set_collidable(value)

    def local_contact_exemptions(self):
        indices=[];neighbors=[]
        if self.contact_entity is not None:
            indices=[self.contact_entity];body=self.geometry(self.contact_entity)
            for i,row in enumerate(self.entities):
                if i==self.contact_entity or not row.get('movable',False):continue
                other=self.geometry(i)
                gap=np.maximum(np.maximum(np.asarray(body['low'])-other['high'],np.asarray(other['low'])-body['high']),0.)
                if np.linalg.norm(gap)<=CONTROL_PROFILE['contact_neighbor_gap']:neighbors.append(i)
        return indices+neighbors

    def configuration_path(self,points):
        from pyrep.robots.configuration_paths.arm_configuration_path import ArmConfigurationPath
        points=np.asarray(points,dtype=float)
        arm=self.env._robot.arm
        if points.ndim!=2 or points.shape[1]!=len(arm.joints) or not len(points) or not np.isfinite(points).all():
            raise ValueError('invalid checked joint path')
        if np.max(np.abs(np.asarray(arm.get_joint_positions())-points[0]))>.03:
            raise RuntimeError('checked path start differs from actual robot; replan required')
        if np.max(np.abs(points-points[0]))<1e-7:
            from types import SimpleNamespace
            return SimpleNamespace(step=lambda:True)
        return ArmConfigurationPath(arm,points.reshape(-1))

    def support_height(self,index,point):
        from pyrep.objects.shape import Shape
        return self.shape_support_height(Shape(self.entities[index]['primary_name']),point)

    def physical_support_height(self,name,point):
        from pyrep.objects.shape import Shape
        return self.shape_support_height(Shape(name),point)

    def shape_support_height(self,shape,point):
        matrix=shape.get_matrix()
        vertices,triangles=self.local_mesh(shape)
        origin=(np.asarray(point)-matrix[:3,3])@matrix[:3,:3]
        down=matrix[:3,:3].T@np.array([0.,0.,-1.])
        hits=ray_hits(vertices,triangles,origin,down)
        if not len(hits):raise ValueError('no physical support below released object')
        return float(point[2]-min(hits))

    def set_contact_entity(self,index):self.contact_entity=index

    def snapshot_state(self):
        from pyrep.const import ObjectType
        objects=self.task._task.get_base().get_objects_in_tree()
        objects+=self.env._robot.arm.get_objects_in_tree()
        rows=[];seen=set()
        for obj in objects:
            name=obj.get_name()
            if name in seen:continue
            seen.add(name)
            row={'name':name,'pose':np.asarray(obj.get_pose()).tolist()}
            if obj.get_type()==ObjectType.JOINT:row['joint']=obj.get_joint_position()
            if obj.get_type()==ObjectType.SHAPE:row['color']=obj.get_color()
            rows.append(row)
        return rows

    def restore_state(self, rows):
        from pyrep.objects.object import Object
        robot_names={obj.get_name() for obj in self.env._robot.arm.get_objects_in_tree()}
        robot_base=self.env._robot.arm.get_name()
        for row in rows:
            obj=Object.get_object(row['name'])
            # Robot link poses derive from joint coordinates. Setting every
            # link/joint world pose and then its joint value applies motion twice.
            if row['name'] not in robot_names or row['name']==robot_base:
                obj.set_pose(row['pose'],reset_dynamics=True)
            if 'color' in row:obj.set_color(row['color'])
        for row in rows:
            if 'joint' not in row:continue
            joint=Object.get_object(row['name'])
            # PyRep's disable_dynamics=True steps the entire simulator twice
            # per joint, moving joints restored earlier. Set all coordinates
            # without intervening physics and verify the result before stepping.
            joint.set_joint_position(row['joint'],disable_dynamics=False)
            joint.set_joint_target_position(row['joint'])
            joint.set_joint_target_velocity(0.)
        arm=self.env._robot.arm
        arm.set_joint_target_positions(arm.get_joint_positions())
        # Restoring saved states happens only at episode reset, never execution.

    def inventory(self):
        from pyrep.const import ObjectType
        root=self.task._task.get_base();groups={}
        for shape in root.get_objects_in_tree(object_type=ObjectType.SHAPE):
            if not (shape.is_renderable() or shape.is_respondable()):continue
            top=shape
            while top.get_parent() is not None and top.get_parent()!=root:top=top.get_parent()
            group=groups.setdefault(top.get_name(),{'root_name':top.get_name(),'parts':[]})
            group['parts'].append(shape.get_name())
        values=[]
        from pyrep.objects.shape import Shape
        for row in groups.values():
            shapes=[Shape(n) for n in row['parts']]
            primary=max(shapes,key=lambda s:bool(s.is_respondable())*1000+np.prod(np.diff(np.array(s.get_bounding_box()).reshape(3,2),axis=1)))
            row.update(primary_name=primary.get_name(),movable=bool(primary.is_dynamic()))
            bounds=[transformed_bounds(s.get_bounding_box(),s.get_matrix()) for s in shapes]
            center=(np.min([b[0] for b in bounds],axis=0)+np.max([b[1] for b in bounds],axis=0))/2
            values.append((tuple(center),row))
        return [row for _,row in sorted(values,key=lambda pair:pair[0])]

    def geometry(self,index):
        from pyrep.objects.shape import Shape
        row=self.entities[index];shapes=[Shape(n) for n in row['parts']]
        try:from .rlbench_geometry import world_vertices,symmetry_axis
        except ImportError:from rlbench_geometry import world_vertices,symmetry_axis
        physical=[s for s in shapes if s.is_respondable()]
        if not physical:physical=[Shape(row['primary_name'])]
        points=np.concatenate([world_vertices(self.local_mesh(s)[0],s.get_matrix()) for s in physical])
        low=points.min(axis=0);high=points.max(axis=0)
        primary=Shape(row['primary_name'])
        local=np.array(primary.get_bounding_box()).reshape(3,2)
        matrix=primary.get_matrix()[:3,:3]
        horizontal=[i for i in range(3) if abs(matrix[2,i])<.7]
        axis=min(horizontal,key=lambda i:local[i,1]-local[i,0])
        closing=matrix[:2,axis]
        # Panda's closing axis is local Y when the tip quaternion yaw is zero.
        yaw=float(np.arctan2(closing[1],closing[0])-np.pi/2)
        yaw=(yaw+np.pi/2)%np.pi-np.pi/2  # symmetric fingers; avoid needless wrist revolutions
        if not hasattr(self,'symmetry_cache'):self.symmetry_cache={}
        key=(row['primary_name'],tuple((local[:,1]-local[:,0]).round(6)))
        if key not in self.symmetry_cache:self.symmetry_cache[key]=symmetry_axis(self.local_mesh(primary)[0])
        return {'center':(low+high)/2,'low':low,'high':high,'movable':row['movable'],'yaw':yaw,
                'symmetry_axis':self.symmetry_cache[key],
                'dimensions':local[:,1]-local[:,0],
                'horizontal_span':float(min(local[i,1]-local[i,0] for i in horizontal))}

    def base_position(self):return np.asarray(self.env._robot.arm.get_position())

    def local_mesh(self,shape):
        # Coppelia 4.1/PyRep allocates native buffers on every mesh query.
        # Mesh topology is immutable within a reset; query once, copy, and only
        # transform cached vertices for FK. Invalidate on every scene reset.
        if not hasattr(self,'mesh_cache'):self.mesh_cache={}
        key=shape.get_handle() if hasattr(shape,'get_handle') else type(shape)
        if key not in self.mesh_cache:
            if hasattr(shape,'get_handle'):
                try:from .rlbench_geometry import native_mesh
                except ImportError:from rlbench_geometry import native_mesh
                vertices,triangles=native_mesh(shape.get_handle())
            else:vertices,triangles,_=shape.get_mesh_data()  # Pure test adapters.
            self.mesh_cache[key]=(np.array(vertices,dtype=float,copy=True).reshape(-1,3),
                                  np.array(triangles,dtype=int,copy=True).reshape(-1,3))
        return self.mesh_cache[key]

    def assembly_points(self):
        from pyrep.const import ObjectType
        points=[]
        for shape in self.env._robot.gripper.get_objects_in_tree(object_type=ObjectType.SHAPE,exclude_base=False):
            if not (shape.is_renderable() or shape.is_respondable()):continue
            box=shape.get_bounding_box();matrix=shape.get_matrix()
            local=np.array(list(itertools.product(*[(box[2*i],box[2*i+1]) for i in range(3)])))
            points.extend(local@matrix[:3,:3].T+matrix[:3,3])
        return np.asarray(points)

    def entry_orientation(self,inward):
        return self.entry_orientations(inward)[0]

    def entry_orientations(self,inward):
        from scipy.spatial.transform import Rotation
        z=np.asarray(inward,dtype=float);z=z/np.linalg.norm(z)
        y=np.array([0.,0.,1.]);x=np.cross(y,z);x=x/np.linalg.norm(x);y=np.cross(z,x)
        base=np.column_stack((x,y,z))
        current=self.tip_pose();rotation=Rotation.from_quat(current[3:]).as_matrix()
        local=(self.assembly_points()-current[:3])@rotation
        choices=[]
        for angle in np.arange(16)*np.pi/8:
            matrix=base@Rotation.from_rotvec([0.,0.,angle]).as_matrix()
            heights=(local@matrix.T)[:,2]
            choices.append((float(np.ptp(heights)),float((Rotation.from_matrix(rotation.T@matrix)).magnitude()),matrix))
        return [Rotation.from_matrix(item[2]).as_quat() for item in sorted(choices,key=lambda item:(round(item[0],4),item[1]))]

    def grasp_clearance(self,orientation):
        """Tip height above an observed support, from the actual hand envelope."""
        from scipy.spatial.transform import Rotation
        current=self.tip_pose()
        local=(self.assembly_points()-current[:3])@Rotation.from_quat(current[3:]).as_matrix()
        heights=(local@Rotation.from_quat(orientation).as_matrix().T)[:,2]
        return max(0.,-float(heights.min()))+.006

    def grasp_standoff(self,index,orientation,proposed):
        """Raise a rejected hover above nearby movable clutter, without moving it."""
        body=self.geometry(index);highest=body['high'][2]
        for i,row in enumerate(self.entities):
            if i==index or not row.get('movable',False):continue
            other=self.geometry(i)
            gap=np.maximum(np.maximum(np.asarray(body['low'])-other['high'],
                                      np.asarray(other['low'])-body['high']),0.)
            if np.linalg.norm(gap)<=CONTROL_PROFILE['contact_neighbor_gap']:highest=max(highest,other['high'][2])
        point=np.asarray(proposed,dtype=float).copy()
        point[2]=max(point[2],highest+self.grasp_clearance(orientation)+.02)
        return point

    def placement_route(self,index,held,*,require_fit=False):
        from pyrep.objects.shape import Shape
        shape=Shape(self.entities[index]['primary_name'])
        box=np.array(shape.get_bounding_box()).reshape(3,2)
        center=box.mean(axis=1);half_height=(held['high'][2]-held['low'][2])/2
        matrix=shape.get_matrix()
        local_up=matrix[:3,:3].T@np.array([0.,0.,1.])
        vertical=int(np.argmax(np.abs(local_up)))
        if abs(local_up[vertical])<.95:raise ValueError('tilted support needs a separate geometry adapter')
        sign=1 if local_up[vertical]>0 else -1
        up=np.zeros(3);up[vertical]=sign
        lower=box[vertical,0 if sign>0 else 1]
        upper=box[vertical,1 if sign>0 else 0]
        inside=center.copy();inside[vertical]=lower+sign*.08
        vertices,triangles=self.local_mesh(shape)
        directions=[up]+[np.eye(3)[axis]*s for axis in range(3) if axis!=vertical for s in (-1,1)]
        opening=None
        for direction in directions:
            if len(ray_hits(vertices,triangles,inside,direction))==0:
                opening=direction;break
        world=lambda point:matrix[:3,:3]@point+matrix[:3,3]
        if opening is None:
            # Flat support: place above it; never invent a hidden goal region.
            inside[vertical]=upper+sign*(half_height+.015)
            return world(inside),world(inside+up*.18)
        if np.dot(opening,up)>.9:
            inside[vertical]=upper+sign*(half_height+.025)
            return world(inside),world(inside+up*.10)
        if hasattr(self,'env'):
            down=ray_hits(vertices,triangles,inside,-up)
            above=ray_hits(vertices,triangles,inside,up)
            if not len(down) or not len(above):raise ValueError('unverified support/ceiling geometry')
            floor=world(inside)[2]-min(down);ceiling=world(inside)[2]+min(above)
            # Only the payload must be contained in the opening. Palm/wrist
            # may remain outside; the shadow planner checks their actual mesh
            # collisions throughout the path instead of requiring their entire
            # bounding boxes to fit between two infinite shelf planes.
            points=np.array([held['low'],held['high']])
            try:from .rlbench_planning import PLANNING_PROFILE
            except ImportError:from rlbench_planning import PLANNING_PROFILE
            lo=floor+PLANNING_PROFILE['payload_floor_clearance']-(points[:,2].min()-held['center'][2])
            hi=ceiling-.006-(points[:,2].max()-held['center'][2])
            if require_fit and lo>hi:raise ValueError('hand/payload does not fit observed opening')
            # Release near the observed support instead of dropping from the
            # middle of the opening. The complete hand/payload envelope stays
            # within the floor/ceiling clearance interval.
            center_z=min(hi,lo+getattr(self,'placement_options',{}).get('release_offset',.01))
            inside[vertical]+=sign*(center_z-world(inside)[2])
        # Deposit fully inside the opening without requiring the palm to reach
        # the back of the receptacle. This is geometry, not a success region.
        corners=np.array(list(itertools.product(*zip(held['low'],held['high']))))
        local_payload=(corners-np.asarray(held['center']))@matrix[:3,:3]
        axis=int(np.argmax(np.abs(opening)))
        half_depth=float(np.max(np.abs(local_payload[:,axis])))
        inset=max(0.,(box[axis,1]-box[axis,0])/2-half_depth-getattr(self,'placement_options',{}).get('front_margin',.04))
        inside+=opening*inset
        approach=inside+opening*((box[axis,1]-box[axis,0])/2+.16)
        return world(inside),world(approach)

    def placement_point(self,index,held):return self.placement_route(index,held)[0]

    def path_to(self,position,quaternion,*,contact=False):
        from pyrep.objects.shape import Shape
        indices=[];neighbors=[]
        if contact and self.contact_entity is not None:
            indices=[self.contact_entity]
            movable=[i for i,row in enumerate(self.entities) if i!=self.contact_entity and row.get('movable',False)]
            if movable:
                body=self.geometry(self.contact_entity)
                for i in movable:
                    other=self.geometry(i)
                    gap=np.maximum(np.maximum(np.asarray(body['low'])-other['high'],
                                              np.asarray(other['low'])-body['high']),0.)
                    if np.linalg.norm(gap)<=CONTROL_PROFILE['contact_neighbor_gap']:neighbors.append(i)
        # Preserve ordinary obstacle avoidance whenever it produces a path.
        # Local movable contact is a fallback, never a global collision bypass.
        attempts=[indices]+([indices+neighbors] if neighbors else [])
        for attempt,exempt in enumerate(attempts):
            flags=[]
            event={'predicted_entity_id':self.contact_entity,'kinematic_contact_entity_ids':exempt,
                   'physical_response_unchanged':True,'local_contact_fallback':attempt>0}
            try:
                for i in exempt:
                    for name in self.entities[i]['parts']:
                        shape=Shape(name);flags.append((shape,shape.is_collidable()));shape.set_collidable(False)
                effort=getattr(self,'planning_effort',{'max_configs':8,'trials_per_goal':3})
                if getattr(self,'linear_motion',False):
                    path=self.env._robot.arm.get_linear_path(position,quaternion=quaternion,ignore_collisions=False)
                else:path=self.env._robot.arm.get_path(position,quaternion=quaternion,ignore_collisions=False,**effort)
                event['path_found']=True
                return path
            except Exception as exc:
                event['path_found']=False;event['failure']=str(exc)
                if attempt==len(attempts)-1:raise RuntimeError('geometry-only Panda path failed: '+str(exc)) from exc
            finally:
                for shape,value in flags:shape.set_collidable(value)
                if contact:
                    if not hasattr(self,'planning_contacts'):self.planning_contacts=[]
                    self.planning_contacts.append(event)

    def calibration_reference(self):
        # Private native success objects supply calibration labels only. No
        # detector coordinates or waypoints are exposed to the executor.
        conditions=self.task._task._success_conditions
        selected=[c._obj.get_name() for c in conditions if hasattr(c,'_obj')]
        if len(selected)!=1:raise ValueError('native goal needs separate adjudication')
        destination='bin' if self.task.get_name()=='put_rubbish_in_bin' else 'cupboard'
        find=lambda name:next(i for i,row in enumerate(self.entities) if name in row['parts'])
        return [{'name':'pick','params':{'target_entity_name':find(selected[0])}},
                {'name':'place','params':{'target_container_name':find(destination)}}]

    def native_success(self):return bool(self.task._task.success()[0])
