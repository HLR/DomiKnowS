"""Goal-blind Panda OSC pick/place. Runs in the isolated LIBERO Python runtime."""
import numpy as np


def bounded_osc_action(position, current, rotation_error, scale, grip):
    scale = np.asarray(scale)
    values = np.r_[np.asarray(position)-np.asarray(current), rotation_error]
    if scale.shape != (6,) or values.shape != (6,) or not np.isfinite(values).all() or not (scale > 0).all():
        raise ValueError('invalid OSC frame/scale')
    return np.r_[np.clip(values/scale, -1, 1), 1. if grip > 0 else -1.]


class FixedExecutor:
    def __init__(self, env, entities, *, horizon=1000):
        self.env = env
        self.entities = tuple(entities)
        self.horizon = int(horizon)
        self.steps = 0
        self.obs = None
        self.events = []
        self.held = None

    def body_position(self, name):
        model = self.env.env.objects_dict.get(name) or self.env.env.fixtures_dict.get(name)
        if model is None:
            raise ValueError('unknown entity body')
        body = self.env.sim.model.body_name2id(model.root_body)
        return np.asarray(self.env.sim.data.body_xpos[body]).copy()

    def tick(self, position, grip, *, count=1):
        from robosuite.utils import transform_utils as T
        for _ in range(count):
            if self.steps >= self.horizon:
                raise TimeoutError('fixed executor horizon exhausted')
            current = np.asarray(self.obs['robot0_eef_pos'])
            rotation = T.quat2mat(np.asarray(self.obs['robot0_eef_quat']))
            desired = np.diag([1., -1., -1.])
            error = T.quat2axisangle(T.mat2quat(desired @ rotation.T))
            controller = self.env.env.robots[0].controller
            scale = np.asarray(controller.output_max)
            action = bounded_osc_action(position, current, error, scale, grip)
            if not np.isfinite(action).all():
                raise ValueError('nonfinite fixed-executor command')
            self.obs, _, _, _ = self.env.step(action)
            self.steps += 1

    def move(self, position, grip, *, limit=80, tolerance=.035):
        for _ in range(limit):
            self.tick(position, grip)
            if np.linalg.norm(np.asarray(self.obs['robot0_eef_pos'])-position) < tolerance:
                self.tick(position, grip, count=4)
                return
        raise TimeoutError(f'OSC waypoint did not converge: target={np.asarray(position).tolist()} '
                           f'actual={np.asarray(self.obs["robot0_eef_pos"]).tolist()}')

    def pick(self, name):
        initial = self.body_position(name)
        offsets = ([0., .035, .025], [0., -.035, .025], [.035, 0., .025],
                   [-.035, 0., .025], [0., 0., .05]) if 'bowl' in name else (
            [0., 0., .01], [0., .018, 0.], [.018, 0., 0.])
        for offset in offsets:
            point = self.body_position(name) + np.asarray(offset)
            above = point + [0., 0., .16]
            self.move(above, -1)
            self.tick(above, -1, count=10)
            try:
                self.move(point, -1, limit=70, tolerance=.025)
            except TimeoutError:
                error = np.asarray(self.obs['robot0_eef_pos']) - point
                if np.linalg.norm(error[:2]) > .025 or not 0 <= error[2] <= .07:
                    continue
            contact_pose = np.asarray(self.obs['robot0_eef_pos']).copy()
            self.tick(contact_pose, 1, count=25)
            self.move(above, 1, limit=100)
            self.tick(above, 1, count=10)
            if self.body_position(name)[2] > initial[2] + .035:
                self.held = name
                self.events.append({'skill': 'pick', 'entity': name, 'success': True})
                return
            self.tick(above, -1, count=10)
        raise RuntimeError('grasp did not lift requested object')

    def place(self, name):
        if self.held is None or self.held == name:
            raise ValueError('place requires a different held object')
        target = self.body_position(name)
        offset = np.asarray(self.obs['robot0_eef_pos']) - self.body_position(self.held)
        release_object = target + [0., 0., .08]
        release = release_object + offset
        above = release + [0., 0., .14]
        self.move(above, 1, limit=120)
        self.move(release, 1, limit=100)
        self.tick(release, -1, count=25)
        self.move(above, -1)
        self.tick(above, -1, count=25)
        self.events.append({'skill': 'place', 'entity': name, 'success': True})
        self.held = None

    def execute(self, plan, obs):
        self.obs = obs
        reason = 'completed'
        try:
            if not plan:
                raise ValueError('empty plan')
            for operation in plan:
                skill = operation.get('name')
                if skill not in ('pick', 'place'):
                    raise ValueError('unsupported skill; no replacement or assistance')
                role = 'target_entity_name' if skill == 'pick' else 'target_container_name'
                params = operation.get('params', {})
                target = params.get(role)
                if set(params) != {role} or type(target) is not int or not 0 <= target < len(self.entities):
                    raise ValueError('invalid entity pointer')
                getattr(self, skill)(self.entities[target])
        except (TimeoutError, RuntimeError, ValueError) as error:
            reason = str(error)
        return {'steps': self.steps, 'termination_reason': reason, 'events': self.events}
