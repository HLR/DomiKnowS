"""Optional real-physics checks; no trained-controller or benchmark performance claim."""
from types import SimpleNamespace
import numpy as np
import pytest
from ..research import ResearchConfig,EvidenceLedger


def test_real_mujoco_path_clone_does_not_write_live_state():
    mujoco=pytest.importorskip("mujoco")
    from ..research.geometry import MujocoPathClone,check_sampled_path
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="wall" type="sphere" pos=".4 0 0" size=".05"/>
      <body name="robot"><joint name="slide" type="slide" axis="1 0 0"/>
      <geom name="robot_geom" type="sphere" size=".05" mass="1"/></body>
    </worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    before=data.qpos.copy()
    def clone():
        result=MujocoPathClone(model,data,[0],[(0,1)],held_object=False)
        result.complete_pairs=True
        return result
    ledger=EvidenceLedger.parse({"episode":"real_physics","step":0})
    proof=check_sampled_path([[0.],[.4]],ledger,clone)
    assert proof.status=="blocked",proof.reason
    assert np.array_equal(data.qpos,before)
    assert not proof.continuous_safety


def test_articulation_sensor_reads_real_joint_after_motion():
    pytest.importorskip("dm_control")
    from dm_control import mjcf
    from ..research.runtime import reconcile
    from ..research.sensors import attach_sensor_adapter
    root=mjcf.RootElement()
    body=root.worldbody.add("body",name="laptop")
    joint=body.add("joint",name="lid",type="hinge")
    body.add("geom",type="box",size=[.1,.1,.01],mass=1)
    physics=mjcf.Physics.from_mjcf_model(root)
    env=SimpleNamespace(physics=physics,task=SimpleNamespace(entities={"laptop":SimpleNamespace(joints=[joint])}))
    context={"research_config":ResearchConfig("research_v1").as_dict(),"planning_facts":{"research":{
        "evidence":{"episode":"real_physics","step":0},"sensor_profile":"simulator_diagnostic",
        "sensors":[{"id":"lid","field":"joint","entity":"laptop","joint":"lid","dependencies":["joint:lid"]}]}}}
    attach_sensor_adapter(env,context)
    for step,angle in enumerate((1.19,1.2,1.5,1.51)):
        physics.bind(joint).qpos=[angle];physics.forward()
        context=reconcile(env,context,step=step,changed=("joint:lid",))
        ledger=EvidenceLedger.parse(context["planning_facts"]["research"]["evidence"])
        assert ledger.value("lid")==pytest.approx(angle)
        assert context["planning_facts"]["joint_observations"][0]["value"]==pytest.approx(angle)


def test_executed_macro_trace_keeps_terminal_control_and_uses_cloned_fk():
    pytest.importorskip("dm_control")
    from dm_control import mjcf
    from ..research.collection import ExecutedStepRecorder
    root=mjcf.RootElement();body=root.worldbody.add("body")
    joint=body.add("joint",type="slide",axis=[1,0,0])
    body.add("geom",type="sphere",size=[.05],mass=1)
    site=body.add("site",name="ee")
    physics=mjcf.Physics.from_mjcf_model(root)
    robot=SimpleNamespace(joints=[joint],n_dof=1,
        get_end_effector_pos=lambda p:np.asarray(p.bind(site).xpos),
        get_end_effector_quat=lambda p:np.array([1.,0.,0.,0.]))
    def observe(require_pcd=True):
        return {"ee_state":np.r_[physics.bind(site).xpos,[1,0,0,0,1]],"rgb":np.zeros((1,8,8,3),dtype=np.uint8)}
    executed=[]
    def step(command):
        executed.append(np.array(command));physics.bind(joint).qpos=[command[0]*.5];physics.forward()
        return SimpleNamespace(last=lambda:len(executed)==2)
    env=SimpleNamespace(robot=robot,physics=physics,step=step,get_observation=observe)
    recorder=ExecutedStepRecorder(env,camera_slots=[0])
    try:
        env.step([.2,.04,.04]);last=env.step([.4,0.,0.])
        assert last.last() and len(recorder.frames)==2
        assert [x[0] for x in recorder.states]==pytest.approx([0.,.1])
        assert [x[0] for x in recorder.actions]==pytest.approx([.2,.4])
        assert physics.bind(joint).qpos[0]==pytest.approx(.2)
        assert [x[-1] for x in recorder.actions]==[1.,0.]
    finally:recorder.close()


def test_flat_surface_postcondition_uses_real_support_and_rejects_held_objects():
    pytest.importorskip("dm_control")
    from dm_control import mjcf
    from ..research.collection import postcondition,measured_support
    root=mjcf.RootElement()
    floor=root.worldbody.add("geom",type="plane",size=[1,1,.1])
    body=root.worldbody.add("body",pos=[0,0,.1]);joint=body.add("joint",type="free")
    geom=body.add("geom",type="box",size=[.05,.05,.05],mass=1)
    physics=mjcf.Physics.from_mjcf_model(root)
    for _ in range(200):physics.step()
    obj=SimpleNamespace(geoms=[geom],get_xpos=lambda p:p.bind(body).xpos,is_grasped=lambda *args:False)
    def unavailable(*args):raise IndexError("No containment sites on this flat table")
    table=SimpleNamespace(geoms=[floor],z_threshold=.1,contain=unavailable)
    env=SimpleNamespace(physics=physics,robot=object(),task=SimpleNamespace(entities={"book":obj,"table":table}))
    assert measured_support(obj,table,physics) is True
    assert postcondition(env,"place",{"target_entity_name":"book","target_container_name":"table"},{}) is True
    obj.is_grasped=lambda *args:True
    assert postcondition(env,"place",{"target_entity_name":"book","target_container_name":"table"},{}) is False
