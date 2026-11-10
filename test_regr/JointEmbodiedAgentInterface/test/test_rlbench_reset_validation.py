import copy
import pytest
from ..generalization.rlbench_bridge import pose_error
from ..generalization.rlbench_reset_validation import validate_reset


@pytest.mark.parametrize('name,amount,accepted',[
    ('unused_waypoint',13.2e-6,True),('waypoint3_physical_shape',13.2e-6,False),
    ('goal_proximity_sensor',13.2e-6,False),('task_root',13.2e-6,False),
    ('robot_root',13.2e-6,False),('robot_tip',.001,True),('robot_tip',.003,False)])
def test_reset_scope_uses_object_role_not_name_and_preserves_strict_physical_and_scoring_checks(name,amount,accepted):
    names=['unused_waypoint','waypoint3_physical_shape','goal_proximity_sensor','task_root','robot_root','robot_tip']
    expected=[{'name':n,'pose':[0.,0.,0.,0.,0.,0.,1.]} for n in names]
    restored={r['name']:copy.deepcopy(r) for r in expected};restored[name]['pose'][0]=amount
    untouched=copy.deepcopy(restored)
    options=dict(robot_names={'robot_root','robot_tip'},robot_base='robot_root',task_base='task_root',
        dummy_names={'unused_waypoint','task_root','robot_tip'},pose_error=pose_error)
    if accepted:
        result=validate_reset(expected,restored,**options)
        assert result['reset_pose_max_error']==0.
        assert result['unused_helper_pose_error']==(amount if name=='unused_waypoint' else 0.)
        assert result['passive_robot_pose_error']==(amount if name=='robot_tip' else 0.)
    else:
        with pytest.raises(ValueError):validate_reset(expected,restored,**options)
    assert restored==untouched, 'verification must not rewind or edit the simulator state'


def test_joint_tolerance_and_complete_object_coverage_are_not_relaxed():
    expected=[{'name':'joint','joint':.2,'pose':[0,0,0,0,0,0,1]}]
    options=dict(robot_names={'joint'},robot_base='base',task_base='scene',dummy_names=set(),pose_error=pose_error)
    with pytest.raises(ValueError,match='missing'):validate_reset(expected,{},**options)
    restored={'joint':dict(expected[0],joint=.20002)}
    with pytest.raises(ValueError,match='physical/scoring'):validate_reset(expected,restored,**options)
