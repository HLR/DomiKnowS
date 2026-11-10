import copy
import json
import pytest
from ..generalization.rlbench_next_test import DEFAULT_CONFIG, build_plan


def test_rlbench_followup_is_reproducible_and_calibration_is_disjoint():
    config=json.loads(DEFAULT_CONFIG.read_text())
    plan=build_plan(config)
    assert plan==build_plan(copy.deepcopy(config))
    assert plan['native_task_classes']==2 and plan['task_variation_cases']==10
    assert len({e['episode_id'] for e in plan['episodes']})==200
    assert len({e['seed'] for e in plan['episodes']})==200
    assert plan['counts']=={'evaluation':100,'calibration':100}
    assert not plan['ready_for_model_evaluation']
    assert 'registered_graspable_object_list' in config['forbidden_generation_inputs']
    assert 'task_low_dim_state' in config['forbidden_generation_inputs']


def test_rlbench_followup_rejects_overlapping_resets_and_unsupported_skills():
    config=json.loads(DEFAULT_CONFIG.read_text())
    config['reset_indices']['calibration'][0]=0
    with pytest.raises(ValueError,match='disjoint'):build_plan(config)
    config=json.loads(DEFAULT_CONFIG.read_text())
    config['tasks'][0]['skills'].append('open')
    with pytest.raises(ValueError,match='unsupported'):build_plan(config)
    config=json.loads(DEFAULT_CONFIG.read_text())
    config['tasks'][0]['variations'].append(0)
    with pytest.raises(ValueError,match='duplicate'):build_plan(config)
