import json
from types import SimpleNamespace

import pytest

from test_regr.JointEmbodiedAgentInterface.utils import matched_vlabench as matched


def test_scene_correct_plan_uses_same_table_as_decoder():
    env = SimpleNamespace(task=SimpleNamespace(target_entity="pear", target_container="plate"))
    entities = ["orange", "pear", "plate"]
    plan = matched.expert_plan(env, "select_fruit", entities)
    assert matched.resolve(plan[0], entities) == {"target_entity_name": "pear"}
    assert matched.resolve(plan[1], entities) == {"target_container_name": "plate"}
    plan[0]["params"]["target_entity_name"] = 0
    assert matched.resolve(plan[0], entities) == {"target_entity_name": "orange"}
    plan[0]["params"]["target_entity_name"] = -1
    with pytest.raises(ValueError, match="Unresolved"):
        matched.resolve(plan[0], entities)


def test_expert_executes_distractor_and_requires_native_success(monkeypatch):
    from test_regr.VLABenchAgentInterface.simulation import program
    env = SimpleNamespace(task=SimpleNamespace(entities={"apple": 1, "pear": 2}, target_entity="pear"))
    row = {}
    trace = SimpleNamespace(row=row, select=lambda *args: None, subject=None)
    calls = []

    def pick(env, target_entity_name):
        calls.append(target_entity_name)
        return [], [], True, True  # upstream also reports True on time-limit LAST

    monkeypatch.setattr(matched, "skill_calls", lambda env, task, skill, args: [(pick, args)])
    monkeypatch.setattr(program, "_task_success", lambda env: False)
    plan = [{"name": "pick", "params": {"target_entity_name": 0}}]
    matched.execute_plan(env, "select_fruit", plan, trace)
    assert calls == ["apple"]
    assert row["native_success"] is False
    assert row["first_failed_skill"] == "pick"


def test_cohort_requires_verified_control_for_every_source_reset(tmp_path):
    path = tmp_path / "controls.jsonl"
    rows = [dict(task=t, seed=s, planner="expert", controller="expert", native_success=True)
            for t, s in matched.source_cases()]
    path.write_text("\n".join(map(json.dumps, rows)))
    assert len(matched.read_controls(path)) == 12
    rows[-1]["native_success"] = False
    path.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="Unresolved positive control"):
        matched.read_controls(path)
