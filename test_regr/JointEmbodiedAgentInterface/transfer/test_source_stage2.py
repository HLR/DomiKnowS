from dataclasses import replace
import json

import pytest
import torch

from ..generalization.executor_policy import EXECUTOR, bind_plan_policy
from ..test.test_constrained_stage2 import eai_world, sample
from ..test.test_stage1_dfa_loss import joint_fixture
from ..utils import observed_eai
from ..utils.structured_inference import StructuredPlanCandidate, select_candidate_with_ilp
from .fresh_metadata.checkpoint import FORMAT, load_planner_snapshot
from .protocol import freeze, sha
from .source_stage2 import SOURCE_PROFILE
from .source_stage2_rollout import HORIZON, PROTOCOL, TASKS, calibration_records, verify_manifest


def test_observed_eai_ignores_task_name_reference_and_goal_override(eai_world):
    row = sample("inside(apple.1, fridge)", init="(open fridge) (containers fridge)")
    row["natural_language_description"] = "Put the first apple in the fridge."
    before, goal = observed_eai.input_and_goal(row, eai_world)
    after, other_goal = observed_eai.input_and_goal({
        **row, "causal_prompt_text": "Target apple_2", "goal": "inside(apple.2, fridge)",
        "task_name": "Choose the second apple", "target_action_tokens": ["grab", "apple_2"],
        "generation_entity_types": ("reference_only",), "task_target_container": "cabinet",
        "transition_model": row["transition_model"].replace("(:goal (inside apple_2 fridge))",
                                                            "(:goal (inside apple_2 cabinet))"),
    }, eai_world)
    assert before == after
    assert before["instruction"] == row["natural_language_description"]
    assert goal == other_goal


def test_stage2_reward_protocol_is_stable_on_resume():
    assert observed_eai.reward_protocol("auto", {"stage": "stage1"}) == observed_eai.OBSERVED
    assert observed_eai.reward_protocol("auto", {"stage": "stage2"}) == observed_eai.LEGACY
    with pytest.raises(ValueError, match="differs"):
        observed_eai.reward_protocol(observed_eai.OBSERVED, {"stage": "stage2"})


def test_ilp_does_not_use_evaluation_outcomes_and_has_no_illegal_fallback():
    def solver(probabilities, eligible, _):
        return max(eligible, key=probabilities.__getitem__), "optimal"

    candidate = StructuredPlanCandidate((1,), -.1, (), True, True, True, 1.0, goal_success=False)
    second = replace(candidate, labels=(2,), log_probability=-.2, goal_success=True)
    assert select_candidate_with_ilp([candidate, second], "eai", solver=solver).selected.labels == (1,)
    infeasible = select_candidate_with_ilp([replace(candidate, dfa_valid=False)], "eai", solver=solver)
    assert infeasible.solver_status == "infeasible"
    assert not infeasible.selected.hard_valid and infeasible.selected.labels == ()


def test_source_capability_uses_compiled_policy_and_keeps_two_legal_sources(joint_fixture):
    _, runtime = joint_fixture
    vocabulary = runtime.vlabench_vocabulary
    policy = bind_plan_policy(EXECUTOR, vocabulary, 12, {"entity_table": ["a", "b"]}, SOURCE_PROFILE)
    def accepts(*tokens):
        return policy.accepts([vocabulary.label_for_token(x) for x in (*tokens, vocabulary.eos_token)])

    for entity in ("obj:0", "obj:1"):
        assert accepts("skill:pick", "arg:target_entity_name", entity)
    assert not accepts("skill:pour", "arg:target_container_name", "obj:1")
    assert not accepts("skill:place", "arg:target_container_name", "obj:1")
    assert accepts("skill:pick", "arg:target_entity_name", "obj:0", "skill:pour",
                   "arg:target_container_name", "obj:1")


def test_epoch6_loader_rejects_lineage_and_preserves_frozen_base(tmp_path):
    planner = torch.nn.Module()
    planner.base = torch.nn.Linear(1, 1).requires_grad_(False)
    planner.lora_weight = torch.nn.Parameter(torch.zeros(1))
    planner.label_heads = torch.nn.ModuleDict({"eai": torch.nn.Linear(1, 1)})
    weights = {name: torch.ones_like(param) for name, param in planner.named_parameters()
               if not name.startswith("base.")}
    path = tmp_path / "epoch6.pt"
    record = {"path": path, "settings_sha256": "lineage", "epoch": 6}
    torch.save({"format": FORMAT, "settings_sha256": "lineage", "epoch": 6, "planner": weights}, path)
    record["sha256"] = sha(path)
    before = planner.base.weight.detach().clone()
    load_planner_snapshot(planner, record)
    assert torch.equal(planner.base.weight, before)
    assert not planner.base.weight.requires_grad
    assert torch.equal(planner.lora_weight, torch.ones(1))
    with pytest.raises(ValueError, match="lineage"):
        load_planner_snapshot(planner, {**record, "epoch": 5})


def test_source_qualification_rejects_same_scene_and_wrong_source_success(tmp_path):
    rows = []
    controls = []
    for split in ("calibration", "training", "validation"):
        for task in TASKS:
            index = len(rows)
            inputs = tmp_path / f"prompt-{index}.json"
            image = tmp_path / f"image-{index}.png"
            image.write_bytes(bytes([index]))
            freeze(inputs, {"instruction": "Pick the red object.", "entities": ["a", "b"], "images": [str(image)]})
            rows.append({"episode_id": f"{task}/{index}", "task": task, "split": split, "seed": index,
                         "scene_sha256": str(index), "generation_input": str(inputs),
                         "files": {str(path): sha(path) for path in (inputs, image)}})
            if split == "calibration":
                controls.append({"episode_id": rows[-1]["episode_id"], "correct": {"success": True},
                                 "wrong_source": {"success": False}})
    proof = tmp_path / "qualification.json"
    freeze(proof, {"passed": True, "executor_sha256": "executor", "controls": controls})
    manifest = {"protocol": PROTOCOL, "horizon": HORIZON, "executor_sha256": "executor", "episodes": rows,
                "qualification": str(proof), "qualification_sha256": sha(proof)}
    assert verify_manifest(manifest) == rows
    duplicated = [*rows[:-1], {**rows[-1], "scene_sha256": rows[0]["scene_sha256"]}]
    with pytest.raises(ValueError, match="scene_sha256"):
        verify_manifest({**manifest, "episodes": duplicated})
    controls[0]["wrong_source"]["success"] = True
    changed = tmp_path / "bad-qualification.json"
    freeze(changed, {"passed": True, "executor_sha256": "executor", "controls": controls})
    with pytest.raises(ValueError, match="failed correct-plan"):
        verify_manifest({**manifest, "qualification": str(changed), "qualification_sha256": sha(changed)})


def test_archived_calibration_requires_replayed_controls_for_every_task(tmp_path):
    path = tmp_path / "calibration.jsonl"
    rows = [{"task": task, "seed": i, "native_success": True, "replay_success": True}
            for i, task in enumerate(TASKS)]
    path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    assert set(calibration_records(path, 1)) == set(TASKS)
    rows[0]["replay_success"] = False
    path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration task and replay"):
        calibration_records(path, 1)
