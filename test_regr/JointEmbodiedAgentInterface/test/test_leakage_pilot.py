import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

from test_regr.VLABenchAgentInterface.dataset import PlanningExample
from test_regr.VLABenchAgentInterface.official.official_protocol import OfficialExample, build_ti_list
from ..utils.leakage_pilot import (
    VERSION, PilotObserver, audit_overlap, digest, donor_map, file_digest,
    freeze_row, guard_settings, load_manifest, pilot_examples, select_pilot,
)


def example(index, task="density_qa", instruction=None):
    return PlanningExample(f"PhysLaw/{task}/example{index}", instruction or f"question {index}",
        ({"name": "press", "params": {"target_entity_name": 0}},),
        (Path(f"{index}.png"),), (Path(f"{index}_mask.png"),), None,
        entities=("button0", "metal_sphere"), task=task, dimension="PhysLaw")


def manifest_for(rows):
    return {"rows": [{**freeze_row(x), "corrected_reference":
                     [{"name": "press", "params": {"target_entity_name": 1}}],
                     "repeated_text_input": False} for x in rows],
            "image_donors": donor_map(rows),
            "instruction_donors": donor_map(rows, different_instruction=True),
            "size": len(rows)}


def test_selection_balances_tasks_and_repeated_inputs_deterministically():
    repeated = example(1, instruction="repeat")
    splits = {"train": [repeated], "test": [replace(repeated, episode_id="other"),
               example(2), example(3), example(4, "other"), example(5, "other")]}
    rows = select_pilot(splits, 4)
    assert [x.episode_id for x in rows] == [x.episode_id for x in select_pilot(splits, 4)]
    assert len({x.episode_id for x in rows}) == 4
    assert "other" in {x.episode_id for x in rows}
    assert sum(x.task == "other" for x in rows) == 2


def test_donors_are_bijections_without_self_or_identical_instruction():
    rows = [example(i, instruction=f"instruction {i//2}") for i in range(12)]
    donors = donor_map(rows, different_instruction=True)
    lookup = {x.episode_id: x for x in rows}
    assert len(set(donors.values())) == len(rows)
    assert all(k != v and lookup[k].instruction != lookup[v].instruction for k, v in donors.items())
    assert donors == donor_map(rows, different_instruction=True)
    with pytest.raises(ValueError, match="donors"):
        donor_map([example(1, instruction="same"), example(2, instruction="same")], different_instruction=True)


def test_references_and_pointer_tables_survive_instruction_and_image_probes():
    rows = [example(1), example(2)]
    manifest = manifest_for(rows)
    shuffled = pilot_examples(rows, manifest, "shuffled-instructions")
    assert shuffled[0].instruction == rows[1].instruction
    assert shuffled[0].entities == rows[0].entities
    assert shuffled[0].operation_sequence[0]["params"]["target_entity_name"] == 1
    assert rows[0].operation_sequence[0]["params"]["target_entity_name"] == 0
    images = pilot_examples(rows, manifest, "shuffled-images")
    assert images[0].image_paths == rows[1].image_paths
    assert images[0].segmented_image_paths == rows[1].segmented_image_paths
    assert images[0].instruction == rows[0].instruction


@pytest.mark.parametrize("condition", ["anonymous-objects", "anonymous-no-images"])
def test_anonymization_is_prompt_only_and_removes_button_hints(condition):
    from test_regr.VLABenchAgentInterface.models import planner_prompt
    class Vocab:
        skills = ("press",)
    class Planner:
        vocabulary = lambda self, domain: Vocab()
        def _prompt_base(self, domain, context):
            return planner_prompt(context["instruction"], context["entity_table"], self.vocabulary(domain))
        def _chat(self, text, count): return f"CHAT {count}: {text}"
    rows = [example(1), example(2)]; manifest = manifest_for(rows)
    processed = pilot_examples(rows, manifest, condition)
    planner = Planner(); old = planner._prompt_base
    with PilotObserver(io.StringIO(), processed, manifest, condition, planner) as observer:
        context = {"instruction": processed[0].instruction, "entity_table": processed[0].entities}
        prompt = planner._prompt_base("vlabench", context)
        assert "button0" not in prompt and "metal_sphere" not in prompt
        assert "Multiple-choice" not in prompt and "corresponding button" not in prompt
        assert "0: numbered object 0" in prompt and "1: numbered object 1" in prompt
        assert processed[0].entities == rows[0].entities and context["entity_table"] == rows[0].entities
        assert (not processed[0].image_paths) == (condition == "anonymous-no-images")
        assert observer.prompt == prompt
    assert planner._prompt_base == old


@pytest.mark.parametrize("changed", ["checkpoint", "manifest", "condition"])
def test_resume_rejects_changed_identity(tmp_path, changed):
    settings = {"checkpoint": "a", "manifest": "b", "condition": "original"}
    path = tmp_path/"settings.json"
    guard_settings(path, settings); guard_settings(path, settings)
    with pytest.raises(ValueError, match="resume configuration"):
        guard_settings(path, {**settings, changed: "changed"})
    assert json.loads(path.read_text()) == settings


def test_manifest_rejects_checkpoint_change_and_tampering(tmp_path):
    checkpoint = tmp_path/"model.pt"; checkpoint.write_bytes(b"original")
    manifest = {"version": VERSION, "checkpoint_sha256": file_digest(checkpoint),
                "rows": [{"episode_id": "one"}], "size": 1, "source_hashes": {}}
    manifest["manifest_sha256"] = digest(manifest)
    path = tmp_path/"manifest.json"; path.write_text(json.dumps(manifest))
    assert load_manifest(path, checkpoint)["size"] == 1
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint/version"):
        load_manifest(path, checkpoint)
    manifest["size"] = 2; path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        load_manifest(path, checkpoint)


def test_probe_requires_manifest_before_model_loading():
    from ..eval_planning import build_eval_parser, run_planning_evaluation
    args = build_eval_parser().parse_args(["--checkpoint", "unused", "--pilot-condition", "anonymous-objects"])
    with pytest.raises(ValueError, match="requires --pilot-manifest"):
        run_planning_evaluation(args)


@pytest.mark.parametrize("changed", ["seed", "planning_root"])
def test_manifest_split_identity_checked_before_model_loading(tmp_path, monkeypatch, changed):
    from ..eval_planning import build_eval_parser, run_planning_evaluation
    from ..utils import leakage_pilot
    checkpoint = tmp_path/"model.pt"; checkpoint.write_bytes(b"fixture")
    args = build_eval_parser().parse_args(["--checkpoint", str(checkpoint), "--pilot-manifest", "manifest",
        "--pilot-condition", "original", "--knowledge", "grammar", "--no-dfa", "--split", "test",
        "--planning-dir", str(tmp_path), "--output", str(tmp_path/"report.json")])
    manifest = {"seed": 42, "planning_root": str(tmp_path)}
    manifest[changed] = 43 if changed == "seed" else str(tmp_path/"different")
    monkeypatch.setattr(leakage_pilot, "load_manifest", lambda *a: manifest)
    with pytest.raises(ValueError, match="seed/planning root"):
        run_planning_evaluation(args)


def test_official_zero_shot_does_not_consume_reference_or_task_identity():
    ex = OfficialExample("id", "secret_task", "PhysLaw", "Choose an object.", Path("image"),
                         Path("mask"), {"secret_gold": 123}, ({"name": "press"},), "hash")
    baseline = build_ti_list("fixed prompt", ex)
    changed = replace(ex, task="OTHER_TASK", dimension="OTHER_DIMENSION", episode_id="OTHER_ID",
                      reference_payload={"LEAK": "answer"}, reference=(), reference_sha256="other")
    assert build_ti_list("fixed prompt", changed) == baseline
    assert "secret" not in str(baseline) and "123" not in str(baseline)
    assert "secret_gold" in str(build_ti_list("fixed prompt", changed, shots=[ex]))


def test_joint_prompt_ignores_gold_plan_and_oracle_task_fields():
    from types import SimpleNamespace
    from ..models import JointQwenVLPlanner
    stub = SimpleNamespace(vocabulary=lambda domain: SimpleNamespace(skills=("pick", "lift")))
    context = {"instruction": "Pick the metal object.", "entity_table": ("metal_sphere", "wood_cube")}
    baseline = JointQwenVLPlanner._prompt_base(stub, "vlabench", context)
    poisoned = {**context, "operation_sequence": [{"SECRET_GOLD": 12345}],
                "target_plan_labels": [999], "task": "SECRET_TASK", "episode_id": "SECRET_ID",
                "task_skill_sequence": ["SECRET_SKILL"], "task_target_entity": "SECRET_TARGET"}
    assert JointQwenVLPlanner._prompt_base(stub, "vlabench", poisoned) == baseline
    assert "metal_sphere" in baseline  # Named scene objects are observable prompt inputs.


def test_decoded_image_overlap_detects_different_png_encoding(tmp_path):
    from PIL import Image
    rows = []
    for i in range(2):
        ex = example(i)
        root = tmp_path/ex.episode_id; (root/"env_config").mkdir(parents=True)
        (root/"input").mkdir()
        (root/"env_config/config.json").write_text(json.dumps({"task": {"components": [{"name": str(i)}]}}))
        p = root/"input/input.png"
        Image.new("RGB", (10, 10), (100, 20, 30)).save(p, compress_level=0 if i == 0 else 9)
        rows.append(replace(ex, image_paths=(p,), segmented_image_paths=()))
    result = audit_overlap({"train": rows[:1], "validation": [], "test": rows[1:]}, tmp_path)
    assert result["cross_split_matches"]["test"]["image_bytes"] == []
    assert result["cross_split_matches"]["test"]["image_pixels"] == [rows[1].episode_id]
