import itertools
import json

import pytest

from ..repair_density_labels import (
    DENSITY_ORDER, episode_repair, generator_repair, migrate, restore, scene_answer,
    updated_density_examples,
)


def config_for(materials, instruction, old="button2"):
    components = [{"name": "table", "class": "Table"}]
    for i, material in enumerate(materials):
        components.extend([
            {"name": f"button{i}", "class": "Button", "position": [i, 0, 0]},
            {"name": material + "_sphere", "class": "RandomGeom",
             "material": material, "position": [i, .1, 0]},
        ])
    return {"task": {"components": components, "instructions": [instruction],
                     "conditions": {"press_button": {"target_button": old}}}}


@pytest.mark.parametrize("materials", list(itertools.permutations(("wood", "stone", "metal"))))
@pytest.mark.parametrize("instruction, material", [
    ("Choose the object with the largest density.", "metal"),
    ("Choose the object with the medium density.", "stone"),
    ("Choose the object with the smallest density.", "wood"),
    ("Choose the object that can float in the water.", "wood"),
])
def test_material_choice_is_independent_of_button_order_and_labels(materials, instruction, material):
    config = config_for(materials, instruction)
    expected = materials.index(material)
    assert scene_answer(config, instruction)[:2] == (f"button{expected}", 1 + expected * 2)
    # Selecting the answer must not depend on stored success conditions.
    del config["task"]["conditions"]
    assert scene_answer(config, instruction)[:2] == (f"button{expected}", 1 + expected * 2)


def write_episode(root, old="button2", target=5):
    episode = root / "PhysicsLaw" / "density_qa" / "example49"
    instruction = "Choose the object with the largest density."
    for directory in ("env_config", "input", "output"):
        (episode / directory).mkdir(parents=True)
    (episode / "env_config/env_config.json").write_text(json.dumps(config_for(("stone", "metal", "wood"), instruction, old)))
    (episode / "output/operation_sequence.json").write_text(json.dumps({
        "skill_sequence": [{"name": "press", "params": {"target_entity_name": target}}],
        "dependency": "Sequential",
    }))
    (episode / "input/instruction.txt").write_text(instruction)
    (episode / "input/input.png").write_bytes(b"unchanged image")
    return episode


@pytest.mark.parametrize("old_target, new_target", [(5, 3), ("button2", "button1")])
def test_repair_updates_both_label_forms_and_preserves_other_fields(tmp_path, old_target, new_target):
    episode = write_episode(tmp_path, target=old_target)
    row, changes = episode_repair(episode)
    assert row["old_button"] == "button2" and row["new_button"] == "button1"
    assert len(changes) == 2
    for path, before, after in changes:
        original, changed = json.loads(before), json.loads(after)
        if path.parent.name == "output":
            original["skill_sequence"][0]["params"]["target_entity_name"] = new_target
        else:
            original["task"]["conditions"]["press_button"]["target_button"] = "button1"
        assert changed == original


def test_apply_backs_up_originals_and_is_idempotent(tmp_path):
    root = tmp_path / "data"
    episode = write_episode(root)
    before = {p: p.read_bytes() for p in episode.rglob("*") if p.is_file()}
    assert migrate(root)["changed_episodes"] == 1
    assert all(p.read_bytes() == content for p, content in before.items())
    backup = tmp_path / "backup"
    report = migrate(root, backup_dir=backup, apply=True)
    assert report["applied"] and len(report["files"]) == 2
    for entry in report["files"]:
        assert (backup / entry["backup"]).read_bytes() == before[type(root)(entry["path"])]
    assert (episode / "input/input.png").read_bytes() == b"unchanged image"
    assert migrate(root)["changed_episodes"] == 0


def test_inconsistent_reference_aborts_whole_batch(tmp_path):
    root = tmp_path / "data"
    episode = write_episode(root, target=1)
    before = (episode / "env_config/env_config.json").read_bytes()
    assert len(migrate(root)["errors"]) == 1
    with pytest.raises(ValueError, match="partial repair"):
        migrate(root, backup_dir=tmp_path / "backup", apply=True)
    assert (episode / "env_config/env_config.json").read_bytes() == before
    assert not (tmp_path / "backup").exists()


@pytest.mark.parametrize("mutation", ["overlap", "unknown_material", "conflicting_instruction", "ambiguous_float"])
def test_ambiguous_grounding_is_rejected(mutation):
    instruction = "Choose the object with the largest density."
    config = config_for(("stone", "metal", "wood"), instruction)
    if mutation == "overlap":
        config["task"]["components"][4]["position"] = [0, .1, 0]
    elif mutation == "unknown_material":
        config["task"]["components"][2]["material"] = "unknown"
    elif mutation == "conflicting_instruction":
        config["task"]["instructions"] = ["Choose the object with the smallest density."]
    else:
        instruction = "Choose the object that can float in the water."
        config = config_for(("wood", "rubber", "metal"), instruction)
    with pytest.raises(ValueError):
        scene_answer(config, instruction)


def test_generator_patch_is_narrow_and_idempotent():
    source = f'DENSITY_ORDER = {list(DENSITY_ORDER)!r}\n' + '''
def load_objects():
    for i, (button_config, material, geom_type) in enumerate(options):
            if i == self.target_material_order_idx:
                self.target_button = button_config["name"]
'''
    repaired = generator_repair(source)
    assert repaired == source.replace("if i == self.target_material_order_idx:", "if material == target_material:")
    assert generator_repair(repaired) == repaired
    with pytest.raises(ValueError, match="unrecognized"):
        generator_repair("unknown source")


def test_restore_roundtrip_is_exact_and_idempotent(tmp_path):
    root = tmp_path / "data"
    episode = write_episode(root)
    originals = {p: p.read_bytes() for p in episode.rglob("*") if p.is_file()}
    backup = tmp_path / "backup"
    migrate(root, backup_dir=backup, apply=True)
    assert restore(root, backup)["changed_files"] == 2
    assert json.loads((episode / "output/operation_sequence.json").read_text())["skill_sequence"][0]["params"]["target_entity_name"] == 3
    assert restore(root, backup, apply=True)["changed_files"] == 2
    assert all(p.read_bytes() == raw for p, raw in originals.items())
    assert restore(root, backup, apply=True)["changed_files"] == 0


@pytest.mark.parametrize("corrupt_backup", [False, True])
def test_restore_preflights_entire_batch_without_overwriting_edits(tmp_path, corrupt_backup):
    root = tmp_path / "data"
    episode = write_episode(root)
    backup = tmp_path / "backup"
    migrate(root, backup_dir=backup, apply=True)
    manifest = json.loads((backup / "manifest.json").read_text())
    if corrupt_backup:
        (backup / manifest["files"][-1]["backup"]).write_bytes(b"corrupt")
    else:
        (episode / "output/operation_sequence.json").write_bytes(b"independent edit")
    current = {p: p.read_bytes() for p in episode.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        restore(root, backup, apply=True)
    assert all(p.read_bytes() == raw for p, raw in current.items())


@pytest.mark.parametrize("mode", ["visual", "metadata-assisted"])
def test_in_memory_labels_preserve_inputs_and_leave_disk_original(tmp_path, mode):
    from ..dataset import load_planning_examples
    episode = write_episode(tmp_path)
    before = {p: p.read_bytes() for p in episode.rglob("*") if p.is_file()}
    examples = load_planning_examples(tmp_path, observation_mode=mode)
    updated = updated_density_examples(examples)
    assert examples[0].operation_sequence[0]["params"]["target_entity_name"] == 5
    assert updated[0].operation_sequence[0]["params"]["target_entity_name"] == 3
    from dataclasses import replace
    assert replace(updated[0], operation_sequence=examples[0].operation_sequence) == examples[0]
    assert all(p.read_bytes() == raw for p, raw in before.items())
