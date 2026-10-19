"""Audit/repair density QA labels from instructions and scene components.

No reference target participates in selecting the answer. This is an offline
dataset migration, not a decoder input adapter. Run without --apply to audit.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from dataclasses import replace


# Public qualitative model from VLABench's density_qa.py, not measured density.
DENSITY_ORDER = ("wood", "rubber", "water", "glass", "stone", "metal")
REPAIR_REVISION = "density-material-button-v1"


def scene_answer(config, instruction):
    """Return a uniquely grounded button and its component/entity index."""
    components = config["task"]["components"]
    buttons = [c for c in components if c.get("class") == "Button"]
    objects = [c for c in components if c.get("class") == "RandomGeom"]
    if len(buttons) != 3 or len(objects) != 3:
        raise ValueError("expected three buttons and three material objects")
    names = [c["name"] for c in components]
    if len(names) != len(set(names)):
        raise ValueError("duplicate component names")
    pairs = {}
    for button in buttons:
        distances = sorted((math.dist(button["position"][:2], obj["position"][:2]), i)
                           for i, obj in enumerate(objects))
        nearest, index = distances[0]
        if not math.isfinite(nearest) or nearest > 0.25 or distances[1][0] <= nearest * 2:
            raise ValueError("ambiguous spatial button/object association")
        pairs[button["name"]] = objects[index]
    if len({obj["name"] for obj in pairs.values()}) != 3:
        raise ValueError("button/object association is not one-to-one")
    materials = [obj["material"] for obj in pairs.values()]
    if len(set(materials)) != 3 or any(m not in DENSITY_ORDER for m in materials):
        raise ValueError("expected distinct materials with known density ordering")

    def choose(text):
        text = " ".join(text.lower().split())
        ordered = sorted(pairs, key=lambda b: DENSITY_ORDER.index(pairs[b]["material"]))
        if "float in the water" in text:
            candidates = [b for b in ordered if DENSITY_ORDER.index(pairs[b]["material"]) < DENSITY_ORDER.index("water")]
        else:
            ranks = [i for i, word in enumerate(("smallest density", "medium density", "largest density")) if word in text]
            candidates = [ordered[ranks[0]]] if len(ranks) == 1 else []
        if len(candidates) != 1:
            raise ValueError("instruction does not select a unique material")
        return candidates[0]

    answer = choose(instruction)
    if any(choose(text) != answer for text in config["task"]["instructions"]):
        raise ValueError("input and environment instructions disagree")
    return answer, names.index(answer), {b: obj["material"] for b, obj in pairs.items()}


def generator_repair(source):
    """Patch the known upstream defect, refusing unknown source variants."""
    bad = "            if i == self.target_material_order_idx:"
    good = "            if material == target_material:"
    if source.count(bad) == 1 and good not in source:
        repaired = source.replace(bad, good)
    elif source.count(good) == 1 and bad not in source:
        repaired = source
    else:
        raise ValueError("unrecognized density generator source")
    tree = ast.parse(repaired)
    order = next(ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DENSITY_ORDER" for t in n.targets))
    if tuple(order) != DENSITY_ORDER:
        raise ValueError("generator density model differs from migration model")
    return repaired


def episode_repair(episode):
    config_path = episode / "env_config" / "env_config.json"
    output_path = episode / "output" / "operation_sequence.json"
    original_config, original_output = config_path.read_bytes(), output_path.read_bytes()
    config, output = json.loads(original_config), json.loads(original_output)
    answer, index, materials = scene_answer(config, (episode / "input" / "instruction.txt").read_text())
    # Only consult old labels after the scene-derived answer has been fixed.
    condition = config["task"]["conditions"]["press_button"]
    old = condition["target_button"]
    sequence = output["skill_sequence"]
    if len(sequence) != 1 or sequence[0]["name"] != "press":
        raise ValueError("expected a single press operation")
    params = sequence[0]["params"]
    target = params["target_entity_name"]
    names = [c["name"] for c in config["task"]["components"]]
    if type(target) is int:
        if not 0 <= target < len(names) or names[target] != old:
            raise ValueError("numeric reference does not agree with component index and condition")
        replacement = index
    elif isinstance(target, str) and target == old:
        replacement = answer
    else:
        raise ValueError("unsupported or inconsistent reference entity encoding")
    condition["target_button"] = answer
    params["target_entity_name"] = replacement
    changes = []
    if old != answer:
        for path, original, value in ((config_path, original_config, config), (output_path, original_output, output)):
            changes.append((path, original, (json.dumps(value, indent=4) + "\n").encode()))
    return {"episode": str(episode), "old_button": old, "new_button": answer,
            "materials": materials, "changed": old != answer}, changes


def migrate(planning_dir, generator=None, backup_dir=None, apply=False):
    root = Path(planning_dir).resolve()
    episodes = sorted(p.parent.parent for p in root.rglob("input/instruction.txt")
                      if p.parent.parent.parent.name == "density_qa")
    if not episodes:
        raise ValueError("no density_qa episodes found")
    rows, changes, errors = [], [], []
    for episode in episodes:
        try:
            row, updates = episode_repair(episode)
            rows.append(row)
            changes.extend(updates)
        except (ValueError, KeyError, IndexError, TypeError, OSError) as exc:
            errors.append({"episode": str(episode), "error": str(exc)})
    if generator:
        path = Path(generator).resolve()
        before = path.read_bytes()
        after = generator_repair(before.decode()).encode()
        if before != after:
            changes.append((path, before, after))
    report = {"revision": REPAIR_REVISION, "created_utc": datetime.now(timezone.utc).isoformat(),
              "planning_dir": str(root), "episodes": len(episodes),
              "changed_episodes": sum(row["changed"] for row in rows),
              "errors": errors, "rows": rows, "applied": False, "files": []}
    if not apply:
        return report
    if errors:
        raise ValueError(f"refusing partial repair: {errors}")
    if not backup_dir:
        raise ValueError("--backup-dir is required with --apply")
    backup = Path(backup_dir).resolve()
    if backup == root or root in backup.parents:
        raise ValueError("backup must be outside the dataset tree")
    backup.mkdir(parents=True, exist_ok=False)
    # Back up every original and manifest before the first dataset/source write.
    for i, (path, before, after) in enumerate(changes):
        backup_name = f"{i:04d}.original"
        (backup / backup_name).write_bytes(before)
        report["files"].append({"path": str(path), "backup": backup_name,
                                "before_sha256": hashlib.sha256(before).hexdigest(),
                                "after_sha256": hashlib.sha256(after).hexdigest()})
    manifest = backup / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for path, before, after in changes:
            if path.read_bytes() != before:
                raise ValueError(f"file changed during repair: {path}")
            temporary = path.with_name(path.name + ".density-repair.tmp")
            temporary.write_bytes(after)
            temporary.replace(path)
    except Exception:
        # Restore only files written by this invocation, preserving external edits.
        for path, before, after in changes:
            if path.read_bytes() == after:
                path.write_bytes(before)
        raise
    report["applied"] = True
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    return report


def restore(planning_dir, backup_dir, apply=False):
    """Restore dataset bytes from a migration manifest, retaining generator fixes.

    Validate the complete batch before writing; unrelated edits are never replaced.
    The supplied planning directory can differ from the original migration root.
    """
    root, backup = Path(planning_dir).resolve(), Path(backup_dir).resolve()
    manifest = json.loads((backup / "manifest.json").read_text())
    source = Path(manifest["planning_dir"])
    changes, seen = [], set()
    for entry in manifest["files"]:
        try:
            relative = Path(entry["path"]).relative_to(source)
        except ValueError:
            continue  # The upstream generator is not part of the dataset.
        target = root / relative
        if not target.resolve().is_relative_to(root):
            raise ValueError(f"restore target escapes dataset: {target}")
        if target in seen:
            raise ValueError(f"duplicate restore target: {target}")
        seen.add(target)
        saved = (backup / entry["backup"]).resolve()
        if not saved.is_relative_to(backup):
            raise ValueError("backup path escapes backup directory")
        original, current = saved.read_bytes(), target.read_bytes()
        digest = lambda data: hashlib.sha256(data).hexdigest()
        if digest(original) != entry["before_sha256"]:
            raise ValueError(f"backup checksum mismatch: {saved}")
        if digest(current) == entry["before_sha256"]:
            continue
        if digest(current) != entry["after_sha256"]:
            raise ValueError(f"file changed since repair; refusing restore: {target}")
        changes.append((target, current, original))
    if not seen:
        raise ValueError("manifest contains no dataset files")
    written = []
    if apply:
        try:
            for target, current, original in changes:
                if target.read_bytes() != current:
                    raise ValueError(f"file changed during restore: {target}")
                temporary = target.with_name(target.name + ".density-restore.tmp")
                temporary.write_bytes(original)
                temporary.replace(target)
                written.append((target, current, original))
        except Exception:
            for target, current, original in written:
                if target.read_bytes() == original:
                    target.write_bytes(current)
            raise
    return {"mode": "restore-original", "planning_dir": str(root),
            "checked_files": len(seen), "changed_files": len(changes),
            "applied": apply, "errors": []}


def updated_density_examples(examples):
    """Correct evaluation references in memory; never alter inputs or disk data."""
    from .graph import canonicalize_plan
    result = []
    for example in examples:
        if example.task != "density_qa":
            result.append(example)
            continue
        if example.env_config_path is None:
            raise ValueError("density label update requires a scene configuration")
        _, changes = episode_repair(Path(example.env_config_path).parent.parent)
        updated = example
        for path, _, data in changes:
            if path.parent.name == "output":
                payload = json.loads(data)
                updated = replace(example, operation_sequence=tuple(canonicalize_plan(payload["skill_sequence"])))
        result.append(updated)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-dir", type=Path, required=True)
    parser.add_argument("--generator", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--restore", action="store_true",
                        help="restore original dataset bytes using --backup-dir; requires --apply to write")
    args = parser.parse_args()
    if args.restore:
        if not args.backup_dir or args.generator:
            parser.error("--restore requires --backup-dir and cannot use --generator")
        report = restore(args.planning_dir, args.backup_dir, args.apply)
    else:
        options = vars(args).copy()
        options.pop("restore")
        report = migrate(**options)
    print(json.dumps(report, indent=2))
    return bool(report["errors"])


if __name__ == "__main__":
    raise SystemExit(main())
