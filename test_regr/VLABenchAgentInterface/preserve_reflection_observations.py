"""Preserve explicitly bound reflectance in future VLABench scene exports.

This patches upstream source with backups. It cannot recover observations that
were omitted from existing snapshots and never changes saved reference labels.
"""
import argparse
import ast
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path


REVISION = "reflection-observation-export-v1"
GENERATOR = "VLABench/tasks/hierarchical_tasks/primitive/physical_series/reflection_qa.py"
ENTITY = "VLABench/tasks/components/specific_entities/common_containers.py"


def patch_generator(source):
    before = 'reflectance=candidate_reflections)'
    after = 'reflectance=candidate_reflections, option_buttons=[button["name"] for button in button_configs])'
    if after in source and before not in source:
        return source
    if source.count(before) != 1:
        raise ValueError("unrecognized reflection generator")
    result = source.replace(before, after)
    ast.parse(result)
    return result


def patch_entity(source):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Mirrors")
    lines = source.splitlines(keepends=True)
    block = ''.join(lines[node.lineno - 1:node.end_lineno])
    if '# reflection-observation-export-v1' in block:
        if 'def save(self, physics)' not in block or 'option_buttons=None' not in block:
            raise ValueError("incomplete reflection export patch")
        return source
    if 'def save(' in block or block.count('               reflectance:list,') != 1:
        raise ValueError("unrecognized Mirrors implementation")
    block = block.replace('               reflectance:list,', '               reflectance:list,\n               option_buttons=None,')
    anchor = '        super()._build(**kwargs)'
    if block.count(anchor) != 1:
        raise ValueError("unrecognized Mirrors build method")
    block = block.replace(anchor, '''        # reflection-observation-export-v1
        self.reflectance = list(reflectance)
        self.option_buttons = None if option_buttons is None else list(option_buttons)
        if self.option_buttons is not None:
            if len(self.option_buttons) != len(self.reflectance) or len(set(self.option_buttons)) != len(self.option_buttons):
                raise ValueError("reflectance needs a unique explicit button binding per value")
        super()._build(**kwargs)''')
    block = block.rstrip() + '''

    def save(self, physics):
        info = super().save(physics)
        info["reflectance"] = [float(self.mjcf_model.find("material", f"reflectance_{i}").reflectance)
                               for i in range(len(self.reflectance))]
        if self.option_buttons is not None:
            info["option_buttons"] = list(self.option_buttons)
        return info
'''
    result = ''.join(lines[:node.lineno - 1]) + block + ''.join(lines[node.end_lineno:])
    ast.parse(result)
    return result


def preserve(upstream_root, *, apply=False, backup_dir=None):
    root = Path(upstream_root).resolve()
    changes = []
    for relative, patch in ((GENERATOR, patch_generator), (ENTITY, patch_entity)):
        path = root / relative
        original = path.read_bytes()
        modified = patch(original.decode()).encode()
        if modified != original:
            changes.append((path, original, modified))
    report = {"revision": REVISION, "created_utc": datetime.now(timezone.utc).isoformat(),
              "upstream_root": str(root), "applied": False, "files": [],
              "existing_episode_labels_modified": False}
    for i, (path, before, after) in enumerate(changes):
        report["files"].append({"path": str(path), "backup": f"{i:02d}.original",
                                "before_sha256": hashlib.sha256(before).hexdigest(),
                                "after_sha256": hashlib.sha256(after).hexdigest(),
                                "patch": ''.join(difflib.unified_diff(before.decode().splitlines(True),
                                                                     after.decode().splitlines(True),
                                                                     fromfile=str(path), tofile=str(path)))})
    if not apply:
        return report
    if backup_dir is None:
        raise ValueError("--backup-dir is required with --apply")
    backup = Path(backup_dir).resolve()
    if backup == root or root in backup.parents:
        raise ValueError("backup must be outside the upstream source tree")
    backup.mkdir(parents=True, exist_ok=False)
    for entry, (_, before, _) in zip(report["files"], changes):
        (backup / entry["backup"]).write_bytes(before)
    manifest = backup / 'manifest.json'
    manifest.write_text(json.dumps(report, indent=2) + '\n')
    try:
        for path, before, after in changes:
            if path.read_bytes() != before:
                raise ValueError(f"source changed during patch: {path}")
            temporary = path.with_name(path.name + '.reflection-export.tmp')
            temporary.write_bytes(after)
            temporary.replace(path)
    except Exception:
        for path, before, after in changes:
            if path.read_bytes() == after:
                path.write_bytes(before)
        raise
    report['applied'] = True
    manifest.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-root', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--backup-dir', type=Path)
    print(json.dumps(preserve(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()
