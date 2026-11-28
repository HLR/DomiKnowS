"""Freeze independent native-sampled LIBERO scenes before any model scoring."""
import argparse
import importlib
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image

from ..anonymous_grounding.planner_binding.freeze_external import carry_calibration, scene
from ..protocol import digest, episodes, freeze, read, sha


def make_manifest(config, protocol, output, *, namespace="spatial-transfer-v1", exclusions=()):
    from libero.libero import benchmark
    from libero.libero.envs import SegmentationRenderEnv

    base = Path(config["libero_base"])
    bridge_path = base / "evaluation_source/test_regr/JointEmbodiedAgentInterface/generalization/libero_campaign.py"
    sys.path.insert(0, str(bridge_path.parent))
    bridge = importlib.import_module("libero_campaign")
    old_path = Path(protocol["external_manifest"])
    calibration_path = Path(protocol["calibration"])
    if any(sha(p) != protocol["hashes"][str(p)] for p in (old_path, calibration_path)):
        raise ValueError("archived LIBERO manifest or calibration changed")
    archived = read(old_path)
    old = episodes(archived)
    cases = {(row["suite"], row["task"]): row for row in old}
    with Image.open(read(old[0]["generation_input"])["images"][0]) as pic:
        width, height = pic.size
    from ..prepare_data import fingerprints

    history = [*archived["episodes"], *(row for path in exclusions for row in read(path)["episodes"])]
    history_images = {row["files"][str(p)] for row in history
                      for p in read(row["generation_input"])["images"]}
    history_fingerprints = {fingerprints(row, "libero")["scene_fingerprint"] for row in history}
    records, audit, seen_scenes = [], [], set()
    for (suite_name, task_name), record in sorted(cases.items()):
        if sha(record["bddl"]) != record["source_bddl_sha256"]:
            raise ValueError("native task source differs")
        suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
        index = next(i for i in range(suite.n_tasks) if suite.get_task(i).name == task_name)
        states = suite.get_task_init_states(index)
        if len(states) != 50:
            raise ValueError("complete official-state inventory required")
        spec = bridge.specification(record["bddl"])
        env = SegmentationRenderEnv(bddl_file_name=record["bddl"], camera_heights=height,
                                    camera_widths=width, horizon=1000, control_freq=20, use_camera_obs=True)
        try:
            env.reset()
            official = set()
            for state in states:
                env.sim.set_state_from_flattened(np.asarray(state))
                env.sim.forward()
                official.add(scene(env, spec))
            for i in range(5):
                label = "native-sample" if namespace == "spatial-transfer-v1" else namespace
                identity = f"{suite_name}/{task_name}/{label}-{i:02}"
                seed = int(digest([namespace, identity])[:8], 16)
                random.seed(seed)
                np.random.seed(seed)
                env.seed(seed)
                obs = env.reset()
                initial = scene(env, spec)
                for _ in range(10):
                    obs, _, _, _ = env.step(np.array([0., 0., 0., 0., 0., 0., -1.]))
                settled = scene(env, spec)
                if {initial, settled} & (official | seen_scenes) or env.check_success():
                    raise ValueError("new native reset overlaps a known state or starts successful: " + identity)
                seen_scenes.update((initial, settled))
                folder = output / "evaluation" / identity
                folder.mkdir(parents=True, exist_ok=False)
                state_path = folder / "initial_state.npy"
                np.save(state_path, env.get_sim_state(), allow_pickle=False)
                raw, numbered, visible = bridge.numbered_image(obs, env, spec["entities"])
                paths = [folder / "raw.png", folder / "numbered.png"]
                Image.fromarray(raw).save(paths[0])
                numbered.save(paths[1])
                if history_images.intersection(map(sha, paths)):
                    raise ValueError("new initial image duplicates an archived reset")
                inputs = bridge.generation_input(spec, paths)
                inputs["visible_entity_ids"] = visible
                public_path, private_path = folder / "generation_input.json", folder / "scoring_private.json"
                freeze(public_path, inputs)
                freeze(private_path, spec)
                fresh = {**record, "episode_id": identity, "state_index": 50 + i,
                         "state_origin": "fresh seeded native random reset",
                         "sampling_seed": seed, "state": str(state_path),
                         "generation_input": str(public_path), "scoring_private": str(private_path),
                         "files": {str(p): sha(p) for p in (*paths, state_path, public_path, private_path)}}
                if fingerprints(fresh, "libero")["scene_fingerprint"] in history_fingerprints:
                    raise ValueError("sampled native scene overlaps previous work: " + identity)
                freeze(folder / "record.json", fresh)
                records.append(fresh)
                audit.append({"episode_id": identity, "initial_scene_sha256": initial,
                              "settled_scene_sha256": settled, "official_scene_count": len(official)})
                print(f"[spatial-confirmation-freeze] {len(records)}/100", flush=True)
        finally:
            env.close()
    manifest = {**archived,
                "episodes": records + [e for e in archived["episodes"] if e["split"] == "calibration"],
                "protocol": "100 seeded native resets, separate from the 50 official states per task"}
    episodes(manifest)
    path = output / "manifest.json"
    freeze(path, manifest)
    freeze(output / "split_audit.json", {"passed": True, "historical_manifest_sha256": sha(old_path),
                                         "official_states_per_task": 50, "scenes": audit, "seed_namespace": namespace,
                                         "additional_exclusions": {str(path): sha(path) for path in exclusions}})
    freeze(output / "calibration_reuse.json", carry_calibration(
        archived, manifest, read(calibration_path), old_path, path, bridge_path
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage2", action="store_true", help="Freeze the separate Stage 2 holdout")
    parser.add_argument("--exclude", action="append", type=Path, default=[])
    args = parser.parse_args()
    if args.stage2 and not args.exclude:
        parser.error("the Stage 2 holdout requires prior scored manifests")
    make_manifest(read(args.archive / "config.json"), read(args.archive / "protocol.json"), args.output,
                  namespace="source-stage2-holdout-v1" if args.stage2 else "spatial-transfer-v1",
                  exclusions=args.exclude)


if __name__ == "__main__":
    main()
