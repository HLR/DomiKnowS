"""Sequential GPU5 experiment driver with durable stage logs and status."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..planning_experiment import write_json, digest


def run(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    module = "test_regr.VLABenchAgentInterface.planning_experiment"
    def command(stage, arguments):
        log = root / f"{stage}.log"
        started = time.time()
        with log.open("a", buffering=1) as stream:
            child = subprocess.Popen([sys.executable, "-u", "-m", module, *map(str, arguments)], stdout=stream, stderr=subprocess.STDOUT)
            write_json(root / "status.json", {"stage": stage, "driver_pid": os.getpid(), "child_pid": child.pid,
                "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"), "log": str(log), "started_at_unix": started})
            print(json.dumps({"stage": stage, "pid": child.pid, "log": str(log)}), flush=True)
            code = child.wait()
        if code:
            write_json(root / "status.json", {"stage": stage, "status": "failed", "exit_code": code,
                "driver_pid": os.getpid(), "child_pid": child.pid, "log": str(log)})
            raise RuntimeError(f"stage {stage} failed; see {log}")
    manifests = root / "manifests"
    if not (manifests / "dataset_inventory.json").exists():
        command("inventory", ["inventory", "--planning-dir", args.planning_dir, "--output", manifests])
    def evaluate(stage, mode, split="test", **options):
        target = root / stage
        command(stage, ["evaluate", "--planning-dir", args.planning_dir, "--model", args.model,
            "--manifest", manifests / f"{split}_manifest.json", "--output", target, "--mode", mode,
            *[value for key, item in options.items() for value in ["--" + key.replace("_", "-"), item]]])
    # Smoke records are separate from reported evaluations.
    for stage, mode, options in [
        ("smoke_raw", "raw", {}), ("smoke_dfa", "dfa", {}),
        ("smoke_supervised", "graph", {"checkpoint": args.supervised}),
        ("smoke_rl", "graph", {"checkpoint": args.reinforcement}),
    ]:
        evaluate(stage, mode, limit=1, **options)
    if args.smoke_only:
        write_json(root / "status.json", {"stage": "smoke", "status": "complete", "driver_pid": os.getpid()})
        return
    evaluate("raw", "raw")
    evaluate("dfa", "dfa")
    evaluate("supervised", "graph", checkpoint=args.supervised)
    evaluate("reinforcement", "graph", checkpoint=args.reinforcement)
    evaluate("corpus_train", "corpus", split="train")
    evaluate("corpus_validation", "corpus", split="validation")
    command("hmm_fit", ["fit-hmm", "--corpus", root / "corpus_train", "--validation-corpus", root / "corpus_validation", "--output", root / "hmm_model"])
    checkpoint = root / "hmm_model/best.pt"
    evaluate("smoke_hmm", "hmm", hmm=checkpoint, weight=1., limit=1)
    selection_path = root / "hmm_selection.json"
    if not selection_path.exists():
        candidates = []
        for weight in (.1, .5, 1.):
            stage = f"hmm_validation_{weight}"
            evaluate(stage, "hmm", split="validation", hmm=checkpoint, weight=weight)
            summary = json.loads((root / stage / "summary.json").read_text())["planning_metrics"]
            score = summary["Macro"] if summary["Macro"] is not None else summary["overall"]["score"]
            candidates.append({"weight": weight, "selection_score": score})
        selected = max(candidates, key=lambda item: (item["selection_score"], -item["weight"]))
        write_json(selection_path, {"candidates": candidates, "selected": selected, "hmm_sha256": digest(checkpoint), "selection_split": "validation"})
    selection = json.loads(selection_path.read_text())
    if selection["hmm_sha256"] != digest(checkpoint):
        raise ValueError("HMM checkpoint changed after validation selection")
    evaluate("hmm", "hmm", hmm=checkpoint, weight=selection["selected"]["weight"])
    write_json(root / "status.json", {"stage": "all", "status": "complete", "driver_pid": os.getpid()})


def main():
    parser = argparse.ArgumentParser()
    for name in ("output", "model", "planning-dir", "supervised", "reinforcement"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
