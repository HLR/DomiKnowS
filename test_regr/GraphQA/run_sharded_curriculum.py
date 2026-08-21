"""Run memory-bounded C2-C5 training as sequential 1K graph shards."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-stage", type=int, default=2)
    parser.add_argument("--end-stage", type=int, default=5)
    parser.add_argument("--shards", type=int, default=10)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--epochs-per-shard", type=int, default=1)
    parser.add_argument("--dev-limit", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--initial-predicate-dir", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    task_dir = args.root / "data/dataset/task_list"
    kb_dir = args.root / "data/knowledge_base"
    image_cache = args.root / "image_cache"
    gqa_info = args.root / "data/gqa_info.json"
    model_dir = args.output_root / "models/sharded_c2_c5"
    log_dir = args.output_root / "logs/sharded_c2_c5"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    previous = args.initial_checkpoint
    for stage in range(args.start_stage, args.end_stage + 1):
        for shard in range(args.shards):
            offset = shard * args.shard_size
            output = model_dir / (
                f"graphqa_c{stage}_shard{shard:02d}_"
                f"offset{offset}_n{args.shard_size}.pt"
            )
            if output.exists():
                print(f"skip_existing={output}", flush=True)
                previous = output
                continue

            command = [
                sys.executable, "-m", "test_regr.GraphQA.train_scallop_mlp",
                "--task-path", str(task_dir / f"train_tasks_c{stage}_10000.pkl"),
                "--dev-task-path", str(task_dir / f"val_tasks_c{stage}_1000.pkl"),
                "--kb-dir", str(kb_dir),
                "--image-cache", str(image_cache),
                "--gqa-info", str(gqa_info),
                "--limit", str(args.shard_size),
                "--offset", str(offset),
                "--dev-limit", str(args.dev_limit),
                "--epochs", str(args.epochs_per_shard),
                "--batch-size", str(args.batch_size),
                "--lr", str(args.lr),
                "--hidden-dim", "1024",
                "--scheduler-step", "10",
                "--scheduler-gamma", "0.1",
                "--prediction-threshold", "0.5",
                "--decode-policy", "threshold",
                "--device", args.device,
                "--output", str(output),
            ]
            if previous is None:
                command.extend([
                    "--init-predicate-dir", str(args.initial_predicate_dir)
                ])
            else:
                command.extend(["--init-checkpoint", str(previous)])
                if shard > 0:
                    command.append("--resume-optimizer")

            log_path = log_dir / f"c{stage}_shard{shard:02d}.log"
            print(
                f"stage={stage} shard={shard} offset={offset} "
                f"checkpoint={previous} output={output}",
                flush=True,
            )
            with log_path.open("w") as stream:
                subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[2],
                    env=os.environ.copy(),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            previous = output

    print(f"final_checkpoint={previous}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
