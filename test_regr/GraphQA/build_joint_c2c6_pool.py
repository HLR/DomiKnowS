#!/usr/bin/env python3
"""Build one shuffled, interleaved train/val pool combining C2-C6.

Concatenates the five per-split 10k train pickles (up to --per-split each)
and the five per-split val pickles (up to --val-per-split each), shuffles
each pool once with a fixed seed, and writes two combined pickles. Used to
give the MLP joint/balanced exposure across all complexity splits in one
continuous run, instead of the current sequential per-split curriculum.

Does not touch any live checkpoint or process -- pure data prep, safe to
run alongside anything else.
"""
from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["c2", "c3", "c4", "c5", "c6"])
    parser.add_argument("--per-split", type=int, default=10000,
                         help="Max train instances to draw from each split.")
    parser.add_argument("--val-per-split", type=int, default=300,
                         help="Max val instances to draw from each split.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-train", type=Path, required=True)
    parser.add_argument("--output-val", type=Path, required=True)
    return parser.parse_args()


def load_and_tag(path, limit, split):
    with open(path, "rb") as handle:
        tasks = list(pickle.load(handle))
    tasks = tasks[:limit]
    for task in tasks:
        if isinstance(task, dict):
            task.setdefault("_source_split", split)
    return tasks


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    train_pool, val_pool = [], []
    per_split_counts = {}
    for split in args.splits:
        train_path = args.task_dir / f"train_tasks_{split}_10000.pkl"
        val_path = args.task_dir / f"val_tasks_{split}_1000.pkl"
        if not train_path.exists():
            raise FileNotFoundError(f"missing train pickle for split {split}: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"missing val pickle for split {split}: {val_path}")
        train_tasks = load_and_tag(train_path, args.per_split, split)
        val_tasks = load_and_tag(val_path, args.val_per_split, split)
        train_pool.extend(train_tasks)
        val_pool.extend(val_tasks)
        per_split_counts[split] = {"train": len(train_tasks), "val": len(val_tasks)}

    rng.shuffle(train_pool)
    rng.shuffle(val_pool)

    args.output_train.parent.mkdir(parents=True, exist_ok=True)
    args.output_val.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_train, "wb") as handle:
        pickle.dump(train_pool, handle)
    with open(args.output_val, "wb") as handle:
        pickle.dump(val_pool, handle)

    print({
        "status": "ok",
        "seed": args.seed,
        "per_split_counts": per_split_counts,
        "train_pool_size": len(train_pool),
        "val_pool_size": len(val_pool),
        "output_train": str(args.output_train),
        "output_val": str(args.output_val),
    })


if __name__ == "__main__":
    main()
