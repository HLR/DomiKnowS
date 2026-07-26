"""Create balanced MATRES-style temporal relation subsets.

The output keeps the existing six-column MATRES text format:

    docid, verb1, verb2, eiid1, eiid2, relation

An optional JSONL audit file records the deterministic DomiKnowS executable
query generated for each selected row, so the question-to-program conversion
can be inspected independently of model training.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from .dataset import load_temporal_instances
from .execution import create_executable_instance
from .graph import TEMPORAL_LABELS


def _read_matres_rows(path: Path):
    rows = []
    with path.open("r", newline="") as data_file:
        reader = csv.reader(data_file, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) != 6:
                raise ValueError(f"Expected 6 tab-separated fields in {path}:{line_number}, got {len(row)}")
            rows.append(row)
    return rows


def _write_matres_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as data_file:
        writer = csv.writer(data_file, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)


def _write_audit(path: Path, output_path: Path):
    instances = load_temporal_instances(output_path, group_by_document=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as audit_file:
        for index, instance in enumerate(instances):
            converted = create_executable_instance(instance)
            audit_file.write(
                json.dumps(
                    {
                        "index": index,
                        "doc_id": instance.get("doc_id"),
                        "query_pair": instance.get("query_pair"),
                        "label": instance["event_pairs"][0]["label"],
                        "logic_str": converted["logic_str"],
                        "logic_label": int(converted["logic_label"]),
                        "pair_learner_examples": converted["pair_learner_examples"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def build_balanced_subset(input_paths, output_path, audit_path=None, total=1000, seed=13, labels=None):
    labels = labels or list(TEMPORAL_LABELS)
    if total % len(labels) != 0:
        raise ValueError(f"--total must be divisible by the number of labels ({len(labels)})")
    per_label = total // len(labels)

    buckets = defaultdict(list)
    for input_path in input_paths:
        for row in _read_matres_rows(Path(input_path)):
            label = row[5].strip()
            normalized = label.upper()
            if normalized == "SIMULTANEOUS":
                normalized = "EQUAL"
            canonical = next((item for item in labels if item.upper() == normalized), None)
            if canonical is not None:
                row = list(row)
                row[5] = canonical
                buckets[canonical].append(row)

    rng = random.Random(seed)
    selected = []
    summary = {}
    for label in labels:
        rows = list(buckets[label])
        rng.shuffle(rows)
        if len(rows) < per_label:
            raise ValueError(f"Label {label} only has {len(rows)} rows, need {per_label}")
        chosen = rows[:per_label]
        selected.extend(chosen)
        summary[label] = len(chosen)
    rng.shuffle(selected)

    _write_matres_rows(output_path, selected)
    if audit_path is not None:
        _write_audit(audit_path, output_path)
    return {"output": str(output_path), "total": len(selected), "per_label": summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True, help="MATRES-style .txt input. Repeatable.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-jsonl", type=Path, default=None)
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    summary = build_balanced_subset(
        input_paths=args.input,
        output_path=args.output,
        audit_path=args.audit_jsonl,
        total=args.total,
        seed=args.seed,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
