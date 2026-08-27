#!/usr/bin/env python3
"""Evaluate one validation-selected Scallop MLP on GraphQA C2--C6.

Reports Scallop-compatible Recall@5 plus Top-1, Recall@10, and separate
predicate-forward/local-execution/ILP timing.  Local and ILP decoding reuse
exactly the same cached atomic probabilities for each example.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import torch
from tqdm import tqdm

from test_regr.GraphQA.dataset import load_kb_facts, load_vqar_tasks, vqar_task_to_graphqa_instance
from test_regr.GraphQA.graph import canonical_relation
from test_regr.GraphQA.scallop_style_qwen_executor import evaluate_instance, evaluate_instance_ilp, index_kb


SCALLOP_RECALL5 = {"C2": 0.8517, "C3": 0.8282, "C4": 0.8325, "C5": 0.8553, "C6": 0.8430}


class ObjectMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, hidden_layers):
        super().__init__()
        layers, width = [], int(input_dim)
        for _ in range(int(hidden_layers)):
            layers += [torch.nn.Linear(width, hidden_dim), torch.nn.ReLU(),
                       torch.nn.BatchNorm1d(hidden_dim), torch.nn.Dropout(0.3)]
            width = hidden_dim
        layers.append(torch.nn.Linear(width, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())


class RelationMLP(torch.nn.Module):
    def __init__(self, feature_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim * 2 + 8, hidden_dim), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_dim), torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x.float())


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


class MLPScorer:
    def __init__(self, checkpoint, device):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("architecture_version") != "vqar_scallop_5_2":
            raise ValueError("Not a vqar_scallop_5_2 checkpoint")
        self.device = torch.device(device)
        self.feature_dim = int(state["feature_dim"])
        hidden = int(state["hidden_dim"])
        counts = state["num_classes"]
        self.indices = state["indices"]
        self.name = ObjectMLP(self.feature_dim, int(counts["name"]), hidden, 2).to(self.device)
        self.attr = ObjectMLP(self.feature_dim, int(counts["attribute"]), hidden, 1).to(self.device)
        self.rel = RelationMLP(self.feature_dim, int(counts["relation"]) + 1, hidden).to(self.device)
        self.name.load_state_dict(state["name_mlp"])
        self.attr.load_state_dict(state["attr_mlp"])
        self.rel.load_state_dict(state["relation_mlp"])
        self.name.eval(); self.attr.eval(); self.rel.eval()
        self.max_candidate_symbols = 128

    def bind(self, instance):
        return BoundMLPScorer(self, instance)


class BoundMLPScorer:
    def __init__(self, base, instance):
        self.base = base
        self.max_candidate_symbols = base.max_candidate_symbols
        self.objects = [str(x) for x in instance.get("objects", [])]
        metadata = instance.get("object_metadata") or {}
        features, boxes = [], []
        for obj in self.objects:
            item = metadata.get(obj, {})
            row = [float(x) for x in (item.get("feature_vector") or [])]
            row = (row + [0.0] * base.feature_dim)[:base.feature_dim]
            box = [float(x) for x in (item.get("bbox") or [0, 0, 0, 0])]
            features.append(row)
            boxes.append((box + [0.0] * 4)[:4])
        x = torch.tensor(features, dtype=torch.float32, device=base.device)
        b = torch.tensor(boxes, dtype=torch.float32, device=base.device)
        _sync(base.device)
        start = time.perf_counter()
        with torch.inference_mode():
            self.name_probs = torch.softmax(base.name(x), dim=-1).cpu()
            self.attr_probs = torch.sigmoid(base.attr(x)).cpu()
            pair_rows, pair_keys = [], []
            for i, src in enumerate(self.objects):
                for j, dst in enumerate(self.objects):
                    if i != j:
                        pair_rows.append(torch.cat((x[i], x[j], b[i], b[j])))
                        pair_keys.append((src, dst))
            if pair_rows:
                rel_logits = base.rel(torch.stack(pair_rows))
                self.rel_probs = torch.softmax(rel_logits, dim=-1).cpu()
            else:
                self.rel_probs = torch.empty((0, 1))
        _sync(base.device)
        self.atomic_seconds = time.perf_counter() - start
        self.object_index = {obj: i for i, obj in enumerate(self.objects)}
        self.pair_index = {pair: i for i, pair in enumerate(pair_keys)}

    @staticmethod
    def _keys(symbol):
        s = str(symbol)
        return (s, s.replace(" ", "_"), s.replace("_", " "))

    def _class_index(self, family, symbol):
        mapping = self.base.indices[family]
        for key in self._keys(symbol):
            if key in mapping:
                return int(mapping[key])
        if family == "relation":
            value = canonical_relation(symbol)
            for key in self._keys(value):
                if key in mapping:
                    return int(mapping[key])
        return None

    def preload_object_symbols(self, objects, symbols):
        return None

    def object_symbol(self, pred, obj, symbol):
        obj_idx = self.object_index.get(str(obj))
        family = "name" if canonical_relation(pred) == "Name" else "attribute"
        class_idx = self._class_index(family, symbol)
        if obj_idx is None or class_idx is None:
            return 0.0
        probs = self.name_probs if family == "name" else self.attr_probs
        return float(probs[obj_idx, class_idx])

    def object_pair(self, pred, src, dst):
        pair_idx = self.pair_index.get((str(src), str(dst)))
        class_idx = self._class_index("relation", pred)
        if pair_idx is None or class_idx is None:
            return 0.0
        return float(self.rel_probs[pair_idx, class_idx])


def rank_metrics(scores, gold):
    ranked = [obj for obj, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
    gold = set(str(x) for x in gold)
    # Recall is undefined for an empty gold set -- excluded (None, which
    # mean() already skips) rather than forced through max(1, len(gold)),
    # which silently scored such examples as 0 and deflated the average.
    # Denominator is min(len(gold), k), not len(gold): some instances have
    # many valid answers (up to 40 seen in train_tasks_c2_10000.pkl), and
    # dividing by the full gold count instead of min(k, len(gold)) caps
    # Recall@5 near k/len(gold) even for a perfect top-k prediction --
    # the standard fix for Recall@k under variable-size relevant sets.
    has_gold = bool(gold)
    return {
        "top1_gold_hit": float(bool(ranked) and ranked[0] in gold),
        "recall_at_5": len(gold.intersection(ranked[:5])) / min(len(gold), 5) if has_gold else None,
        "recall_at_10": len(gold.intersection(ranked[:10])) / min(len(gold), 10) if has_gold else None,
        "prediction": ranked[0] if ranked else None,
    }


def mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def timing(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return {"mean": None, "median": None, "p95": None, "total": 0.0}
    ordered = sorted(values)
    return {"mean": statistics.fmean(values), "median": statistics.median(values),
            "p95": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
            "total": sum(values)}


def summarize(rows):
    attempted = [row for row in rows if row.get("ilp_attempted", True)]
    solved = [row for row in rows if row.get("ilp_top1_gold_hit") is not None]
    return {
        "examples": len(rows),
        "local": {key: mean(rows, "local_" + key) for key in ("top1_gold_hit", "recall_at_5", "recall_at_10")},
        "ilp": {"enabled": bool(attempted), "attempted": len(attempted),
                "evaluated": len(solved), "unsupported": len(attempted) - len(solved),
                "top1_gold_hit": mean(solved, "ilp_top1_gold_hit"),
                "recall_at_5": mean(solved, "ilp_recall_at_5"),
                "mean_returned_answers": mean(solved, "ilp_returned_answers")},
        "timing_seconds": {key: timing(rows, key) for key in
                           ("atomic_seconds", "local_reasoning_seconds", "local_total_seconds",
                            "ilp_reasoning_seconds", "ilp_total_seconds")},
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--task-dir", type=Path, required=True)
    p.add_argument("--kb-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--splits", nargs="+", default=["C2", "C3", "C4", "C5", "C6"])
    p.add_argument("--validation-recall-at-5", type=float, default=0.876210)
    p.add_argument("--global-consistency", action="store_true")
    p.add_argument("--ilp-top-k", type=int, default=5)
    p.add_argument("--skip-ilp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kb = load_kb_facts(args.kb_dir)
    kb_by_src, kb_by_rel_dst = index_kb(kb)
    model = MLPScorer(args.checkpoint, args.device)
    all_rows, split_summaries = [], {}
    pred_path = args.output_dir / "predictions.jsonl"
    with pred_path.open("w") as prediction_file:
        for split in args.splits:
            path = args.task_dir / f"test_tasks_{split.lower()}_1000.pkl"
            tasks = load_vqar_tasks(path, limit=args.limit)
            rows = []
            for task in tqdm(tasks, desc=split):
                instance = vqar_task_to_graphqa_instance(task, kb_facts=kb)
                scorer = model.bind(instance)
                start = time.perf_counter()
                local_scores = evaluate_instance(instance, scorer, kb_by_src, kb_by_rel_dst)
                local_seconds = time.perf_counter() - start
                local = rank_metrics(local_scores, instance.get("expected_answers", []))
                if args.skip_ilp:
                    ilp_scores = None
                    ilp_seconds = None
                else:
                    start = time.perf_counter()
                    ilp_scores = evaluate_instance_ilp(
                        instance, scorer, kb_by_src, kb_by_rel_dst,
                        global_consistency=args.global_consistency,
                        top_k=args.ilp_top_k,
                    )
                    ilp_seconds = time.perf_counter() - start
                row = {"split": split, "question_id": instance.get("source_question_id"),
                       "gold": [str(x) for x in instance.get("expected_answers", [])],
                       "atomic_seconds": scorer.atomic_seconds,
                       "local_reasoning_seconds": local_seconds,
                       "local_total_seconds": scorer.atomic_seconds + local_seconds,
                       **{"local_" + k: v for k, v in local.items()}}
                if args.skip_ilp:
                    row.update({"ilp_attempted": False, "ilp_top1_gold_hit": None,
                                "ilp_recall_at_5": None, "ilp_returned_answers": None,
                                "ilp_prediction": None, "ilp_reasoning_seconds": None,
                                "ilp_total_seconds": None})
                elif ilp_scores is None:
                    row.update({"ilp_attempted": True, "ilp_top1_gold_hit": None, "ilp_recall_at_5": None,
                                "ilp_returned_answers": 0, "ilp_prediction": None,
                                "ilp_reasoning_seconds": ilp_seconds, "ilp_total_seconds": scorer.atomic_seconds + ilp_seconds})
                else:
                    ilp = rank_metrics({k: v for k, v in ilp_scores.items() if v > 0}, instance.get("expected_answers", []))
                    row.update({"ilp_attempted": True, "ilp_top1_gold_hit": ilp["top1_gold_hit"],
                                "ilp_recall_at_5": ilp["recall_at_5"],
                                "ilp_returned_answers": len(ilp_scores),
                                "ilp_prediction": ilp["prediction"],
                                "ilp_reasoning_seconds": ilp_seconds, "ilp_total_seconds": scorer.atomic_seconds + ilp_seconds})
                rows.append(row); all_rows.append(row)
                prediction_file.write(json.dumps(row) + "\n"); prediction_file.flush()
            split_summaries[split] = summarize(rows)
            partial = {"status": "running", "checkpoint": str(args.checkpoint), "splits": split_summaries}
            (args.output_dir / "summary.partial.json").write_text(json.dumps(partial, indent=2))
    summary = {"status": "complete", "protocol": "Scallop Table 2 C2-C6 test; checkpoint selected by validation Recall@5",
               "checkpoint": str(args.checkpoint), "validation_recall_at_5": args.validation_recall_at_5,
               "global_consistency": args.global_consistency, "splits": split_summaries,
               "combined": summarize(all_rows), "scallop_paper_recall_at_5": SCALLOP_RECALL5,
               "scallop_paper_combined_recall_at_5": 0.8422}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.output_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "examples", "ours_local_top1", "ours_local_recall_at_5", "ours_local_recall_at_10",
                         "ours_ilp_top1", "ours_ilp_recall_at_5", "scallop_recall_at_5", "atomic_mean_s", "local_total_mean_s", "ilp_total_mean_s"])
        for split in args.splits:
            s = split_summaries[split]
            writer.writerow([split, s["examples"], s["local"]["top1_gold_hit"], s["local"]["recall_at_5"],
                             s["local"]["recall_at_10"], s["ilp"]["top1_gold_hit"], s["ilp"]["recall_at_5"], SCALLOP_RECALL5[split],
                             s["timing_seconds"]["atomic_seconds"]["mean"],
                             s["timing_seconds"]["local_total_seconds"]["mean"],
                             s["timing_seconds"]["ilp_total_seconds"]["mean"]])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
