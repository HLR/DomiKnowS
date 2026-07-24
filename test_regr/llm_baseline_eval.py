"""Direct LLM multiple-choice baselines for TemporalRelation and GraphQA.

This is intentionally separate from DomiKnowS predicate training. The model is
asked to choose one answer from candidate labels/objects and we report accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from test_regr.TemporalRelation.dataset import DEFAULT_TEMPORAL_DATA_ROOT, discover_temporal_datasets, load_temporal_instances
from test_regr.TemporalRelation.graph import TEMPORAL_LABELS, unpack_pair
from test_regr.TemporalRelation.llm_inference import SmallCausalLMChoiceBackend, parse_choice
from test_regr.TemporalRelation.oracle import answer_label
from test_regr.GraphQA.dataset import DEFAULT_VQAR_ROOT, discover_vqar_dataset, load_kb_facts, load_vqar_tasks, vqar_task_to_graphqa_instance
from test_regr.GraphQA.execution import create_query_logic, materialize_bounded_facts
from test_regr.GraphQA.oracle import answer_object


DEFAULT_QWEN3_8B = "/localscratch/premsrit/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"


def progress_iter(iterable, args, desc, total=None):
    if not getattr(args, "progress", True):
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, leave=True, mininterval=1.0)
    except Exception:
        return iterable


def default_matres_path(root, split):
    discovered = discover_temporal_datasets(root)
    target = f"{split}.txt"
    for path in discovered["matres"]:
        path = Path(path)
        if path.name == target:
            return path
    raise FileNotFoundError(f"Could not find MATRES split {target} under {root}")


def default_graphqa_task_path(root, split):
    discovered = discover_vqar_dataset(root)
    names = {
        "train": ["train_tasks_c2_1000.pkl", "train_tasks_c2_10000.pkl", "train_tasks.pkl"],
        "val": ["val_tasks_c2_1000.pkl", "val_tasks.pkl"],
        "test": ["test_tasks_c2_1000.pkl", "test_tasks.pkl"],
    }[split]
    by_name = {path.name: path for path in discovered["task_paths"]}
    for name in names:
        if name in by_name:
            return by_name[name]
    for path in discovered["task_paths"]:
        if path.name.startswith(split):
            return path
    raise FileNotFoundError(f"Could not find GraphQA/VQAR task split {split} under {root}")


def normalized_limit(limit):
    return None if limit in (None, 0) else limit


def write_jsonl(path, rows, summary):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as output_file:
        for row in rows:
            output_file.write(json.dumps({"type": "prediction", **row}, sort_keys=True) + "\n")
        output_file.write(json.dumps({"type": "summary", **summary}, sort_keys=True) + "\n")


def format_prompt(prompt, choices):
    lines = [prompt.strip(), "", "Candidate answers:"]
    for idx, choice in enumerate(choices):
        lines.append(f"{chr(ord('A') + idx)}. {choice}")
    lines.append("")
    lines.append("Return exactly one candidate answer. You may return the letter or the answer text.")
    return "\n".join(lines)


def choose(backend, prompt, choices):
    return backend.choose(prompt, choices)


def temporal_prompt(instance):
    query_pair = instance.get("query_pair") or (instance.get("event_pairs") or [{}])[0]
    e1, e2, _ = unpack_pair(query_pair)
    event_text = {event.get("id"): event.get("text") for event in instance.get("events", []) if isinstance(event, dict)}
    return "\n".join([
        "Task: classify the temporal relation of event E1 relative to event E2.",
        "Answer Before if E1 happened earlier than E2.",
        "Answer After if E1 happened later than E2.",
        "Answer Equal if they are simultaneous.",
        "Answer Vague if the relation is unclear.",
        "",
        f"Document/Text: {instance.get('text') or instance.get('doc_id')}",
        f"E1: {e1} ({event_text.get(e1, e1)})",
        f"E2: {e2} ({event_text.get(e2, e2)})",
    ])


def expand_temporal_query_instances(documents):
    instances = []
    for document in documents:
        pairs = list(document.get("event_pairs", []))
        for pair in pairs:
            e1, e2, label = unpack_pair(pair)
            instance = dict(document)
            instance["events"] = list(document.get("events", []))
            instance["event_pairs"] = pairs
            instance["query_pair"] = {"e1": e1, "e2": e2, "label": label}
            instances.append(instance)
    return instances


def evaluate_temporal(args):
    path = args.path or default_matres_path(args.root, args.split)
    documents = load_temporal_instances(path, limit=None, group_by_document=True)
    instances = expand_temporal_query_instances(documents)
    if normalized_limit(args.limit) is not None:
        instances = instances[: normalized_limit(args.limit)]
    backend = SmallCausalLMChoiceBackend(args.model_path, args.device, args.max_new_tokens)
    total = correct = errors = 0
    rows = []
    iterator = progress_iter(enumerate(instances), args, f"temporal:{path.name}", total=len(instances))
    for index, instance in iterator:
        gold = answer_label(instance)
        if gold is None:
            continue
        total += 1
        try:
            pred = choose(backend, temporal_prompt(instance), list(TEMPORAL_LABELS))
            ok = pred == gold
            correct += int(ok)
            rows.append({"index": index, "doc_id": instance.get("doc_id"), "pred": pred, "gold": gold, "ok": ok})
        except Exception as exc:
            errors += 1
            rows.append({"index": index, "doc_id": instance.get("doc_id"), "gold": gold, "error": str(exc)})
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(acc=f"{correct / max(total, 1):.3f}", errors=errors, n=total)
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(acc=f"{correct / max(total, 1):.3f}", errors=errors, n=total)
        if args.print_examples and len(rows) <= args.print_examples:
            print(json.dumps(rows[-1], sort_keys=True), flush=True)
    summary = {"task": "temporal", "path": str(path), "documents": len(documents), "loaded": len(instances), "total": total, "correct": correct, "errors": errors, "accuracy": correct / total if total else 0.0}
    write_jsonl(args.output_jsonl, rows, summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


def graphqa_prompt(instance):
    query_logic = create_query_logic(instance)
    facts = materialize_bounded_facts(instance)
    fact_preview = facts[:80]
    return "\n".join([
        "Task: answer a GraphQA object question by selecting exactly one candidate object.",
        "Use the scene facts, bounded TypeOf propagation, and query logic.",
        "",
        f"Objects: {', '.join(instance.get('objects', []))}",
        f"Query: {instance.get('query')}",
        f"DomiKnowS-style logic: {query_logic}",
        f"Facts preview: {fact_preview}",
    ])


def load_graphqa_instances(args):
    path = args.task_path or default_graphqa_task_path(args.root, args.split)
    tasks = load_vqar_tasks(path, limit=normalized_limit(args.limit))
    kb_facts = [] if args.no_kb else load_kb_facts(args.kb_dir)
    instances = []
    failures = []
    for index, task in enumerate(tasks):
        try:
            # Keep direct baseline bounded: using all 226k KB facts in the prompt is impossible.
            instance = vqar_task_to_graphqa_instance(task, kb_facts=[])
            if kb_facts and args.max_kb_facts > 0:
                instance["kb_facts"] = _filter_kb_for_graphqa_prompt(instance, kb_facts, args.max_kb_facts)
            instance["facts"] = materialize_bounded_facts(instance)
            instances.append(instance)
        except Exception as exc:
            failures.append((index, type(exc).__name__, str(exc)))
    return path, instances, failures


def _filter_kb_for_graphqa_prompt(instance, kb_facts, max_kb_facts):
    symbols = set(instance.get("symbols", []))
    for pred, _left, right in instance.get("query", {}).get("conditions", []):
        if isinstance(right, str):
            symbols.add(right)
    selected = []
    frontier = set(symbols)
    for _depth in range(2):
        next_frontier = set()
        for pred, left, right in kb_facts:
            if len(selected) >= max_kb_facts:
                return selected
            if pred in {"TypeOf", "Hypernym"} and left in frontier:
                selected.append(("TypeOf", left, right))
                next_frontier.add(right)
        frontier = next_frontier
    return selected


def evaluate_graphqa(args):
    path, instances, failures = load_graphqa_instances(args)
    backend = SmallCausalLMChoiceBackend(args.model_path, args.device, args.max_new_tokens)
    total = correct = errors = 0
    rows = []
    iterator = progress_iter(enumerate(instances), args, f"graphqa:{path.name}", total=len(instances))
    for index, instance in iterator:
        gold = instance.get("expected_answer")
        if gold is None:
            try:
                gold = answer_object(instance)
            except Exception:
                gold = None
        if gold is None:
            continue
        total += 1
        choices = list(instance.get("objects", []))
        try:
            pred = choose(backend, graphqa_prompt(instance), choices)
            ok = str(pred) == str(gold)
            correct += int(ok)
            rows.append({"index": index, "qid": instance.get("source_question_id"), "pred": pred, "gold": gold, "ok": ok})
        except Exception as exc:
            errors += 1
            rows.append({"index": index, "qid": instance.get("source_question_id"), "gold": gold, "error": str(exc)})
        if args.print_examples and len(rows) <= args.print_examples:
            print(json.dumps(rows[-1], sort_keys=True), flush=True)
    summary = {
        "task": "graphqa",
        "path": str(path),
        "loaded": len(instances),
        "conversion_failures": len(failures),
        "total": total,
        "correct": correct,
        "errors": errors,
        "accuracy": correct / total if total else 0.0,
    }
    write_jsonl(args.output_jsonl, rows, summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    if failures[:5]:
        print(json.dumps({"first_failures": failures[:5]}, sort_keys=True), flush=True)
    return 0


def add_common(parser):
    parser.add_argument("--model-path", default=DEFAULT_QWEN3_8B)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="Number of examples to evaluate; 0 means full split.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--print-examples", type=int, default=5)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-jsonl", type=Path, default=None, help="Optional path to save per-example predictions and final summary.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate direct LLM baselines for TemporalRelation and GraphQA.")
    sub = parser.add_subparsers(dest="task", required=True)

    temporal = sub.add_parser("temporal")
    add_common(temporal)
    temporal.add_argument("--root", type=Path, default=DEFAULT_TEMPORAL_DATA_ROOT)
    temporal.add_argument("--path", type=Path, default=None)
    temporal.add_argument("--split", choices=["timebank", "aquaint", "platinum"], default="aquaint")
    temporal.set_defaults(func=evaluate_temporal)

    graphqa = sub.add_parser("graphqa")
    add_common(graphqa)
    graphqa.add_argument("--root", type=Path, default=DEFAULT_VQAR_ROOT)
    graphqa.add_argument("--task-path", type=Path, default=None)
    graphqa.add_argument("--split", choices=["train", "val", "test"], default="val")
    graphqa.add_argument("--kb-dir", type=Path, default=None)
    graphqa.add_argument("--no-kb", action="store_true")
    graphqa.add_argument("--max-kb-facts", type=int, default=64)
    graphqa.set_defaults(func=evaluate_graphqa)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
