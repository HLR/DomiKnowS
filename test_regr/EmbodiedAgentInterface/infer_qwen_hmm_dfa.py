"""Inference-only EAI evaluation: DomiKnowS default Qwen generator + HMM + DFA.

This script does not train or load a trained checkpoint. It follows the normal
`main.py` construction path: load examples, call `build_trainable_program(...)`
to create the graph/bundle/default DomiKnowS generator, compile the graph to a
DFA, and decode with HMM + DFA using that generator's next-label logits.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import evaluate_settings as ev
from main import build_trainable_program

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference-only EAI DomiKnowS-Qwen + HMM + DFA decoding.")
    parser.add_argument("--dataset", choices=["all", "behavior", "virtualhome"], default="all")
    parser.add_argument("--split", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples loaded before building graph/vocabulary.")
    parser.add_argument("--eval-limit", type=int, default=None, help="Limit selected examples scored.")
    parser.add_argument("--eval-split", choices=["dev", "train", "full"], default="full")
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument(
        "--dfa-build-mode",
        choices=["per-example", "batch", "combined"],
        default="per-example",
        help="per-example builds one base DFA plus per-sample wrappers; batch uses chunked graph DFAs; combined uses one legacy graph DFA.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Only used with --dfa-build-mode batch. Build graph/DFA and decode in chunks of this many selected examples.",
    )
    parser.add_argument("--max-steps", type=int, default=135)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", default=str(SCRIPT_DIR / "results_qwen_hmm_dfa_inference.txt"))
    parser.add_argument("--show", type=int, default=0)

    # Match the graph/generator construction arguments expected by main.py.
    parser.add_argument("--program", choices=["solver", "primal-dual"], default="solver")
    parser.add_argument("--baseline-model", choices=["bert-gru", "tiny-transformer", "causal-lm"], default="causal-lm")
    parser.add_argument("--feature-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--encoder-model-path", default="bert-base-uncased")
    parser.add_argument("--encoder-max-length", type=int, default=256)
    parser.add_argument("--finetune-encoder", action="store_true")
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--llm-backbone-path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--llm-device-map", default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="*", default=None)

    parser.add_argument("--hmm", default=str(SCRIPT_DIR / "models/eai_all_qwen25_ctrlg_hmm.npz"))
    parser.add_argument("--hmm-search", choices=["greedy", "beam", "sample"], default="greedy")
    parser.add_argument("--hmm-beam-size", type=int, default=4)
    parser.add_argument("--hmm-dfa-objective", choices=["ctrl_g", "log_linear_blend"], default="ctrl_g")
    parser.add_argument("--hmm-dfa-base", choices=["auto", "backend", "hmm"], default="auto")
    parser.add_argument("--hmm-base-weight", type=float, default=None)
    parser.add_argument("--hmm-weight", type=float, default=1.0)
    parser.add_argument("--hmm-hf-weight", type=float, default=0.0, help="Log-linear blend objective backend label-bias weight.")
    parser.add_argument("--hmm-lookahead-weight", type=float, default=0.0)
    parser.add_argument("--hmm-lookahead-max-steps", type=int, default=8)
    parser.add_argument("--hmm-keep-rejected", action="store_true")
    return parser.parse_args()


def _chunks(items, size):
    if size <= 0:
        yield list(items)
        return
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _new_counts():
    return {
        "examples": 0,
        "exact": 0,
        "token_correct": 0,
        "token_total": 0,
        "dfa_valid": 0,
        "gt_state_success": 0,
        "gt_state_recall_total": 0.0,
        "pred_len_total": 0,
    }


def _add_batch_counts(counts, predictions, examples, vocabulary, dfa, show, offset):
    eos_label = vocabulary.eos_label
    for local_idx, (sample, pred) in enumerate(zip(examples, predictions)):
        global_idx = offset + local_idx
        gold = ev.gold_labels(sample, len(sample["target_action_labels"]))
        padded = ev.pad_prediction(pred, gold, eos_label)
        pred_trimmed = ev.trim_at_eos(pred, eos_label)
        gold_trimmed = ev.trim_at_eos(gold, eos_label)
        counts["examples"] += 1
        counts["exact"] += int(padded == gold)
        counts["token_correct"] += sum(int(p == g) for p, g in zip(padded, gold))
        counts["token_total"] += len(gold)
        counts["dfa_valid"] += int(dfa.accepts(pred_trimmed) or dfa.accepts(padded))
        goal_result = ev.evaluate_goal_satisfaction(pred_trimmed, sample, vocabulary)
        predicted_state = goal_result["predicted_state"]
        gold_state = goal_result["gold_state"]
        recall = goal_result["recall"]
        counts["gt_state_success"] += int(goal_result["is_success"] == 1.0)
        counts["gt_state_recall_total"] += recall
        counts["pred_len_total"] += len(pred)
        if global_idx < show:
            print()
            print(f"## DomiKnowS HMM+DFA decoder with Qwen-distilled HMM (no training) example {global_idx}: {sample.get('task_id', 'task')}")
            print(f"Instruction: {sample.get('natural_language_description') or sample.get('text')}")
            print(f"Gold: {ev.labels_to_actions(gold_trimmed, vocabulary)}")
            print(f"Pred: {ev.labels_to_actions(pred_trimmed, vocabulary)}")
            print(f"Gold state: {sorted(gold_state)}")
            print(f"Pred state: {sorted(predicted_state)}")


def _score_from_counts(name, counts):
    examples = counts["examples"]
    if not examples:
        return {
            "name": name,
            "examples": 0,
            "exact_sequence": 0.0,
            "token_accuracy": 0.0,
            "dfa_valid": 0.0,
            "gt_state_success": 0.0,
            "gt_state_recall": 0.0,
            "avg_pred_len": 0.0,
        }
    return {
        "name": name,
        "examples": examples,
        "exact_sequence": counts["exact"] / examples,
        "token_accuracy": counts["token_correct"] / counts["token_total"] if counts["token_total"] else 0.0,
        "dfa_valid": counts["dfa_valid"] / examples,
        "gt_state_success": counts["gt_state_success"] / examples,
        "gt_state_recall": counts["gt_state_recall_total"] / examples,
        "avg_pred_len": counts["pred_len_total"] / examples,
    }


def _restore_attrs(args, previous):
    for name, marker in previous.items():
        if marker is _MISSING:
            try:
                delattr(args, name)
            except AttributeError:
                pass
        else:
            setattr(args, name, marker)


_MISSING = object()


def _build_program_with_constraints(args, examples, *, enforce_action_object, enforce_action_object_constraints):
    attrs = {
        "_enforce_action_object": bool(enforce_action_object),
        "_enforce_action_object_constraints": bool(enforce_action_object_constraints),
    }
    previous = {name: getattr(args, name, _MISSING) for name in attrs}
    try:
        for name, value in attrs.items():
            setattr(args, name, value)
        return build_trainable_program(args, examples, args.device)
    finally:
        _restore_attrs(args, previous)


def _capture_shared_llm(args, program):
    if (
        args.baseline_model == "causal-lm"
        and not args.use_lora
        and getattr(args, "_shared_llm_model", None) is None
        and hasattr(program, "autoregressive_head")
    ):
        args._shared_llm_model = program.autoregressive_head.model
        args._shared_llm_tokenizer = program.autoregressive_head.tokenizer


def _example_dfa(base_dfa, vocabulary, sample):
    sample_examples = [sample]
    action_tokens = ev.action_tokens_requiring_object_from_examples(sample_examples)
    object_tokens = ev.object_tokens_from_examples(sample_examples)
    compatibility = ev.action_object_constraint_tokens_from_examples(sample_examples)
    action_sequence_tokens = ev.action_tokens_from_examples(sample_examples)
    dfa = ev.action_object_runtime_dfa(
        base_dfa,
        vocabulary,
        action_tokens,
        object_tokens,
        compatibility,
        action_sequence_tokens,
    )
    overlay_count = len(getattr(dfa, "overlays", ()) or ())
    stats = {
        "constraint_count": overlay_count,
        "runtime_overlay_count": overlay_count,
        "action_requires_object_count": len(action_tokens),
        "object_count": len(object_tokens),
        "action_sequence_count": len(action_sequence_tokens),
        "compatibility_action_count": len(compatibility),
        "compatibility_pair_count": sum(len(objects) for objects in compatibility.values()),
    }
    return dfa, stats


def _decode_one_dfa_per_example(args, all_examples, examples):
    from domiknows.generation import constraints_to_dfa_from_graph
    from domiknows.generation.applications.hybrid import HybridController

    counts = _new_counts()
    started = time.perf_counter()
    print("per-example mode: building base program", flush=True)
    program, bundle = _build_program_with_constraints(
        args,
        all_examples,
        enforce_action_object=False,
        enforce_action_object_constraints=False,
    )
    _capture_shared_llm(args, program)
    print("per-example mode: compiling base DFA", flush=True)
    base_dfa = constraints_to_dfa_from_graph(program.graph, bundle)
    base_dfa_states = len(base_dfa.states)
    print(f"per-example mode: base DFA states={base_dfa_states}", flush=True)

    scorer_head = ev.load_hmm_generation_head(args.hmm, bundle.vocabulary, args.device)
    failures = 0
    lookahead_entries_total = 0
    lookahead_entries_count = 0
    lookahead_entries_max = 0
    composed_dfa_states_max = 0
    trace_log = None
    iterator = ev.progress_bar(enumerate(examples), total=len(examples), desc="4 product HMM+DFA per-example")
    for index, sample in iterator:
        dfa, trace_stats = _example_dfa(base_dfa, bundle.vocabulary, sample)
        composed_dfa_states = len(getattr(dfa, "states", ()))
        composed_dfa_states_max = max(composed_dfa_states_max, composed_dfa_states)
        controller = HybridController(
            dfa=dfa,
            vocabulary=bundle.vocabulary,
            generator=program.autoregressive_head,
            scorer_head=scorer_head,
            tokenizer=None,
        )
        prompt = torch.tensor([[int(bundle.vocabulary.eos_label)]], dtype=torch.long, device=args.device)
        results = controller.decode_hmm_dfa(
            prompt,
            search=args.hmm_search,
            num_return_sequences=1,
            beam_size=args.hmm_beam_size,
            max_new_tokens=args.max_steps,
            keep_rejected=args.hmm_keep_rejected,
            temperature=0.0 if args.hmm_search != "sample" else 1.0,
            hmm_dfa_objective=args.hmm_dfa_objective,
            hmm_dfa_base=args.hmm_dfa_base,
            base_weight=args.hmm_base_weight,
            hmm_weight=args.hmm_weight,
            hf_weight=args.hmm_hf_weight,
            lookahead_weight=args.hmm_lookahead_weight,
            lookahead_max_steps=args.hmm_lookahead_max_steps,
            trace_context={
                "example_index": index,
                "task_id": sample.get("task_id", "task"),
                "dataset": args.dataset,
                "dfa_build_mode": args.dfa_build_mode,
                "base_dfa_states": base_dfa_states,
                "composed_dfa_states": composed_dfa_states,
                **trace_stats,
            },
        )
        if results:
            trace_log = trace_log or results[0].metadata.get("trace_log")
            predictions = [results[0].labels]
            entries = results[0].metadata.get("lookahead_entries")
            if entries is not None:
                entries = int(entries)
                lookahead_entries_total += entries
                lookahead_entries_count += 1
                lookahead_entries_max = max(lookahead_entries_max, entries)
        else:
            failures += 1
            predictions = [ev._invalid_prediction(bundle, sample, args.max_steps)]
            if ev.tqdm is not None and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(decode_failures=failures)
        _add_batch_counts(counts, predictions, [sample], bundle.vocabulary, dfa, args.show, index)
    if failures:
        print(f"4 product HMM+DFA per-example: decode_failures={failures}")
    elapsed_seconds = time.perf_counter() - started
    performance_line = f"per-example mode: done in {elapsed_seconds:.1f}s"
    print(performance_line, flush=True)
    return (
        _score_from_counts(
            "DomiKnowS HMM+DFA decoder with Qwen-distilled HMM (no training)",
            counts,
        ),
        {
            "base_dfa_states": base_dfa_states,
            "examples_decoded": len(examples),
            "composed_dfa_states_max": composed_dfa_states_max,
            "lookahead_entries_avg": (
                lookahead_entries_total / lookahead_entries_count
                if lookahead_entries_count
                else None
            ),
            "lookahead_entries_max": lookahead_entries_max if lookahead_entries_count else None,
            "trace_log": trace_log,
            "elapsed_seconds": elapsed_seconds,
            "performance_line": performance_line,
        },
    )


def _decode_batched(args, all_examples, examples):
    from domiknows.generation import constraints_to_dfa_from_graph

    counts = _new_counts()
    started = time.perf_counter()
    batch_size = int(args.batch_size)
    if args.dfa_build_mode == "combined":
        build_batches = [(list(all_examples), list(examples))]
    else:
        if batch_size <= 0:
            raise ValueError("--batch-size must be > 0 when --dfa-build-mode batch")
        build_batches = [(batch, batch) for batch in _chunks(examples, batch_size)]
    dfa_state_counts = []
    offset = 0
    for batch_index, (build_examples, decode_examples) in enumerate(build_batches, start=1):
        batch_started = time.perf_counter()
        print(
            f"batch {batch_index}/{len(build_batches)}: build_examples={len(build_examples)} decode_examples={len(decode_examples)}",
            flush=True,
        )
        program, bundle = _build_program_with_constraints(
            args,
            build_examples,
            enforce_action_object=True,
            enforce_action_object_constraints=True,
        )
        print(f"batch {batch_index}/{len(build_batches)}: compiling DFA", flush=True)
        dfa = constraints_to_dfa_from_graph(program.graph, bundle)
        dfa_state_counts.append(len(dfa.states))
        print(f"batch {batch_index}/{len(build_batches)}: DFA states={len(dfa.states)}", flush=True)
        desc = f"4 product HMM+DFA batch {batch_index}/{len(build_batches)}"
        predictions = ev._domiknows_hmm_dfa_predictions(
            args,
            dfa,
            bundle,
            program.autoregressive_head,
            decode_examples,
            desc=desc,
        )
        _add_batch_counts(counts, predictions, decode_examples, bundle.vocabulary, dfa, args.show, offset)
        _capture_shared_llm(args, program)
        offset += len(decode_examples)
        print(f"batch {batch_index}/{len(build_batches)}: done in {time.perf_counter() - batch_started:.1f}s", flush=True)
    elapsed_seconds = time.perf_counter() - started
    return (
        _score_from_counts(
            "DomiKnowS HMM+DFA decoder with Qwen-distilled HMM (no training)",
            counts,
        ),
        {
            "base_dfa_states": dfa_state_counts[0] if len(dfa_state_counts) == 1 else dfa_state_counts,
            "batches": len(build_batches),
            "trace_log": getattr(args, "_hmm_dfa_trace_log", None),
            "elapsed_seconds": elapsed_seconds,
            "performance_line": f"{args.dfa_build_mode} mode: done in {elapsed_seconds:.1f}s",
        },
    )


def main():
    args = parse_args()
    if args.batch_size is not None and args.batch_size <= 0 and args.dfa_build_mode == "batch":
        raise ValueError("--batch-size must be > 0 when --dfa-build-mode batch")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    all_examples = ev.load_examples(args, args.device)
    if not all_examples:
        raise ValueError("No EAI examples were loaded.")
    examples = ev.select_eval_examples(all_examples, args.eval_split, args.dev_fraction, args.eval_limit)
    if not examples:
        raise ValueError(f"No examples selected for eval_split={args.eval_split!r}.")

    if args.dfa_build_mode == "per-example":
        score, metadata = _decode_one_dfa_per_example(args, all_examples, examples)
    else:
        score, metadata = _decode_batched(args, all_examples, examples)

    dfa_metadata = f"base_dfa_states={metadata['base_dfa_states']}"
    if args.dfa_build_mode == "per-example":
        dfa_metadata += f" examples_decoded={metadata['examples_decoded']}"
        if metadata.get("lookahead_entries_avg") is not None:
            dfa_metadata += (
                f" composed_dfa_states_max={metadata['composed_dfa_states_max']}"
                f" lookahead_entries_avg={metadata['lookahead_entries_avg']:.1f}"
                f" lookahead_entries_max={metadata['lookahead_entries_max']}"
            )
    else:
        dfa_metadata += f" batches={metadata['batches']}"
    if metadata.get("trace_log"):
        dfa_metadata += f" trace_log={metadata['trace_log']}"

    experiment_date = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "EAI inference-only DomiKnowS HMM+DFA decoder with Qwen-distilled HMM",
        f"experiment_date={experiment_date}",
        f"performance={metadata['performance_line']}",
        f"dataset={args.dataset} eval_split={args.eval_split} examples={len(examples)} loaded_examples={len(all_examples)} dfa_build_mode={args.dfa_build_mode} batch_size={args.batch_size} max_steps={args.max_steps}",
        dfa_metadata,
        f"generator={args.baseline_model} qwen={args.llm_backbone_path}",
        f"hmm={args.hmm}",
        f"hmm_search={args.hmm_search} hmm_dfa_objective={args.hmm_dfa_objective} hmm_dfa_base={args.hmm_dfa_base} "
        f"hmm_base_weight={args.hmm_base_weight} hmm_weight={args.hmm_weight} "
        f"log_linear_backend_generator_weight={args.hmm_hf_weight} lookahead_weight={args.hmm_lookahead_weight}",
        "",
        ev.format_score(score),
        "",
        json.dumps([score], indent=2),
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    for line in lines[:8] + [ev.format_score(score)]:
        print(line)
    print(f"saved_results={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
