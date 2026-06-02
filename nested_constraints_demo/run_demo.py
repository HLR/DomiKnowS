"""End-to-end demo runner for the nested-constraints DFA pipeline + PMD training."""
from __future__ import annotations

import argparse
import warnings
from functools import partial

from domiknows.generation import constrained_label_greedy_decode
from domiknows.generation.dfa.graph_discovery import analyze_generation_constraints
from domiknows.generation.dfa.core import minimize_dfa
from domiknows.generation.dfa._lc_normalize import normalize_lc

try:
    from .constraints import CONSTRAINT_DESCRIPTIONS
    from .corpus import BUCKETS, verify_acceptance
    from .graph import build_bundle
    from .learning_program import build_learning_program
    from .stream_generator import PROMPT_ORDER
    from .utils import (
        AdamWithGradSnapshot,
        _enable_domiknows_production_logging,
        _enable_remote_debug,
        capture_parameter_snapshot,
        format_lc_source,
        format_mirror_tree,
        print_constrained_greedy_inference,
        print_gradient_snapshot,
        print_greedy_inference,
        print_inference_header,
        print_no_training_requested,
        print_parameter_update_snapshot,
        print_trained_batch,
        print_training_header,
        reset_optimizer_grad_snapshot,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
    from constraints import CONSTRAINT_DESCRIPTIONS
    from corpus import BUCKETS, verify_acceptance
    from graph import build_bundle
    from learning_program import build_learning_program
    from stream_generator import PROMPT_ORDER
    from utils import (
        AdamWithGradSnapshot,
        _enable_domiknows_production_logging,
        _enable_remote_debug,
        capture_parameter_snapshot,
        format_lc_source,
        format_mirror_tree,
        print_constrained_greedy_inference,
        print_gradient_snapshot,
        print_greedy_inference,
        print_inference_header,
        print_no_training_requested,
        print_parameter_update_snapshot,
        print_trained_batch,
        print_training_header,
        reset_optimizer_grad_snapshot,
    )


_enable_domiknows_production_logging()


def _print_pipeline_trace(graph, bundle) -> None:
    """Walk every head LC and print the input -> normalized -> analyses trace."""
    print("=" * 72)
    print("LC -> DFA pipeline trace")
    print("=" * 72)
    head_lcs = [(name, lc) for name, lc in graph.logicalConstrains.items() if getattr(lc, "headLC", True)]
    for index, ((lc_name, lc), (label, description)) in enumerate(zip(head_lcs, CONSTRAINT_DESCRIPTIONS)):
        print(f"\n--- LC #{index + 1}: {label} ({lc_name}) ---")
        print(f"  intent: {description}")
        print("\n  original (Python source):")
        print(format_lc_source(lc, indent=2))
        normal = normalize_lc(lc, bundle=bundle)
        print("\n  normalized mirror tree:")
        print(format_mirror_tree(normal.tree, indent=2))
        if normal.irregular_children:
            print(f"\n  irregular siblings ({len(normal.irregular_children)}) salvaged off:")
            for irregular in normal.irregular_children:
                print(f"    - {irregular!r}")
        if normal.is_constant is not None:
            print(f"\n  constant-folded to: {normal.is_constant}")

    # Disable analyses warnings inside the trace section so we don't print
    # the salvage warning twice (it already appeared during the build).
    print("\n--- analyses ---")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analyses = analyze_generation_constraints(graph, bundle, on_unsupported="ignore")
    for analysis in analyses:
        status = "supported" if analysis.supported else "unsupported"
        dfa_count = len(analysis.dfas) if analysis.supported else 0
        reason = analysis.reason or ""
        print(f"  {analysis.lc_name:>40} | {analysis.lc_type:<10} | {status:>11} | dfas={dfa_count} | {reason}")


def _print_dfa_size_comparison(graph, bundle) -> None:
    """Show the minimization win: raw product vs minimized state count."""
    from domiknows.generation.dfa.graph_discovery import constraints_to_dfa_from_graph

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dfa_raw = constraints_to_dfa_from_graph(graph, bundle, minimize=False, on_unsupported="ignore")
        dfa_min = constraints_to_dfa_from_graph(graph, bundle, minimize=True, on_unsupported="ignore")
    print("\n--- DFA size ---")
    print(f"  product (unminimized):  {len(dfa_raw.states):>5} states")
    print(f"  after minimize_dfa:     {len(dfa_min.states):>5} states")
    delta = len(dfa_raw.states) - len(dfa_min.states)
    print(f"  saved by minimization:  {delta:>5} states")


def _print_corpus_acceptance(artifacts) -> None:
    """Iterate the labelled corpus and report per-sequence DFA verdicts."""
    print("\n" + "=" * 72)
    print("Corpus acceptance (DFA verdict vs bucket expectation)")
    print("=" * 72)
    records = verify_acceptance(artifacts.dfa, artifacts.bundle, BUCKETS)
    last_bucket = None
    for record in records:
        if record.bucket != last_bucket:
            print(f"\n  [{record.bucket}] ({record.rule_hint})")
            last_bucket = record.bucket
        symbols = " ".join(record.symbols) if record.symbols else "(empty)"
        verdict = "accepts" if record.accepted else "rejects"
        match = "OK" if record.passes else "FAIL"
        print(f"    {match:<4}  {verdict}  {symbols}")
    fails = [record for record in records if not record.passes]
    if fails:
        print(f"\n  {len(fails)}/{len(records)} sequences disagreed with the bucket label!")
    else:
        print(f"\n  all {len(records)} sequences match their bucket labels.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--learner",
        choices=("discrete-hmm", "hmm", "graph-hmm", "energy"),
        default="discrete-hmm",
        help="Compact-label learner attached through ModuleLearner.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=15,
        help="Number of stream batches before inference.  15 is a stable default for the nested constraint set; values >= 20 occasionally drive the constraint loss into NaN.",
    )
    parser.add_argument("--stream-count", type=int, default=4, help="Examples per training batch.")
    parser.add_argument(
        "--inference-prompt",
        choices=PROMPT_ORDER,
        default="with_A",
        help="Prompt used for post-training greedy inference.",
    )
    parser.add_argument("--pad-size", type=int, default=6, help="Maximum generated length.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic stream seed.")
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate for the compact head and PMD c-loss.")
    parser.add_argument("--beta", type=float, default=0.3, help="PMD constraint-loss weight.")
    parser.add_argument("--remote-debug", action="store_true", help="Enable debugpy before building.")
    parser.add_argument("--debug-host", default="127.0.0.1", help="debugpy host.")
    parser.add_argument("--debug-port", type=int, default=5678, help="debugpy port.")
    parser.add_argument("--debug-wait", action="store_true", help="Wait for debugger to attach.")
    args = parser.parse_args(argv)
    if args.stream_count <= 0:
        parser.error("--stream-count must be positive")
    if args.pad_size < 2:
        parser.error("--pad-size must be at least 2")
    if args.debug_port <= 0 or args.debug_port > 65535:
        parser.error("--debug-port must be in the range 1..65535")
    if args.remote_debug:
        _enable_remote_debug(args.debug_host, args.debug_port, wait=args.debug_wait)

    print("Nested logical constraints with paths -- DomiKnowS LC -> DFA demo")
    print("  Vocabulary: A, B, C, D, END (+ _other padding label)")
    print("  Three head LCs registered:")
    for label, description in CONSTRAINT_DESCRIPTIONS:
        print(f"    - {label}: {description}")
    print(f"  Inference prompt: {args.inference_prompt}")
    print(f"  Learner: {args.learner}; PMD beta={args.beta}; steps={args.steps}; lr={args.lr}")

    # Build the artifacts once.  ``learning_program.build_learning_program``
    # registers the constraints, runs ``discover_generation_enforcement`` with
    # ``on_unsupported="warn"`` (so the heterogeneous-andL salvage warning
    # fires once here) and produces the PMD program + DFA.
    artifacts = build_learning_program(
        learner=args.learner,
        stream_count=args.stream_count,
        stream_seed=args.seed,
        inference_prompt=args.inference_prompt,
        pad_size=args.pad_size,
        beta=args.beta,
    )

    _print_pipeline_trace(artifacts.graph, artifacts.bundle)
    _print_dfa_size_comparison(artifacts.graph, artifacts.bundle)
    _print_corpus_acceptance(artifacts)

    # Training loop -- same shape as Tasks/real_hmm_pmd_learning/run_demo.py.
    print_training_header()
    lr = args.lr
    if args.steps <= 0:
        print_no_training_requested()
    elif args.steps * args.stream_count <= 10:
        print("  PMD warmup note: constraint loss activates after 10 generated samples; this run will not cross that threshold.")
    for step in range(max(0, args.steps)):
        before_hmm = capture_parameter_snapshot(artifacts.model, hmm_only=True)
        matched_hmm_names = any(
            any(keyword in name.lower() for keyword in ("hmm", "transition", "emission", "initial", "start"))
            for name in before_hmm
        )
        artifacts.stream_examples = artifacts.training_source.next_batch(step)
        reset_optimizer_grad_snapshot()
        artifacts.program.train(
            artifacts.training_source.training_data(artifacts.stream_examples),
            train_epoch_num=1,
            Optim=partial(AdamWithGradSnapshot, lr=lr),
            c_lr=lr,
            print_loss=False,
            persist_c_session=True,
        )
        print_trained_batch(step, len(artifacts.stream_examples))
        print_gradient_snapshot(artifacts.model, step=step)
        after_hmm = capture_parameter_snapshot(artifacts.model, hmm_only=matched_hmm_names)
        print_parameter_update_snapshot(before_hmm, after_hmm, step=step, hmm_matched=matched_hmm_names)

    # Inference.
    print_inference_header()
    prompt = int(artifacts.inference_prompt_token_id)
    inference_result = artifacts.model.greedy_label_inference(
        artifacts.bundle.vocabulary,
        [prompt],
        max_new_tokens=artifacts.model.pad_size,
    )
    print_greedy_inference(artifacts, inference_result)
    inference_dfa_accepted = artifacts.dfa.accepts(inference_result.labels)
    print("Verification of learned greedy inference_result:")
    print("  verifier_call: artifacts.dfa.accepts(inference_result.labels)")
    print(f"  dfa_accepted: {inference_dfa_accepted}")

    constrained_inference_result = constrained_label_greedy_decode(
        artifacts.model,
        [prompt],
        artifacts.bundle.vocabulary,
        artifacts.dfa,
        max_new_tokens=artifacts.model.pad_size,
    )
    print_constrained_greedy_inference(artifacts, constrained_inference_result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
