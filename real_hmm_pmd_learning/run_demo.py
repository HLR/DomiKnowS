"""Run the one-constraint DomiKnowS PMD learning demo with a compact learner."""
from __future__ import annotations

import argparse
from functools import partial

from domiknows.generation import GenerationCandidate, HybridController, constrained_label_greedy_decode

try:
    from .utils import (
        AdamWithGradSnapshot,
        _enable_domiknows_production_logging,
        _enable_remote_debug,
        capture_parameter_snapshot,
        print_constrained_greedy_inference,
        print_gradient_snapshot,
        print_demo_header,
        print_greedy_inference,
        print_inference_header,
        print_hybrid_controller_ranking,
        print_no_training_requested,
        print_parameter_update_snapshot,
        reset_optimizer_grad_snapshot,
        print_trained_batch,
        print_training_header,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from utils import (
        AdamWithGradSnapshot,
        _enable_domiknows_production_logging,
        _enable_remote_debug,
        capture_parameter_snapshot,
        print_constrained_greedy_inference,
        print_gradient_snapshot,
        print_demo_header,
        print_greedy_inference,
        print_inference_header,
        print_hybrid_controller_ranking,
        print_no_training_requested,
        print_parameter_update_snapshot,
        reset_optimizer_grad_snapshot,
        print_trained_batch,
        print_training_header,
    )

_enable_domiknows_production_logging()

try:
    from .learning_program import build_learning_program
    from .stream_generator import PROMPT_ORDER
except ImportError:  # pragma: no cover - direct script execution fallback
    from learning_program import build_learning_program
    from stream_generator import PROMPT_ORDER


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learner", choices=("discrete-hmm", "hmm", "graph-hmm", "energy"), default="discrete-hmm", help="Compact-label learner attached through ModuleLearner.")
    parser.add_argument("--steps", type=int, default=20, help="Number of live stream PMD training batches before learned-learner inference.  ~20 is enough for the gated DiscreteHMM head to differentiate the three demo prompts.")
    parser.add_argument("--stream-count", type=int, default=4, help="Number of generator outputs in each live stream batch.")
    parser.add_argument("--inference-prompt", choices=PROMPT_ORDER, default="AB", help="Prompt used for learned greedy inference after training.")
    parser.add_argument("--pad-size", type=int, default=6, help="Maximum generated length used for padding/truncation and random stream generation.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed for the mock generator stream.")
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate for the compact-label learner and PMD constraint model.")
    parser.add_argument("--beta", type=float, default=0.3, help="Weight for the PMD constraint loss; larger values enforce rules harder but can drown out the prompt-conditioning signal on this tiny dataset.")
    parser.add_argument("--remote-debug", action="store_true", help="Enable debugpy remote debugging before building the demo.")
    parser.add_argument("--debug-host", default="127.0.0.1", help="Host/interface for --remote-debug.")
    parser.add_argument("--debug-port", type=int, default=5678, help="Port for --remote-debug.")
    parser.add_argument("--debug-wait", action="store_true", help="Wait for a debugger client to attach before continuing.")
    args = parser.parse_args(argv)
    if args.stream_count <= 0:
        parser.error("--stream-count must be positive")
    if args.pad_size < 2:
        parser.error("--pad-size must be at least 2")
    if args.debug_port <= 0 or args.debug_port > 65535:
        parser.error("--debug-port must be in the range 1..65535")

    if args.remote_debug:
        _enable_remote_debug(args.debug_host, args.debug_port, wait=args.debug_wait)

    # 1. Build the graph, bundle, PMD program, and mock generator source. 
    artifacts = build_learning_program(
        learner=args.learner,
        stream_count=args.stream_count,
        stream_seed=args.seed,
        inference_prompt=args.inference_prompt,
        pad_size=args.pad_size,
        beta=args.beta,
    )

    print_demo_header(artifacts)
    print_training_header()

    lr = args.lr
    if args.steps <= 0:
        print_no_training_requested()
    elif args.steps * args.stream_count <= 10:
        print("  PMD warmup note: constraint loss activates after 10 generated samples; this run will not cross that threshold.")
    # 2. Train on live stream batches.
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

    # 3. Run inference explicitly after training.
    print_inference_header()
    
    # Prompt used for inference test 
    prompt = int(artifacts.inference_prompt_token_id)
    
    # Run greedy inference on the learned model to see its generated output
    inference_result = artifacts.model.greedy_label_inference(
        artifacts.bundle.vocabulary,
        [prompt],
        max_new_tokens=artifacts.model.pad_size,
        dfa=artifacts.dfa
    )
    print_greedy_inference(artifacts, inference_result)


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
