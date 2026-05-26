"""Run the one-constraint DomiKnowS PMD learning demo with a compact learner."""
from __future__ import annotations

import argparse
import io
from contextlib import redirect_stderr
from functools import partial

import torch

try:
    from .learning_program import build_learning_program
    from .stream_generator import PROMPT_ORDER
    from .utils import (
        print_demo_header,
        print_learning_snapshot,
        print_no_training_requested,
        print_stream_batch,
        print_trained_batch,
        print_training_header,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from learning_program import build_learning_program
    from stream_generator import PROMPT_ORDER
    from utils import (
        print_demo_header,
        print_learning_snapshot,
        print_no_training_requested,
        print_stream_batch,
        print_trained_batch,
        print_training_header,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learner", choices=("graph-hmm", "energy"), default="graph-hmm", help="Compact-label learner attached through ModuleLearner.")
    parser.add_argument("--steps", type=int, default=1, help="Number of live stream PMD training batches before learned-learner inference.")
    parser.add_argument("--stream-count", type=int, default=4, help="Number of generator outputs in each live stream batch.")
    parser.add_argument("--inference-prompt", choices=PROMPT_ORDER, default="AB", help="Prompt used for learned greedy inference after training.")
    parser.add_argument("--pad-size", type=int, default=100, help="Maximum generated length used for padding/truncation and random stream generation.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed for the mock generator stream.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for the compact-label learner and PMD constraint model.")
    args = parser.parse_args(argv)
    if args.stream_count <= 0:
        parser.error("--stream-count must be positive")
    if args.pad_size < 2:
        parser.error("--pad-size must be at least 2")

    # 1. Build the graph, bundle, PMD program, and mock generator source. 
    artifacts = build_learning_program(
        learner=args.learner,
        stream_count=args.stream_count,
        stream_seed=args.seed,
        inference_prompt=args.inference_prompt,
        pad_size=args.pad_size,
    )

    print_demo_header(artifacts)
    print_stream_batch(artifacts.stream_examples, title="\nInitial generator stream")
    print_learning_snapshot(artifacts, title="\nBefore training")
    print_training_header()

    lr = args.lr
    if args.steps <= 0:
        print_no_training_requested()
        return 0
    
    # 2. Train on live stream batches, printing PMD predictions after each batch.
    for step in range(max(0, args.steps)):
        artifacts.stream_examples = artifacts.training_source.next_batch(step)
        print_stream_batch(artifacts.stream_examples, title=f"  live stream batch {step + 1}")
        with redirect_stderr(io.StringIO()):
            artifacts.program.train(
                artifacts.training_source.training_data(artifacts.stream_examples),
                train_epoch_num=1,
                Optim=partial(torch.optim.Adam, lr=lr),
                c_lr=lr,
                print_loss=False,
            )
        print_trained_batch(step, len(artifacts.stream_examples))

    print_learning_snapshot(artifacts, title="\nAfter training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
