"""Run HMM/WFA generation heads through DomiKnowS PMD and DFA decoding."""
from __future__ import annotations

import argparse

import torch

try:
    from .automata_program import (
        build_automata_learning_program,
        constrained_decode,
        make_optimizers,
        run_one_automata_training_step,
        target_labels_for_sample,
    )
except ImportError:
    from automata_program import (
        build_automata_learning_program,
        constrained_decode,
        make_optimizers,
        run_one_automata_training_step,
        target_labels_for_sample,
    )


def _prediction_summary(artifacts) -> str:
    labels = target_labels_for_sample(artifacts).tolist()
    with torch.no_grad():
        log_probs = artifacts.model(
            None,
            artifacts.sample_data["instruction_tokens"],
            torch.tensor(labels, dtype=torch.long),
        )
    preds = [int(torch.argmax(row).item()) for row in log_probs]
    return f"labels={labels} preds={preds} accepted={artifacts.dfa.accepts(preds)}"


def _constrained_summary(artifacts) -> str:
    result = constrained_decode(artifacts)
    prompt_len = int(artifacts.sample_data["instruction_tokens"].shape[-1])
    generated_token_ids = result.token_ids[prompt_len:]
    text = artifacts.tokenizer.decode(generated_token_ids)
    return (
        f"labels={result.labels} token_ids={generated_token_ids} "
        f"text={text!r} accepted={result.accepted}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("hmm", "wfa"), default="hmm")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--pad-size", type=int, default=4)
    parser.add_argument("--state-count", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--supervised-weight", type=float, default=3.0)
    parser.add_argument("--constraint-weight", type=float, default=1.0)
    parser.add_argument("--automata-weight", type=float, default=1.0)
    args = parser.parse_args(argv)

    artifacts = build_automata_learning_program(
        kind=args.kind,
        pad_size=args.pad_size,
        state_count=args.state_count,
        trainable=True,
    )

    print(f"Automata learning path: {args.kind.upper()} head + DomiKnowS PMD constraint loss")
    print("Enforcement path: graph constraints -> DFA mask for automata-head decoding")
    print("Trainable parameters:", artifacts.model.trainable_parameter_names())
    print("Before:", _prediction_summary(artifacts))
    optimizers = make_optimizers(artifacts, lr=args.lr)
    for step in range(args.steps):
        losses = run_one_automata_training_step(
            artifacts,
            lr=args.lr,
            optimizers=optimizers,
            supervised_weight=args.supervised_weight,
            constraint_weight=args.constraint_weight,
            automata_weight=args.automata_weight,
        )
        print(f"Step {step + 1}: {losses}")
    print("After unconstrained:", _prediction_summary(artifacts))
    print("After DFA-constrained:", _constrained_summary(artifacts))
    print("Vocabulary:", artifacts.bundle.vocabulary.labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
