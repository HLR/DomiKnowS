"""Run the explicit HMM factor-graph generation demo."""
from __future__ import annotations

import argparse

import torch

try:
    from .hmm_factor_program import (
        build_hmm_factor_program,
        constrained_decode,
        make_optimizers,
        run_one_hmm_factor_step,
        target_labels_for_sample,
    )
except ImportError:
    from hmm_factor_program import (
        build_hmm_factor_program,
        constrained_decode,
        make_optimizers,
        run_one_hmm_factor_step,
        target_labels_for_sample,
    )


def _prediction_summary(artifacts) -> str:
    labels = target_labels_for_sample(artifacts)
    with torch.no_grad():
        generated = artifacts.generated_model(None, artifacts.sample_data["instruction_tokens"], labels)
        latent = artifacts.latent_model(None, artifacts.sample_data["instruction_tokens"], labels)
    generated_preds = [int(torch.argmax(row).item()) for row in generated]
    latent_preds = [artifacts.bundle.state_names[int(torch.argmax(row).item())] for row in latent]
    return (
        f"labels={labels.tolist()} generated_preds={generated_preds} "
        f"latent_preds={latent_preds} accepted={artifacts.dfa.accepts(generated_preds)}"
    )


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
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--pad-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--state-names", nargs="+", default=["PER", "O", "LOC"])
    parser.add_argument("--supervised-weight", type=float, default=3.0)
    parser.add_argument("--constraint-weight", type=float, default=1.0)
    parser.add_argument("--hmm-weight", type=float, default=1.0)
    args = parser.parse_args(argv)

    artifacts = build_hmm_factor_program(
        pad_size=args.pad_size,
        state_names=tuple(args.state_names),
        trainable=True,
    )

    print("HMM factor graph path: generated_token + latent_state DataNodes")
    print("Enforcement path: generated_token projection -> graph-discovered DFA")
    print("State names:", artifacts.bundle.state_names)
    print("Trainable parameters:", artifacts.head.trainable_parameter_names())
    print("Before:", _prediction_summary(artifacts))
    optimizers = make_optimizers(artifacts, lr=args.lr)
    for step in range(args.steps):
        losses = run_one_hmm_factor_step(
            artifacts,
            lr=args.lr,
            optimizers=optimizers,
            supervised_weight=args.supervised_weight,
            constraint_weight=args.constraint_weight,
            hmm_weight=args.hmm_weight,
        )
        print(f"Step {step + 1}: {losses}")
    print("After unconstrained:", _prediction_summary(artifacts))
    print("After DFA-constrained:", _constrained_summary(artifacts))
    print("Vocabulary:", artifacts.bundle.vocabulary.labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
