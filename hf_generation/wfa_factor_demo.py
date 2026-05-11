"""Run the explicit spectral-WFA factor-graph generation demo."""
from __future__ import annotations

import argparse

import torch

try:
    from .wfa_factor_program import (
        build_wfa_factor_program,
        constrained_decode,
        make_optimizers,
        run_one_wfa_factor_step,
        target_labels_for_sample,
    )
    from .loss_logging import format_loss_log, print_loss_log_note
except ImportError:
    from wfa_factor_program import (
        build_wfa_factor_program,
        constrained_decode,
        make_optimizers,
        run_one_wfa_factor_step,
        target_labels_for_sample,
    )
    from loss_logging import format_loss_log, print_loss_log_note


def _prediction_summary(artifacts) -> str:
    labels = target_labels_for_sample(artifacts)
    with torch.no_grad():
        generated = artifacts.generated_model(None, artifacts.sample_data["instruction_tokens"], labels)
        states = artifacts.state_model(None, artifacts.sample_data["instruction_tokens"], labels)
    generated_preds = [int(torch.argmax(row).item()) for row in generated]
    state_preds = [artifacts.bundle.state_names[int(torch.argmax(row).item())] for row in states]
    return (
        f"labels={labels.tolist()} generated_preds={generated_preds} "
        f"wfa_state_preds={state_preds} accepted={artifacts.dfa.accepts(generated_preds)}"
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
    parser.add_argument("--state-names", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--supervised-weight", type=float, default=3.0)
    parser.add_argument("--constraint-weight", type=float, default=1.0)
    parser.add_argument("--wfa-weight", type=float, default=1.0)
    parser.add_argument("--factor-weight", type=float, default=1.0)
    parser.add_argument("--latent-weight", type=float, default=0.0)
    parser.add_argument("--allowed-mass-weight", type=float, default=0.0)
    parser.add_argument("--latent-mode", choices=("marked", "auto", "marked-and-auto"), default="marked")
    parser.add_argument("--latent-diagnostics", action="store_true")
    parser.add_argument("--no-transition-pairs", action="store_true")
    args = parser.parse_args(argv)

    artifacts = build_wfa_factor_program(
        pad_size=args.pad_size,
        state_names=tuple(args.state_names),
        trainable=True,
        include_transition_pairs=not args.no_transition_pairs,
        latent_mode=args.latent_mode,
    )

    pair_status = "enabled" if artifacts.bundle.include_transition_pairs else "disabled"
    print("Spectral WFA factor graph path: generated_token + wfa_state DataNodes")
    print(f"Transition-pair factor DataNodes: {pair_status}")
    print("Enforcement path: generated_token projection -> graph-discovered DFA")
    print("State names:", artifacts.bundle.state_names)
    print("Trainable parameters:", artifacts.head.trainable_parameter_names())
    print("Before:", _prediction_summary(artifacts))
    print_loss_log_note()
    optimizers = make_optimizers(artifacts, lr=args.lr)
    for step in range(args.steps):
        losses = run_one_wfa_factor_step(
            artifacts,
            lr=args.lr,
            optimizers=optimizers,
            supervised_weight=args.supervised_weight,
            constraint_weight=args.constraint_weight,
            wfa_weight=args.wfa_weight,
            factor_weight=args.factor_weight,
            latent_weight=args.latent_weight,
            allowed_mass_weight=args.allowed_mass_weight,
            latent_diagnostics=args.latent_diagnostics,
        )
        print(f"Step {step + 1}: {format_loss_log(losses)}")
    print("After unconstrained:", _prediction_summary(artifacts))
    print("After DFA-constrained:", _constrained_summary(artifacts))
    print("Vocabulary:", artifacts.bundle.vocabulary.labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
