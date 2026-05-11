"""Compare non-prompt and prompt-conditioned automata heads."""
from __future__ import annotations

import argparse

import torch

try:
    from .loss_logging import format_loss_log, print_loss_log_note
    from .prompt_automata_program import (
        build_prompt_automata_learning_program,
        constrained_decode,
        make_optimizers,
        run_one_prompt_automata_training_step,
        target_labels_for_sample,
    )
except ImportError:
    from loss_logging import format_loss_log, print_loss_log_note
    from prompt_automata_program import (
        build_prompt_automata_learning_program,
        constrained_decode,
        make_optimizers,
        run_one_prompt_automata_training_step,
        target_labels_for_sample,
    )


def _prediction_summary(model, artifacts) -> str:
    labels = target_labels_for_sample(artifacts).tolist()
    with torch.no_grad():
        log_probs = model(
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
    parser.add_argument("--encoder", choices=("embedding", "frozen-backbone"), default="embedding")
    parser.add_argument("--real-hf", action="store_true")
    parser.add_argument("--model", default="roneneldan/TinyStories-1M")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--pad-size", type=int, default=4)
    parser.add_argument("--state-count", type=int, default=3)
    parser.add_argument("--dynamics-conditioning", choices=("none", "gated"), default="gated")
    parser.add_argument("--dynamics-experts", type=int, default=2)
    parser.add_argument("--step-dynamics-conditioning", choices=("none", "prefix-gated"), default="prefix-gated")
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--supervised-weight", type=float, default=3.0)
    parser.add_argument("--constraint-weight", type=float, default=1.0)
    parser.add_argument("--automata-weight", type=float, default=1.0)
    parser.add_argument("--latent-weight", type=float, default=0.0)
    parser.add_argument("--allowed-mass-weight", type=float, default=0.0)
    parser.add_argument("--latent-mode", choices=("marked", "auto", "marked-and-auto"), default="marked")
    parser.add_argument("--latent-diagnostics", action="store_true")
    parser.add_argument("--show-transformers-load-report", action="store_true")
    args = parser.parse_args(argv)

    artifacts = build_prompt_automata_learning_program(
        kind=args.kind,
        encoder_kind=args.encoder,
        real_hf=args.real_hf,
        model_name=args.model,
        pad_size=args.pad_size,
        state_count=args.state_count,
        dynamics_conditioning=args.dynamics_conditioning,
        dynamics_expert_count=args.dynamics_experts,
        step_dynamics_conditioning=args.step_dynamics_conditioning,
        trainable=True,
        quiet_transformers=not args.show_transformers_load_report,
        latent_mode=args.latent_mode,
    )

    print(f"Prompt-conditioned automata path: {args.kind.upper()} head + {args.encoder} prompt encoder")
    print(f"Dynamics conditioning: {artifacts.dynamics_conditioning}")
    print(f"Step dynamics conditioning: {artifacts.step_dynamics_conditioning}")
    print("Dynamics weights:", _dynamics_weights_summary(artifacts))
    print("Step dynamics weights:", _step_dynamics_weights_summary(artifacts))
    print("Baseline non-prompt:", _prediction_summary(artifacts.baseline_model, artifacts))
    print("Trainable parameters:", artifacts.model.trainable_parameter_names())
    print("Before prompt-conditioned:", _prediction_summary(artifacts.model, artifacts))
    print_loss_log_note()
    optimizers = make_optimizers(artifacts, lr=args.lr)
    for step in range(args.steps):
        losses = run_one_prompt_automata_training_step(
            artifacts,
            lr=args.lr,
            optimizers=optimizers,
            supervised_weight=args.supervised_weight,
            constraint_weight=args.constraint_weight,
            automata_weight=args.automata_weight,
            latent_weight=args.latent_weight,
            allowed_mass_weight=args.allowed_mass_weight,
            latent_diagnostics=args.latent_diagnostics,
        )
        print(f"Step {step + 1}: {format_loss_log(losses)}")
    print("After prompt-conditioned:", _prediction_summary(artifacts.model, artifacts))
    print("After DFA-constrained:", _constrained_summary(artifacts))
    print("Vocabulary:", artifacts.bundle.vocabulary.labels)
    return 0


def _dynamics_weights_summary(artifacts) -> list[float]:
    with torch.no_grad():
        weights = artifacts.model.prompt_dynamics_weights(artifacts.sample_data["instruction_tokens"])
    return [round(float(value), 4) for value in weights]


def _step_dynamics_weights_summary(artifacts) -> list[list[float]]:
    labels = target_labels_for_sample(artifacts).tolist()
    prefix = []
    rows = []
    with torch.no_grad():
        for label in labels[: min(3, len(labels))]:
            weights = artifacts.model.step_dynamics_weights(artifacts.sample_data["instruction_tokens"], prefix)
            rows.append([round(float(value), 4) for value in weights])
            prefix.append(int(label))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
