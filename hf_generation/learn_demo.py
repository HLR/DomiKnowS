"""Run the Collie-style learning path for the HuggingFace generation example."""
from __future__ import annotations

import argparse

import torch
from domiknows.generation import constrained_label_greedy_decode

try:
    from .learning_program import build_learning_program, make_optimizers, run_one_training_step
except ImportError:
    from learning_program import build_learning_program, make_optimizers, run_one_training_step


def _prediction_summary(artifacts) -> str:
    target_ids = artifacts.sample_data["target_token_ids"][0][: artifacts.model.pad_size]
    labels = [artifacts.bundle.vocabulary.label_for_token_id(int(token_id)) for token_id in target_ids]
    if len(labels) < artifacts.model.pad_size:
        labels.extend([artifacts.bundle.vocabulary.eos_label] * (artifacts.model.pad_size - len(labels)))
    with torch.no_grad():
        log_probs = artifacts.model(
            None,
            artifacts.sample_data["instruction_tokens"],
            torch.tensor(labels, dtype=torch.long),
        )
    preds = [int(torch.argmax(row).item()) for row in log_probs]
    return f"labels={labels} preds={preds} accepted={artifacts.dfa.accepts(preds)}"


def _constrained_decode_summary(artifacts) -> str:
    prompt_ids = artifacts.sample_data["instruction_tokens"]
    result = constrained_label_greedy_decode(
        artifacts.model,
        prompt_ids,
        artifacts.bundle.vocabulary,
        artifacts.dfa,
        max_new_tokens=artifacts.model.pad_size,
    )
    prompt_len = int(prompt_ids.shape[-1])
    generated_token_ids = result.token_ids[prompt_len:]
    text = artifacts.tokenizer.decode(generated_token_ids)
    return (
        f"labels={result.labels} token_ids={generated_token_ids} "
        f"text={text!r} accepted={result.accepted}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-hf", action="store_true", help="use a real frozen HuggingFace backbone")
    parser.add_argument("--model", default="roneneldan/TinyStories-1M", help="HuggingFace model id")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--pad-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--supervised-weight", type=float, default=3.0)
    parser.add_argument("--constraint-weight", type=float, default=1.0)
    parser.add_argument("--constrained-decoding", action="store_true")
    parser.add_argument("--show-transformers-load-report", action="store_true")
    args = parser.parse_args(argv)

    artifacts = build_learning_program(
        real_hf=args.real_hf,
        model_name=args.model,
        pad_size=args.pad_size,
        constrained_decoding=args.constrained_decoding,
        quiet_transformers=not args.show_transformers_load_report,
    )

    print("Learning path: supervised compact-label loss + DomiKnowS PMD constraint loss")
    print("Enforcement path: graph constraints -> DFA mask for final learned-head decoding")
    print("Trainable parameters:", artifacts.model.trainable_parameter_names())
    print("Before:", _prediction_summary(artifacts))
    optimizers = make_optimizers(artifacts, lr=args.lr)
    for step in range(args.steps):
        losses = run_one_training_step(
            artifacts,
            lr=args.lr,
            optimizers=optimizers,
            supervised_weight=args.supervised_weight,
            constraint_weight=args.constraint_weight,
        )
        print(f"Step {step + 1}: {losses}")
    print("After unconstrained:", _prediction_summary(artifacts))
    print("After DFA-constrained:", _constrained_decode_summary(artifacts))
    print("Vocabulary:", artifacts.bundle.vocabulary.labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
