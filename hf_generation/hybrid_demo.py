"""Hybrid controller demo: large generator proposes, compact head reranks."""
from __future__ import annotations

import argparse

from domiknows.generation import GenerationCandidate, HybridController

try:
    from .learning_program import build_learning_program, make_optimizers, run_one_training_step
    from .loss_logging import format_loss_log, print_loss_log_note
    from .run_demo import build_demo
except ImportError:
    from learning_program import build_learning_program, make_optimizers, run_one_training_step
    from loss_logging import format_loss_log, print_loss_log_note
    from run_demo import build_demo


def run_hybrid_demo(
    *,
    prompt: str = "Once",
    steps: int = 3,
    num_candidates: int = 3,
    max_new_tokens: int = 4,
) -> dict:
    """Train a tiny compact head, then use it to rerank generator candidates."""
    graph, bundle, enforcement, dfa, adapter, tokenizer = build_demo()
    learning = build_learning_program(pad_size=max_new_tokens)
    optimizers = make_optimizers(learning, lr=0.05)
    losses = []
    for _step in range(int(steps)):
        losses.append(run_one_training_step(learning, optimizers=optimizers))

    controller = HybridController(
        generator=adapter,
        vocabulary=bundle.vocabulary,
        dfa=dfa,
        scorer_head=learning.model,
        enforcement=enforcement,
        tokenizer=tokenizer,
        constraints=enforcement.dfa_constraints,
    )

    ranked = controller.generate_verify_rerank(
        prompt,
        num_candidates,
        max_new_tokens=max_new_tokens,
        hard_decode=True,
        temperature=0.8,
        top_p=0.95,
        keep_rejected=True,
        explain=True,
    )

    rejected = GenerationCandidate(text=" The dog<eos>", source="manual_rejected")
    repair = controller.suggest_repair(rejected, prompt_ids=tokenizer(prompt, return_tensors="pt").input_ids)
    rejected_score = controller.score_candidate(
        tokenizer(prompt, return_tensors="pt").input_ids,
        rejected,
        explain=True,
    )
    risk = controller.predict_failure_risk(
        tokenizer(prompt, return_tensors="pt").input_ids,
        ranked[0].candidate.labels if ranked else [],
    )

    return {
        "graph": graph.name,
        "prompt": prompt,
        "losses": losses,
        "ranked": ranked,
        "rejected_score": rejected_score,
        "repair": repair,
        "risk": risk,
        "vocabulary": bundle.vocabulary.labels,
    }


def print_summary(summary: dict) -> None:
    """Print a compact human-readable summary."""
    print("Path: hybrid controller/scorer")
    print("  large generator proposes candidates; compact DomiKnowS head reranks and diagnoses")
    print("Prompt:", repr(summary["prompt"]))
    if summary["losses"]:
        print("Training losses:")
        print_loss_log_note()
        for index, losses in enumerate(summary["losses"], start=1):
            print(f"  step {index}: {format_loss_log(losses)}")
    print("\nRanked candidates:")
    for rank, item in enumerate(summary["ranked"], start=1):
        candidate = item.candidate
        score = item.score
        print(f"  {rank}. text={candidate.text!r} labels={candidate.labels}")
        print(
            "     "
            f"accepted={score.accepted} total={score.total:.4f} "
            f"head={score.head_logprob:.4f} risk={score.risk:.4f}"
        )
        if score.rejection:
            print("     rejection:", score.rejection)
    print("\nRejected candidate diagnostic:")
    print("  accepted:", summary["rejected_score"].accepted)
    print("  rejection:", summary["rejected_score"].rejection)
    print("  repair suggestions:", summary["repair"]["suggestions"])
    print("  next labels:", summary["repair"]["next_labels"])
    print("\nBest-candidate next-step risk:", round(float(summary["risk"]), 4))
    print("Vocabulary:", summary["vocabulary"])
    print("Graph:", summary["graph"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Once")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--num-candidates", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args(argv)
    print_summary(
        run_hybrid_demo(
            prompt=args.prompt,
            steps=args.steps,
            num_candidates=args.num_candidates,
            max_new_tokens=args.max_new_tokens,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
