"""Compare the three compact-label learners on the one-constraint demo.

Trains each of ``discrete-hmm``, ``graph-hmm``, and ``energy`` on the same
generator stream and then evaluates them on each demo prompt (AB / CD /
short).  Reports the side-by-side metrics that matter for picking a learner:

* training time and inference time
* trainable parameter count
* DFA acceptance of the greedy output (does it satisfy ``atMostAL(B, 1)``?)
* prompt-conditioning: are the three prompts producing distinct outputs?
* model log-score of the greedy sequence
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Iterable

import torch

try:
    from .learning_program import build_learning_program
    from .stream_generator import PROMPT_ORDER, prompt_spec
    from .utils import (
        AdamWithGradSnapshot,
        _enable_domiknows_production_logging,
        reset_optimizer_grad_snapshot,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from learning_program import build_learning_program
    from stream_generator import PROMPT_ORDER, prompt_spec
    from utils import (
        AdamWithGradSnapshot,
        _enable_domiknows_production_logging,
        reset_optimizer_grad_snapshot,
    )

_enable_domiknows_production_logging()


LEARNERS = ("discrete-hmm", "graph-hmm", "energy")


@dataclass
class PromptResult:
    """Greedy inference result for one (learner, prompt) pair."""

    prompt_name: str
    labels: tuple[int, ...]
    symbols: tuple[str, ...]
    log_score: float
    dfa_accepted: bool
    inference_seconds: float


@dataclass
class LearnerReport:
    """All metrics collected for one learner."""

    learner: str
    trainable_param_count: int
    trainable_tensor_count: int
    train_seconds: float
    prompt_results: dict[str, PromptResult] = field(default_factory=dict)

    @property
    def prompts_distinct(self) -> bool:
        """Are the greedy outputs different across the demo prompts?"""
        outputs = {result.labels for result in self.prompt_results.values()}
        return len(outputs) == len(self.prompt_results)

    @property
    def all_dfa_accepted(self) -> bool:
        return all(result.dfa_accepted for result in self.prompt_results.values())


def _count_trainable(model: torch.nn.Module) -> tuple[int, int]:
    """Return (parameter element count, distinct tensor count) for trainable parameters."""
    tensors = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return sum(parameter.numel() for parameter in tensors), len(tensors)


def _train_one_learner(
    learner: str,
    *,
    steps: int,
    stream_count: int,
    seed: int,
    pad_size: int,
    lr: float,
    beta: float,
) -> tuple[object, LearnerReport]:
    """Build the learning program for *learner*, run training, return artifacts and metrics."""
    artifacts = build_learning_program(
        learner=learner,
        stream_count=stream_count,
        stream_seed=seed,
        pad_size=pad_size,
        beta=beta,
    )
    param_count, tensor_count = _count_trainable(artifacts.model)

    start = time.perf_counter()
    for step in range(max(0, steps)):
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
    train_seconds = time.perf_counter() - start

    report = LearnerReport(
        learner=learner,
        trainable_param_count=param_count,
        trainable_tensor_count=tensor_count,
        train_seconds=train_seconds,
    )
    return artifacts, report


def _evaluate_prompt(artifacts, prompt_name: str) -> PromptResult:
    """Greedy-decode under the trained model for one demo prompt."""
    prompt = prompt_spec(prompt_name)
    prompt_token_id = int(prompt["token_id"])

    start = time.perf_counter()
    inference_result = artifacts.model.greedy_label_inference(
        artifacts.bundle.vocabulary,
        [prompt_token_id],
        max_new_tokens=artifacts.model.pad_size,
    )
    inference_seconds = time.perf_counter() - start

    labels = tuple(int(label) for label in inference_result.labels)
    symbols = tuple(
        artifacts.bundle.vocabulary.token_for_label(label) for label in labels
    )
    raw_score = getattr(inference_result, "score", None)
    log_score = float(raw_score) if raw_score is not None else float("nan")
    dfa_accepted = bool(artifacts.dfa.accepts(labels))
    return PromptResult(
        prompt_name=prompt_name,
        labels=labels,
        symbols=symbols,
        log_score=log_score,
        dfa_accepted=dfa_accepted,
        inference_seconds=inference_seconds,
    )


def _evaluate_learner(
    artifacts,
    report: LearnerReport,
    *,
    prompts: Iterable[str],
) -> LearnerReport:
    """Fill *report* with per-prompt greedy inference results."""
    for prompt_name in prompts:
        report.prompt_results[prompt_name] = _evaluate_prompt(artifacts, prompt_name)
    return report


def _format_seconds(value: float) -> str:
    if value < 1.0:
        return f"{value * 1000:.1f} ms"
    return f"{value:.2f} s"


def _format_symbols(symbols: tuple[str, ...], *, limit: int = 12) -> str:
    cleaned = [str(symbol) for symbol in symbols]
    if len(cleaned) <= limit:
        return " ".join(cleaned) if cleaned else "<empty>"
    head = " ".join(cleaned[:limit])
    return f"{head} ... (+{len(cleaned) - limit} more)"


def _print_header(args) -> None:
    print("Performance comparison across compact-label learners")
    print(f"  Constraint: token B may appear at most once")
    print(
        f"  Training args: steps={args.steps}, stream_count={args.stream_count}, "
        f"pad_size={args.pad_size}, lr={args.lr:g}, beta={args.beta:g}, seed={args.seed}"
    )
    print(f"  Evaluating prompts: {', '.join(args.prompts)}")
    print()


def _print_per_learner_summary(report: LearnerReport) -> None:
    print(f"=== Learner: {report.learner} ===")
    print(
        f"  trainable params : {report.trainable_param_count} across "
        f"{report.trainable_tensor_count} tensors"
    )
    print(
        f"  training time    : {_format_seconds(report.train_seconds)} "
        f"(over the configured number of stream batches)"
    )
    for prompt_name, result in report.prompt_results.items():
        info = prompt_spec(prompt_name)
        marker = "OK" if result.dfa_accepted else "FAIL"
        print(
            f"  prompt {prompt_name:>5} ({info['text']}): "
            f"{_format_symbols(result.symbols)}  "
            f"[score {result.log_score:7.3f}, dfa {marker}, "
            f"infer {_format_seconds(result.inference_seconds)}]"
        )
    differentiated = "yes" if report.prompts_distinct else "no"
    print(f"  prompt outputs distinct across all evaluated prompts: {differentiated}")
    print()


def _print_comparison_table(reports: list[LearnerReport], prompts: list[str]) -> None:
    print("Side-by-side comparison")
    columns = ["learner", "train_s", "params", "distinct", "dfa_pass"] + [
        f"infer_{prompt}_ms" for prompt in prompts
    ]
    rows = []
    for report in reports:
        cells = [
            report.learner,
            f"{report.train_seconds:.2f}",
            str(report.trainable_param_count),
            "yes" if report.prompts_distinct else "no",
            f"{sum(1 for r in report.prompt_results.values() if r.dfa_accepted)}/{len(report.prompt_results)}",
        ]
        for prompt in prompts:
            result = report.prompt_results.get(prompt)
            cells.append(f"{result.inference_seconds * 1000:.1f}" if result else "-")
        rows.append(cells)

    widths = [max(len(column), *(len(row[index]) for row in rows)) for index, column in enumerate(columns)]
    sep = "  "

    def _fmt(row: list[str]) -> str:
        return sep.join(value.ljust(width) for value, width in zip(row, widths))

    print(_fmt(columns))
    print(_fmt(["-" * width for width in widths]))
    for row in rows:
        print(_fmt(row))
    print()


def _print_prompt_outputs(reports: list[LearnerReport], prompts: list[str]) -> None:
    print("Greedy outputs per learner / prompt")
    for prompt in prompts:
        info = prompt_spec(prompt)
        print(f"  prompt {prompt} ({info['text']}):")
        for report in reports:
            result = report.prompt_results.get(prompt)
            if result is None:
                print(f"    {report.learner:>13}: (not evaluated)")
                continue
            marker = "OK" if result.dfa_accepted else "FAIL"
            print(
                f"    {report.learner:>13}: {_format_symbols(result.symbols)} "
                f"[dfa {marker}, log_score {result.log_score:7.3f}]"
            )
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--learners",
        nargs="+",
        default=list(LEARNERS),
        choices=LEARNERS,
        help="Subset of learners to compare; defaults to all three.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=list(PROMPT_ORDER),
        choices=PROMPT_ORDER,
        help="Demo prompts to evaluate after training.",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--stream-count", type=int, default=4)
    parser.add_argument("--pad-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--beta", type=float, default=0.3)
    args = parser.parse_args(argv)
    if args.stream_count <= 0:
        parser.error("--stream-count must be positive")
    if args.pad_size < 2:
        parser.error("--pad-size must be at least 2")

    _print_header(args)

    reports: list[LearnerReport] = []
    for learner in args.learners:
        artifacts, report = _train_one_learner(
            learner,
            steps=args.steps,
            stream_count=args.stream_count,
            seed=args.seed,
            pad_size=args.pad_size,
            lr=args.lr,
            beta=args.beta,
        )
        _evaluate_learner(artifacts, report, prompts=args.prompts)
        reports.append(report)
        _print_per_learner_summary(report)

    _print_comparison_table(reports, args.prompts)
    _print_prompt_outputs(reports, args.prompts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
