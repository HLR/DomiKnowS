"""HMM distillation decoder baseline for HF generation decoders.

This demo keeps the baseline offline by using the tiny HuggingFace-shaped
``MockCausalLM`` from this task.  It distills the mock small-LLM next-token
distribution into a compact HMM head, then compares decoder accuracy across:

* raw unconstrained small-LLM greedy decoding;
* DFA hard decoding over small-LLM logits;
* compact learner + DFA product decoding;
* strict HMM + DFA product decoding with future-success lookahead.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

import torch

from domiknows.generation import (
    HMMGenerationHead,
    HuggingFaceGenerationAdapter,
    HybridController,
    TokenVocabulary,
    eos_closure_dfa,
    forbidden_token_dfa,
    max_non_eos_dfa,
    product_dfa,
    required_token_dfa,
    union_dfa,
)
from domiknows.generation.dfa.decoder import ConstrainedGenerationResult

try:
    from .mock_hf import MockCausalLM, MockTokenizer
except ImportError:
    from mock_hf import MockCausalLM, MockTokenizer


VOCAB = ["<eos>", " The", " cat", " mat", " dog"]


@dataclass(frozen=True)
class DecoderReport:
    """One decoded output and its constraint-accuracy flags."""

    mode: str
    prompt: str
    labels: list[int]
    token_ids: list[int]
    text: str
    accepted: bool
    has_required_cat: bool
    avoids_forbidden_dog: bool


def _prompt_ids(tokenizer, prompt: str) -> torch.Tensor:
    return tokenizer(prompt, return_tensors="pt").input_ids


def _full_ids(prompt_ids: torch.Tensor, vocabulary, prefix_labels: Sequence[int]) -> list[int]:
    ids = [int(item) for item in prompt_ids.reshape(-1).tolist()]
    for label in prefix_labels:
        ids.append(vocabulary.token_id_for_label(int(label)))
    return ids


def _teacher_label_logits(model, vocabulary, full_ids: Sequence[int]) -> torch.Tensor:
    ids = torch.tensor([list(int(item) for item in full_ids)], dtype=torch.long)
    output = model(ids)
    token_logits = output.logits[0, -1, :]
    label_logits = token_logits.new_full((vocabulary.label_count,), -20.0)
    for label in range(vocabulary.label_count):
        try:
            token_id = vocabulary.token_id_for_label(label)
        except ValueError:
            continue
        if 0 <= token_id < token_logits.numel():
            label_logits[label] = token_logits[token_id]
    return label_logits


def label_token_id_map(vocabulary) -> tuple[int | None, ...]:
    """Map compact generation labels back to raw tokenizer ids when possible."""
    token_ids: list[int | None] = []
    for label in range(vocabulary.label_count):
        try:
            token_ids.append(vocabulary.token_id_for_label(label))
        except ValueError:
            token_ids.append(None)
    return tuple(token_ids)


def build_eai_demo():
    """Build the offline mock LM, vocabulary, adapter, and equivalent DFA."""
    tokenizer = MockTokenizer()
    vocabulary = TokenVocabulary(VOCAB, eos_token="<eos>", tokenizer=tokenizer)
    dfa = product_dfa(
        [
            eos_closure_dfa(vocabulary),
            max_non_eos_dfa(vocabulary, 3),
            required_token_dfa(vocabulary, " cat"),
            forbidden_token_dfa(vocabulary, " dog"),
            union_dfa(
                [
                    required_token_dfa(vocabulary, " The"),
                    required_token_dfa(vocabulary, " mat"),
                ]
            ),
        ]
    )
    adapter = HuggingFaceGenerationAdapter(MockCausalLM(), tokenizer, vocabulary)
    return vocabulary, dfa, adapter, tokenizer


def _distillation_prefixes(vocabulary) -> list[list[int]]:
    the = vocabulary.label_for_token(" The")
    cat = vocabulary.label_for_token(" cat")
    mat = vocabulary.label_for_token(" mat")
    dog = vocabulary.label_for_token(" dog")
    eos = vocabulary.eos_label
    return [
        [],
        [the],
        [the, cat],
        [the, cat, mat],
        [cat],
        [cat, mat],
        [mat],
        [mat, cat],
        [dog],
        [dog, cat],
        [cat, eos],
    ]


def train_distilled_hmm(
    *,
    adapter,
    tokenizer,
    vocabulary,
    prompts: Sequence[str],
    steps: int = 40,
    lr: float = 0.15,
    state_count: int = 4,
    max_new_tokens: int = 4,
    temperature: float = 1.0,
    random_seed: int = 7,
) -> tuple[HMMGenerationHead, list[float]]:
    """Distill small-LLM next-label distributions into an HMMGenerationHead."""
    torch.manual_seed(int(random_seed))
    head = HMMGenerationHead(
        label_count=vocabulary.label_count,
        state_count=state_count,
        pad_size=max_new_tokens,
        label_to_token_id=label_token_id_map(vocabulary),
        trainable=True,
        random_seed=random_seed,
    )
    optimizer = torch.optim.Adam(head.parameters(), lr=float(lr))
    prefixes = _distillation_prefixes(vocabulary)
    losses: list[float] = []

    for _step in range(int(steps)):
        optimizer.zero_grad()
        total = head.emission_logits.new_zeros(())
        count = 0
        for prompt in prompts:
            prompt_ids = _prompt_ids(tokenizer, prompt)
            for prefix in prefixes:
                ids = _full_ids(prompt_ids, vocabulary, prefix)
                teacher = _teacher_label_logits(adapter.model, vocabulary, ids)
                teacher_probs = torch.softmax(teacher / float(temperature), dim=-1)
                student = head.next_label_logits(torch.tensor([ids], dtype=torch.long))
                log_student = torch.log_softmax(student / float(temperature), dim=-1)
                total = total + torch.nn.functional.kl_div(log_student, teacher_probs, reduction="batchmean")
                count += 1
        loss = total / max(1, count)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    return head, losses


def _raw_greedy_decode(adapter, prompt_ids: torch.Tensor, vocabulary, max_new_tokens: int) -> ConstrainedGenerationResult:
    ids = [int(item) for item in prompt_ids.reshape(-1).tolist()]
    prompt_len = len(ids)
    labels: list[int] = []
    for _step in range(int(max_new_tokens)):
        logits = adapter.model(torch.tensor([ids], dtype=torch.long)).logits[0, -1, :]
        token_id = int(torch.argmax(logits).item())
        ids.append(token_id)
        labels.append(vocabulary.label_for_token_id(token_id))
        if labels[-1] == vocabulary.eos_label:
            break
    return ConstrainedGenerationResult(
        token_ids=ids,
        labels=labels,
        final_state=None,
        accepted=False,
        score=None,
        scores=None,
    )


def _report_from_result(mode: str, prompt: str, result, prompt_ids: torch.Tensor, tokenizer, vocabulary, dfa) -> DecoderReport:
    prompt_len = int(prompt_ids.shape[-1])
    token_ids = list(result.token_ids[prompt_len:])
    labels = list(result.labels)
    cat_label = vocabulary.label_for_token(" cat")
    dog_label = vocabulary.label_for_token(" dog")
    accepted = bool(getattr(result, "accepted", False)) and dfa.accepts(labels)
    return DecoderReport(
        mode=mode,
        prompt=prompt,
        labels=labels,
        token_ids=token_ids,
        text=tokenizer.decode(token_ids),
        accepted=accepted,
        has_required_cat=cat_label in labels,
        avoids_forbidden_dog=dog_label not in labels,
    )


def _report_from_ranked(mode: str, prompt: str, ranked, prompt_ids: torch.Tensor, tokenizer, vocabulary, dfa) -> DecoderReport:
    if not ranked:
        return DecoderReport(mode, prompt, [], [], "", False, False, True)
    return _report_from_result(mode, prompt, ranked[0].candidate.raw, prompt_ids, tokenizer, vocabulary, dfa)


def _report_from_hmm_dfa_decode(mode: str, prompt: str, results, tokenizer, vocabulary, dfa) -> DecoderReport:
    if not results:
        return DecoderReport(mode, prompt, [], [], "", False, False, True)
    result = results[0]
    labels = list(result.labels)
    token_ids = list(result.token_ids)
    cat_label = vocabulary.label_for_token(" cat")
    dog_label = vocabulary.label_for_token(" dog")
    accepted = bool(result.accepted) and dfa.accepts(labels)
    return DecoderReport(
        mode=mode,
        prompt=prompt,
        labels=labels,
        token_ids=token_ids,
        text=tokenizer.decode(token_ids),
        accepted=accepted,
        has_required_cat=cat_label in labels,
        avoids_forbidden_dog=dog_label not in labels,
    )


def _accuracy(reports: Sequence[DecoderReport]) -> dict[str, float]:
    total = max(1, len(reports))
    return {
        "accepted_accuracy": sum(1 for item in reports if item.accepted) / total,
        "cat_accuracy": sum(1 for item in reports if item.has_required_cat) / total,
        "dog_avoidance": sum(1 for item in reports if item.avoids_forbidden_dog) / total,
    }


def evaluate_decoders(
    *,
    prompts: Sequence[str],
    adapter,
    tokenizer,
    vocabulary,
    dfa,
    hmm_head: HMMGenerationHead,
    max_new_tokens: int = 4,
) -> dict[str, object]:
    """Run the baseline decoders and return per-mode accuracy reports."""
    controller = HybridController(
        generator=adapter,
        vocabulary=vocabulary,
        dfa=dfa,
        scorer_head=hmm_head,
        tokenizer=tokenizer,
    )
    reports: dict[str, list[DecoderReport]] = {
        "raw_lm_greedy": [],
        "dfa_greedy": [],
        "dfa_beam": [],
        "dfa_sample": [],
        "product_compact_learner_dfa": [],
        "product_hmm_dfa": [],
    }

    for index, prompt in enumerate(prompts):
        prompt_ids = _prompt_ids(tokenizer, prompt)
        raw = _raw_greedy_decode(adapter, prompt_ids, vocabulary, max_new_tokens)
        raw = ConstrainedGenerationResult(
            token_ids=raw.token_ids,
            labels=raw.labels,
            final_state=None,
            accepted=dfa.accepts(raw.labels),
            score=None,
            scores=None,
        )
        reports["raw_lm_greedy"].append(_report_from_result("raw_lm_greedy", prompt, raw, prompt_ids, tokenizer, vocabulary, dfa))
        reports["dfa_greedy"].append(
            _report_from_result(
                "dfa_greedy",
                prompt,
                adapter.constrained_greedy(prompt_ids, dfa, max_new_tokens=max_new_tokens),
                prompt_ids,
                tokenizer,
                vocabulary,
                dfa,
            )
        )
        reports["dfa_beam"].append(
            _report_from_result(
                "dfa_beam",
                prompt,
                adapter.constrained_beam_search(
                    prompt_ids,
                    dfa,
                    max_new_tokens=max_new_tokens,
                    beam_size=3,
                    early_stopping=False,
                ),
                prompt_ids,
                tokenizer,
                vocabulary,
                dfa,
            )
        )
        reports["dfa_sample"].append(
            _report_from_result(
                "dfa_sample",
                prompt,
                adapter.constrained_sample(
                    prompt_ids,
                    dfa,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                    top_p=0.95,
                    generator=torch.Generator().manual_seed(11 + index),
                ),
                prompt_ids,
                tokenizer,
                vocabulary,
                dfa,
            )
        )
        ranked = controller.generate_verify_rerank(
            prompt,
            1,
            decode_strategy="product_compact_learner_dfa",
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        reports["product_compact_learner_dfa"].append(
            _report_from_ranked(
                "product_compact_learner_dfa",
                prompt,
                ranked,
                prompt_ids,
                tokenizer,
                vocabulary,
                dfa,
            )
        )
        reports["product_hmm_dfa"].append(
            _report_from_hmm_dfa_decode(
                "product_hmm_dfa",
                prompt,
                controller.decode_hmm_dfa(
                    prompt,
                    search="beam",
                    num_return_sequences=1,
                    beam_size=4,
                    lookahead_weight=1.0,
                    max_new_tokens=max_new_tokens,
                ),
                tokenizer,
                vocabulary,
                dfa,
            )
        )

    return {
        "reports": reports,
        "accuracy": {mode: _accuracy(items) for mode, items in reports.items()},
    }


def run_hmm_distillation_decoder_baseline(
    *,
    prompts: Sequence[str] = ("Once", "Story", "A tale"),
    steps: int = 40,
    lr: float = 0.15,
    state_count: int = 4,
    max_new_tokens: int = 4,
) -> dict[str, object]:
    """Train the distilled HMM and evaluate decoder baseline accuracy."""
    vocabulary, dfa, adapter, tokenizer = build_eai_demo()
    hmm_head, losses = train_distilled_hmm(
        adapter=adapter,
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        prompts=prompts,
        steps=steps,
        lr=lr,
        state_count=state_count,
        max_new_tokens=max_new_tokens,
    )
    evaluation = evaluate_decoders(
        prompts=prompts,
        adapter=adapter,
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        dfa=dfa,
        hmm_head=hmm_head,
        max_new_tokens=max_new_tokens,
    )
    return {
        "prompts": tuple(prompts),
        "losses": losses,
        "hmm_head": hmm_head,
        "vocabulary": vocabulary.labels,
        **evaluation,
    }


def print_summary(summary: dict[str, object]) -> None:
    """Print a compact accuracy table."""
    print("HMM distillation decoder baseline")
    print("  teacher: offline small HuggingFace-shaped MockCausalLM")
    print("  student: compact HMMGenerationHead distilled from teacher label logits")
    losses = summary["losses"]
    if losses:
        print(f"Distillation loss: first={losses[0]:.4f} last={losses[-1]:.4f} steps={len(losses)}")
    print("\nDecoder accuracy")
    print("  mode                         accepted  cat  no_dog")
    for mode, metrics in summary["accuracy"].items():
        print(
            f"  {mode:<28} "
            f"{metrics['accepted_accuracy']:.2f}      "
            f"{metrics['cat_accuracy']:.2f}  "
            f"{metrics['dog_avoidance']:.2f}"
        )
    print("\nExample outputs")
    for mode, reports in summary["reports"].items():
        first = reports[0]
        print(f"  {mode:<28} labels={first.labels} text={first.text!r} accepted={first.accepted}")
    print("Vocabulary:", summary["vocabulary"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--state-count", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--prompts", nargs="+", default=["Once", "Story", "A tale"])
    args = parser.parse_args(argv)
    print_summary(
        run_hmm_distillation_decoder_baseline(
            prompts=args.prompts,
            steps=args.steps,
            lr=args.lr,
            state_count=args.state_count,
            max_new_tokens=args.max_new_tokens,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
