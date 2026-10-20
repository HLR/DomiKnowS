"""Label-free HMM lookahead for the standard VLABench protocol.

CLAMP-style test-time adaptation without labels: the frozen VLM is sampled on the benchmark's
own inputs (two images and the instruction under the official prompt, no entity names), the
sampled plan tokens train a small HMM, and the HMM's lookahead steers decoding under the
grammar DFA. No reference plan is read while sampling or fitting.

Labels enter in one place only: the lookahead weight is chosen on the 450 validation episodes
of the seed-42 split by their upstream total_score, and the summary says so. The 450 test
episodes of that split are never used for sampling, fitting or selection, so they give a clean
held-out figure next to the all-episodes figure.

Run ``python -m test_regr.VLABenchAgentInterface.official_hmm --help``. Every stage resumes.

Known limit: sampling and fitting are cheap (an hour and a few minutes on one GPU), but the
lookahead table is a token-level dynamic programme over every state of the grammar DFA. With 64
ids and pretty-printed JSON that is millions of (state, depth) entries already at depth 6, and at
the default depth of 50 it ran out of memory (about 93 GB). Depth 6 is too short to steer a plan of
about 60 tokens, so decoding with the lookahead needs a smaller language (fewer ids, compact JSON)
than the grammar-DFA row uses, and its result would not be comparable with that row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Sequence

try:
    from .dataset import deterministic_split
    from .official_protocol import (
        DFA_ENTITY_CAP, OfficialExample, build_ti_list, load_official_examples, load_official_prompt,
        make_grammar_dfa_generator, official_report, run_official_protocol,
    )
except ImportError:  # pragma: no cover - flat-module execution
    from dataset import deterministic_split
    from official_protocol import (
        DFA_ENTITY_CAP, OfficialExample, build_ti_list, load_official_examples, load_official_prompt,
        make_grammar_dfa_generator, official_report, run_official_protocol,
    )

WEIGHTS = (0.0, 0.1, 0.5, 1.0)
SAMPLING = {"temperature": 0.8, "top_p": 0.95}


def split_examples(examples: Sequence[OfficialExample], seed: int = 42) -> dict[str, list[OfficialExample]]:
    """The repository's seed-42 80/10/10 split (episodes are in the same sorted order)."""
    return deterministic_split(list(examples), seed=seed)


def canonical_plan_text(text: str) -> tuple[str | None, str | None]:
    """The plan in a sampled reply, with quoted ids as integers and four-space indentation.

    The DFA allows integer ids only, so teacher samples that quote them ("6") are rewritten
    to the form the constrained decoder emits before they train the HMM.
    """
    plan = text.split("```")[0].strip()
    try:
        value = json.loads(plan)
    except ValueError as error:
        return None, f"not JSON: {error}"
    if not isinstance(value, list) or not value:
        return None, "not a non-empty JSON array"
    for operation in value:
        params = operation.get("params") if isinstance(operation, dict) else None
        if isinstance(params, dict):
            for key, argument in params.items():
                if isinstance(argument, str) and argument.strip().isdigit():
                    params[key] = int(argument.strip())
    return json.dumps(value, indent=4), None


def plan_tokens(text: str, tokenizer: Any) -> tuple[list[int] | None, str | None]:
    """Tokens of the canonical plan in a sampled reply, ending in EOS."""
    plan, error = canonical_plan_text(text)
    if plan is None:
        return None, error
    return list(tokenizer(plan, add_special_tokens=False).input_ids) + [int(tokenizer.eos_token_id)], None


def _append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _done(path: Path) -> set[str]:
    return {json.loads(line)["episode_id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()} if path.exists() else set()


def sample_corpus(planner: Any, examples: Sequence[OfficialExample], output_dir: str | Path, *, seed: int = 42,
                  max_new_tokens: int = 512, batch_size: int = 8, progress=print) -> Path:
    """Sample one unconstrained reply per episode from the frozen model and keep the plan tokens."""
    import torch
    from PIL import Image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kept, rejected = output_dir / "episodes.jsonl", output_dir / "rejected.jsonl"
    seen = _done(kept) | _done(rejected)
    prompt = load_official_prompt()
    tokenizer, previous_padding = planner.tokenizer, planner.tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        pending = [example for example in examples if example.episode_id not in seen]
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            texts, images = [], []
            try:
                for example in batch:
                    content = []
                    for kind, value in build_ti_list(prompt, example):
                        if kind == "text":
                            content.append({"type": "text", "text": value})
                        else:
                            image = Image.open(value).convert("RGB")
                            images.append(image)
                            content.append({"type": "image", "image": image})
                    texts.append(planner.processor.apply_chat_template(
                        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
                        enable_thinking=False) + "```json\n")
                inputs = planner.processor(text=texts, images=images, padding=True, return_tensors="pt").to(
                    next(planner.model.parameters()).device)
                length = inputs.input_ids.shape[1]
                torch.manual_seed(seed + len(seen) + offset)
                torch.cuda.manual_seed_all(seed + len(seen) + offset)
                generated = planner.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                                   use_cache=True, **SAMPLING)[:, length:]
            finally:
                for image in images:
                    image.close()
            for example, row in zip(batch, generated.tolist()):
                text = tokenizer.decode(row, skip_special_tokens=True)
                tokens, error = plan_tokens(text, tokenizer)
                if tokens is None:
                    _append(rejected, {"episode_id": example.episode_id, "raw": text, "parse_error": error})
                else:
                    _append(kept, {"episode_id": example.episode_id, "sampled": text.split("```")[0].strip(),
                                   "raw": canonical_plan_text(text)[0], "tokens": tokens, "parse_error": None})
            progress(f"[official-hmm] corpus {output_dir.name}: {min(offset + batch_size, len(pending))}/{len(pending)}")
    finally:
        tokenizer.padding_side = previous_padding
    return output_dir


def finalize_corpus(output_dir: str | Path, planner: Any, model_name: str) -> None:
    """Write the identity files the HMM fit checks (episode ids and teacher tokenizer)."""
    output_dir = Path(output_dir)
    ids = sorted(_done(output_dir / "episodes.jsonl"))
    (output_dir / "settings.json").write_text(json.dumps({"mode": "corpus", "episode_ids": ids, "sampling": SAMPLING,
                                                          "protocol": "official-prompt, no entity names"}, indent=2), encoding="utf-8")
    (output_dir / "model_provenance.json").write_text(json.dumps(
        {"tokenizer_sha256": planner.checksum, "model": model_name}, indent=2), encoding="utf-8")


def load_hmm(path: str | Path, planner: Any, device: str = "cuda") -> tuple[Any, str]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["tokenizer_sha256"] != planner.checksum:
        raise ValueError("HMM was fitted for a different tokenizer")
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return tuple(payload[key].to(device) for key in ("initial", "transition", "emission")), digest


def decoding_label(hmm_sha256: str | None, weight: float) -> str:
    base = f"grammar-dfa:{DFA_ENTITY_CAP}"
    return base if hmm_sha256 is None or weight <= 0 else f"{base}+hmm:{hmm_sha256[:12]}:w{weight:g}"


def heldout_report(final_scores: str | Path, test_ids: set[str]) -> dict[str, Any]:
    """Upstream-style report over the held-out test episodes only (no sampling, fitting or selection used them)."""
    rows = [row for row in map(json.loads, Path(final_scores).read_text(encoding="utf-8").splitlines()) if row["episode_id"] in test_ids]
    report = official_report(rows)
    return {"episodes": len(rows), **{key: report[key] for key in (
        "official_total_score_by_dimension", "official_dimension_examples", "dimension_mean_of_total_score",
        "official_mapped_examples", "unmapped_examples")}}


def run(args: argparse.Namespace) -> dict[str, Any]:
    from test_regr.VLABenchAgentInterface.native_planning import FrozenNativePlanner
    from test_regr.VLABenchAgentInterface.planning_experiment import fit_hmm

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    examples = load_official_examples(args.planning_dir)
    splits = split_examples(examples, args.seed)
    planner = FrozenNativePlanner.load(args.model, load_in_4bit=args.load_in_4bit)
    for name in ("train", "validation"):
        directory = root / f"corpus_{name}"
        sample_corpus(planner, splits[name], directory, seed=args.seed, batch_size=args.batch_size)
        finalize_corpus(directory, planner, str(args.model))
    checkpoint = root / "hmm_model" / "best.pt"
    if not checkpoint.exists():
        fit_hmm(argparse.Namespace(corpus=root / "corpus_train", validation_corpus=root / "corpus_validation",
                                   output=root / "hmm_model", vocab_size=args.vocab_size))
    hmm, digest = load_hmm(checkpoint, planner)

    def evaluate(stage: str, items: Sequence[OfficialExample], weight: float, standard: bool) -> dict[str, Any]:
        summary_path = root / stage / "summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        generate = make_grammar_dfa_generator(planner, args.max_new_tokens, hmm=hmm, weight=weight)
        return run_official_protocol(items, generate, output_dir=root / stage, model_name=str(args.model), seed=args.seed,
                                     standard=standard, decoding=decoding_label(digest, weight))

    selection_path = root / "hmm_selection.json"
    if not selection_path.exists():
        candidates = []
        for weight in WEIGHTS:
            summary = evaluate(f"select_w{weight:g}", splits["validation"], weight, False)
            candidates.append({"weight": weight, "validation_mean_total_score": summary["dimension_mean_of_total_score"]})
        best = max(candidates, key=lambda item: (item["validation_mean_total_score"], -item["weight"]))
        selection_path.write_text(json.dumps({
            "candidates": candidates, "selected": best, "hmm_sha256": digest,
            "labels_used": "validation split reference plans, only to choose the lookahead weight",
        }, indent=2), encoding="utf-8")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["hmm_sha256"] != digest:
        raise ValueError("HMM checkpoint changed after weight selection")
    weight = selection["selected"]["weight"]
    final = evaluate("final", examples, weight, True)
    heldout = heldout_report(root / "final" / "final_scores.jsonl", {example.episode_id for example in splits["test"]})
    (root / "final" / "heldout_test_report.json").write_text(json.dumps(heldout, indent=2, sort_keys=True), encoding="utf-8")
    return {"selected_weight": weight, "final": final, "heldout": heldout}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planning-dir", required=True, help="vlm_evaluation_v1.0 style folder")
    parser.add_argument("--model", required=True, help="local Qwen3-VL-8B-Instruct folder")
    parser.add_argument("--output", required=True, help="resumable working folder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    result = run(parser.parse_args(argv))
    print(json.dumps({key: value for key, value in result.items() if key == "selected_weight"}), flush=True)


if __name__ == "__main__":
    main()
