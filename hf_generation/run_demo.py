"""Run the HuggingFace constrained generation demo.

The default mode is completely offline and uses ``mock_hf.MockCausalLM``.  Use
``--real-hf`` to load a real HuggingFace causal LM when dependencies and model
cache/network are available.
"""
from __future__ import annotations

import argparse

import torch

from domiknows.generation import (  # noqa: E402
    HuggingFaceGenerationAdapter,
    constraints_to_dfa,
    discover_generation_enforcement,
    mask_logits_for_dfa,
)

try:
    from .graph import EOS_TOKEN, VOCAB, build_generation_graph
    from .mock_hf import MockCausalLM, MockTokenizer
except ImportError:
    from graph import EOS_TOKEN, VOCAB, build_generation_graph
    from mock_hf import MockCausalLM, MockTokenizer


def load_backend(
    real_hf: bool = False,
    model_name: str = "roneneldan/TinyStories-1M",
    quiet_transformers: bool = True,
):
    """Return ``(tokenizer, model)`` for mock or real HuggingFace execution."""
    if not real_hf:
        return MockTokenizer(), MockCausalLM()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    if quiet_transformers:
        hf_logging.set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"
    return tokenizer, model


def generation_vocab_for_tokenizer(tokenizer, real_hf: bool = False) -> tuple[list[str], str]:
    """Return a demo vocabulary whose EOS token matches *tokenizer*."""
    if not real_hf:
        return list(VOCAB), EOS_TOKEN
    eos_token = tokenizer.eos_token or "<|endoftext|>"
    return [eos_token, " The", " cat", " mat", " dog"], eos_token


def build_demo(
    real_hf: bool = False,
    model_name: str = "roneneldan/TinyStories-1M",
    quiet_transformers: bool = True,
):
    """Build tokenizer/model, graph bundle, enforcement, DFA, and adapter."""
    tokenizer, model = load_backend(
        real_hf=real_hf,
        model_name=model_name,
        quiet_transformers=quiet_transformers,
    )
    vocab, eos_token = generation_vocab_for_tokenizer(tokenizer, real_hf=real_hf)
    graph, bundle = build_generation_graph(tokenizer, vocab, eos_token=eos_token)
    enforcement = discover_generation_enforcement(graph, bundle, on_unsupported="error")
    dfa = constraints_to_dfa(enforcement.dfa_constraints, bundle.vocabulary)
    adapter = HuggingFaceGenerationAdapter(model, tokenizer, bundle.vocabulary)
    return graph, bundle, enforcement, dfa, adapter, tokenizer


def run_all_modes(
    prompt: str = "Once",
    max_new_tokens: int = 4,
    real_hf: bool = False,
    model_name: str = "roneneldan/TinyStories-1M",
    quiet_transformers: bool = True,
) -> dict[str, object]:
    """Run greedy, beam, and sampling decoders and return their results."""
    _graph, bundle, enforcement, dfa, adapter, tokenizer = build_demo(
        real_hf=real_hf,
        model_name=model_name,
        quiet_transformers=quiet_transformers,
    )
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generator = torch.Generator().manual_seed(3)

    results = {
        "greedy": adapter.constrained_greedy(prompt_ids, dfa, max_new_tokens=max_new_tokens),
        "beam": adapter.constrained_beam_search(
            prompt_ids,
            dfa,
            max_new_tokens=max_new_tokens,
            beam_size=3,
            early_stopping=False,
        ),
        "sample": adapter.constrained_sample(
            prompt_ids,
            dfa,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.95,
            generator=generator,
        ),
        "constraints": enforcement.dfa_constraints,
        "vocabulary": bundle.vocabulary,
    }
    return results


def _next_logits_for_space_log(model, token_ids: list[int], device: torch.device) -> torch.Tensor:
    model_input = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    return model(model_input).logits[0, -1, :]


def _known_label_rows(logits, masked, vocabulary):
    probs = torch.softmax(masked, dim=-1)
    rows = []
    for label, token in enumerate(vocabulary.tokens):
        token_id = vocabulary.token_id_for_label(label)
        allowed = bool(masked[token_id] > -5e8)
        rows.append(
            {
                "label": label,
                "token": token,
                "token_id": token_id,
                "raw": float(logits[token_id].item()),
                "allowed": allowed,
                "prob": float(probs[token_id].item()) if allowed else 0.0,
            }
        )
    return rows


def _format_space_rows(rows, *, selected_label: int | None = None, max_allowed: int = 4) -> str:
    allowed = [row for row in rows if row["allowed"]]
    allowed.sort(key=lambda row: row["raw"], reverse=True)
    blocked = [row for row in rows if not row["allowed"]]
    parts = []
    for row in allowed[:max_allowed]:
        mark = "*" if row["label"] == selected_label else ""
        parts.append(f"{row['token']!r}:raw={row['raw']:.2f},p={row['prob']:.2f}{mark}")
    if len(allowed) > max_allowed:
        parts.append(f"... {len(allowed) - max_allowed} more allowed")
    if blocked:
        parts.append("blocked=" + ",".join(repr(row["token"]) for row in blocked[:4]))
    return "; ".join(parts)


def _filter_sampling_logits_for_log(
    logits: torch.Tensor,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
    fill_value: float = -1e9,
) -> torch.Tensor:
    filtered = logits.clone()
    if top_k is not None and top_k < filtered.numel():
        values, _indices = torch.topk(filtered, top_k)
        filtered = filtered.masked_fill(filtered < values[-1], fill_value)
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        filtered[sorted_indices[remove]] = fill_value
    return filtered


def _label_text(labels: list[int], vocabulary) -> str:
    if not labels:
        return "[]"
    return "[" + " ".join(repr(vocabulary.token_for_label(label)) for label in labels) + "]"


def _print_decoder_space_log(
    *,
    model,
    prompt_ids: torch.Tensor,
    dfa,
    vocabulary,
    max_new_tokens: int,
    beam_size: int = 3,
    sample_temperature: float = 0.8,
    sample_top_p: float = 0.95,
    sample_seed: int = 3,
    max_steps: int = 4,
) -> None:
    """Print a compact view of the constrained action space used by the demo."""
    device = prompt_ids.device
    base_ids = [int(item) for item in prompt_ids.squeeze(0).tolist()]
    eos_label = vocabulary.eos_label

    print("\nDecoder spaces")
    print("  raw = unmasked model logit; p = probability after DFA mask/filter; * = selected")
    print("  DFA blocks labels that cannot still reach an accepting state within the token budget")

    print("\n  greedy space")
    token_ids = list(base_ids)
    labels: list[int] = []
    state = dfa.start_state
    for step_idx in range(min(max_new_tokens, max_steps)):
        remaining = max_new_tokens - step_idx
        logits = _next_logits_for_space_log(model, token_ids, device)
        allowed = {int(label) for label in dfa.allowed_tokens(state, remaining_steps=remaining)}
        masked = mask_logits_for_dfa(logits, allowed, vocabulary)
        next_id = int(torch.argmax(masked).item())
        next_label = vocabulary.label_for_token_id(next_id)
        rows = _known_label_rows(logits, masked, vocabulary)
        print(
            f"    step {step_idx + 1} prefix={_label_text(labels, vocabulary)} -> "
            f"{_format_space_rows(rows, selected_label=next_label)}"
        )
        token_ids.append(next_id)
        labels.append(next_label)
        state = dfa.step(state, next_label)
        if next_label == eos_label and dfa.is_accepting(state):
            break

    print(f"    final greedy path: {_label_text(labels, vocabulary)} accepted={dfa.is_accepting(state)}")

    print(f"\n  beam space (beam_size={beam_size})")
    beams = [{"ids": list(base_ids), "labels": [], "state": dfa.start_state, "score": 0.0, "finished": False}]
    finished = []
    for step_idx in range(min(max_new_tokens, max_steps)):
        expansions = []
        print(f"    step {step_idx + 1}")
        for beam_index, beam in enumerate(beams):
            if beam["finished"]:
                expansions.append(beam)
                print(f"      beam {beam_index + 1} {_label_text(beam['labels'], vocabulary)} is finished")
                continue
            remaining = max_new_tokens - step_idx
            logits = _next_logits_for_space_log(model, beam["ids"], device)
            allowed = {int(label) for label in dfa.allowed_tokens(beam["state"], remaining_steps=remaining)}
            try:
                masked = mask_logits_for_dfa(logits, allowed, vocabulary)
            except ValueError:
                continue
            log_probs = torch.log_softmax(masked, dim=-1)
            valid_ids = torch.nonzero(masked > -5e8, as_tuple=False).flatten()
            local_beam = min(beam_size, int(valid_ids.numel()))
            local_scores, local_positions = torch.topk(log_probs[valid_ids], local_beam)
            branch_bits = []
            for score_delta, position in zip(local_scores.tolist(), local_positions.tolist()):
                next_id = int(valid_ids[int(position)].item())
                next_label = vocabulary.label_for_token_id(next_id)
                next_state = dfa.step(beam["state"], next_label)
                child = {
                    "ids": beam["ids"] + [next_id],
                    "labels": beam["labels"] + [next_label],
                    "state": next_state,
                    "score": float(beam["score"] + score_delta),
                    "finished": next_label == eos_label and dfa.is_accepting(next_state),
                }
                expansions.append(child)
                if child["finished"]:
                    finished.append(child)
                branch_bits.append(f"{vocabulary.token_for_label(next_label)!r}:{child['score']:.2f}")
            print(f"      beam {beam_index + 1} {_label_text(beam['labels'], vocabulary)} expands -> {', '.join(branch_bits)}")
        if not expansions:
            break
        expansions.sort(key=lambda item: item["score"], reverse=True)
        beams = expansions[:beam_size]
        print(
            "      kept -> "
            + ", ".join(f"{_label_text(beam['labels'], vocabulary)} score={beam['score']:.2f}" for beam in beams)
        )
    beam_ranked = sorted(finished or beams, key=lambda item: item["score"], reverse=True)
    if beam_ranked:
        best = beam_ranked[0]
        print(f"    final beam path: {_label_text(best['labels'], vocabulary)} accepted={dfa.is_accepting(best['state'])}")

    print(f"\n  sample space (temperature={sample_temperature}, top_p={sample_top_p}, seed={sample_seed})")
    token_ids = list(base_ids)
    labels = []
    state = dfa.start_state
    generator = torch.Generator(device=device).manual_seed(sample_seed)
    for step_idx in range(min(max_new_tokens, max_steps)):
        remaining = max_new_tokens - step_idx
        logits = _next_logits_for_space_log(model, token_ids, device)
        allowed = {int(label) for label in dfa.allowed_tokens(state, remaining_steps=remaining)}
        masked = mask_logits_for_dfa(logits, allowed, vocabulary)
        constrained = masked / float(sample_temperature)
        filtered = _filter_sampling_logits_for_log(constrained, top_p=sample_top_p)
        if torch.all(filtered <= -5e8):
            filtered = constrained
        probs = torch.softmax(filtered, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        next_label = vocabulary.label_for_token_id(next_id)
        rows = _known_label_rows(logits, filtered, vocabulary)
        print(
            f"    step {step_idx + 1} prefix={_label_text(labels, vocabulary)} -> "
            f"{_format_space_rows(rows, selected_label=next_label)}"
        )
        token_ids.append(next_id)
        labels.append(next_label)
        state = dfa.step(state, next_label)
        if next_label == eos_label and dfa.is_accepting(state):
            break
    print(f"    final sample path: {_label_text(labels, vocabulary)} accepted={dfa.is_accepting(state)}")


def _print_result(name: str, result, tokenizer, prompt_len: int) -> None:
    generated_ids = result.token_ids[prompt_len:]
    print(f"\n{name}")
    print("  token_ids:", generated_ids)
    print("  text:", repr(tokenizer.decode(generated_ids)))
    print("  labels:", result.labels)
    print("  accepted:", result.accepted)
    if result.score is not None:
        print("  score:", round(float(result.score), 4))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-hf", action="store_true", help="load a real HuggingFace causal LM")
    parser.add_argument("--model", default="roneneldan/TinyStories-1M", help="HuggingFace model id")
    parser.add_argument("--prompt", default="Once", help="prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--show-space-log",
        action="store_true",
        help="print the DFA-filtered action/search space for greedy, beam, and sampling",
    )
    parser.add_argument(
        "--hide-space-log",
        action="store_true",
        help="suppress the default mock-mode action/search-space trace",
    )
    parser.add_argument(
        "--show-transformers-load-report",
        action="store_true",
        help="show benign Transformers checkpoint loading warnings/reports",
    )
    args = parser.parse_args(argv)

    graph, bundle, enforcement, dfa, adapter, tokenizer = build_demo(
        real_hf=args.real_hf,
        model_name=args.model,
        quiet_transformers=not args.show_transformers_load_report,
    )
    prompt_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

    print("Path: hard enforcement only (graph constraints -> DFA logits mask; no PMD learning)")
    if not args.real_hf:
        print("Mock LM: intentionally branchy so greedy, beam, and sample show different DFA-valid paths")
    prompt_token_ids = [int(item) for item in prompt_ids.squeeze(0).tolist()]
    print("Initial prompt:")
    print("  text:", repr(args.prompt))
    print("  token_ids:", prompt_token_ids)
    print("  decoded:", repr(tokenizer.decode(prompt_token_ids)))
    print("  max_new_tokens:", args.max_new_tokens)
    print("Discovered DFA constraints:")
    for constraint in enforcement.dfa_constraints:
        print(" -", constraint.name)

    if args.show_space_log or (not args.real_hf and not args.hide_space_log):
        _print_decoder_space_log(
            model=adapter.model,
            prompt_ids=prompt_ids,
            dfa=dfa,
            vocabulary=bundle.vocabulary,
            max_new_tokens=args.max_new_tokens,
        )

    greedy = adapter.constrained_greedy(prompt_ids, dfa, max_new_tokens=args.max_new_tokens)
    beam = adapter.constrained_beam_search(
        prompt_ids,
        dfa,
        max_new_tokens=args.max_new_tokens,
        beam_size=3,
        early_stopping=False,
    )
    sample = adapter.constrained_sample(
        prompt_ids,
        dfa,
        max_new_tokens=args.max_new_tokens,
        temperature=0.8,
        top_p=0.95,
        generator=torch.Generator().manual_seed(3),
    )

    prompt_len = prompt_ids.shape[1]
    _print_result("greedy", greedy, tokenizer, prompt_len)
    _print_result("beam", beam, tokenizer, prompt_len)
    _print_result("sample", sample, tokenizer, prompt_len)

    print("\nDFA states:", len(dfa.states))
    print("Vocabulary:", bundle.vocabulary.labels)
    print("Graph:", graph.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
