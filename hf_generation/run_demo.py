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
    generator = torch.Generator().manual_seed(0)

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
    print("Discovered DFA constraints:")
    for constraint in enforcement.dfa_constraints:
        print(" -", constraint.name)

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
        generator=torch.Generator().manual_seed(0),
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
