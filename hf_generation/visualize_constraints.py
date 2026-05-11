"""Launch a local web viewer for the HF-generation constraint DFA."""
from __future__ import annotations

import argparse

from domiknows.generation import create_generation_debug_app, run_generation_debug_server

try:
    from .run_demo import build_demo
except ImportError:
    from run_demo import build_demo


def _parse_sequence(text: str | None) -> list[int] | None:
    if text is None:
        return None
    if not text.strip():
        return []
    return [int(part.strip()) for part in text.split(",")]


def build_visualization(sequence: list[int] | None = None, prompt: str = "Once", max_new_tokens: int = 4):
    """Build the demo DFA and choose a trace sequence."""

    graph, bundle, enforcement, dfa, adapter, tokenizer = build_demo(real_hf=False)
    if sequence is None:
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
        result = adapter.constrained_greedy(prompt_ids, dfa, max_new_tokens=max_new_tokens)
        sequence = list(result.labels)
    labels = {idx: token for idx, token in enumerate(bundle.vocabulary.labels)}
    return graph, bundle, enforcement, dfa, sequence, labels


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", help="comma-separated compact labels to trace, for example 1,2,3,0")
    parser.add_argument("--prompt", default="Once")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--no-server-smoke", action="store_true", help="build the app and query summary without blocking")
    args = parser.parse_args(argv)

    _graph, bundle, enforcement, dfa, sequence, labels = build_visualization(
        sequence=_parse_sequence(args.sequence),
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )
    title = "HF generation constraint debug trace"
    if args.no_server_smoke:
        app = create_generation_debug_app(dfa, sequence=sequence, title=title, symbol_labels=labels)
        with app.test_client() as client:
            response = client.get("/api/summary")
            print(response.get_json())
        return 0

    print("Discovered DFA constraints:")
    for constraint in enforcement.dfa_constraints:
        print(" -", constraint.name)
    print("Sequence:", sequence)
    print("Vocabulary:", bundle.vocabulary.labels)
    print(f"Open http://{args.host}:{args.port}")
    run_generation_debug_server(
        dfa,
        sequence=sequence,
        title=title,
        symbol_labels=labels,
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
