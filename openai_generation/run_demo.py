"""Run OpenAI-compatible generate-then-verify with DomiKnowS constraints."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

from domiknows.generation import (
    OpenAIResponsesAdapter,
    analyze_generation_constraints,
    discover_generation_enforcement,
)

try:
    from .graph import build_generation_graph
    from .mock_openai import MockOpenAIClient, MockTokenizer
except ImportError:
    from graph import build_generation_graph
    from mock_openai import MockOpenAIClient, MockTokenizer


DEFAULT_MODELS = {
    "mock": "mock-openai-compatible",
    "openai": "gpt-4.1-mini",
    "ollama": "llama3.2",
    "vllm": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llamacpp": "local-model",
}

DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434/v1/",
    "vllm": "http://localhost:8000/v1/",
    "llamacpp": "http://localhost:8080/v1/",
}

MOCK_OUTPUTS = {
    "accepted": " The cat mat<eos>",
    "rejected": " The dog<eos>",
}


@dataclass(frozen=True)
class BackendProfile:
    """Resolved backend settings for the demo."""

    name: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    supports_logprobs_request: bool = False
    notes: tuple[str, ...] = ()


def backend_profile(
    backend: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> BackendProfile:
    """Resolve task-level backend defaults."""
    backend = backend.lower()
    if backend not in DEFAULT_MODELS:
        raise ValueError(f"unknown backend {backend!r}")
    resolved_base_url = base_url or DEFAULT_BASE_URLS.get(backend)
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
    if backend in {"ollama", "vllm", "llamacpp"} and not resolved_key:
        resolved_key = "not-needed"
    return BackendProfile(
        name=backend,
        model=model or DEFAULT_MODELS[backend],
        base_url=resolved_base_url,
        api_key=resolved_key,
        supports_logprobs_request=backend in {"mock", "vllm", "llamacpp"},
        notes=_backend_notes(backend),
    )


def build_client(profile: BackendProfile, *, mock_text: str, include_logprobs: bool = False):
    """Build either the mock client or an OpenAI SDK client."""
    if profile.name == "mock":
        return MockOpenAIClient(mock_text, include_logprobs=include_logprobs)
    if profile.name == "openai" and not profile.api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --backend openai; use --backend mock for offline mode")
    from openai import OpenAI

    kwargs: dict[str, Any] = {}
    if profile.api_key:
        kwargs["api_key"] = profile.api_key
    if profile.base_url:
        kwargs["base_url"] = profile.base_url
    return OpenAI(**kwargs)


def parse_extra_params(values: list[str]) -> dict[str, Any]:
    """Parse repeated ``key=value`` CLI params into a request dict."""
    params: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"extra param must be key=value, got {value!r}")
        key, raw = value.split("=", 1)
        params[key] = _coerce_scalar(raw)
    return params


def run_generation(
    *,
    backend: str = "mock",
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    prompt: str = "Once",
    max_output_tokens: int = 16,
    mock_output: str = "accepted",
    custom_mock_output: str | None = None,
    request_logprobs: bool = False,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the demo and return a structured summary used by tests and CLI."""
    tokenizer = MockTokenizer()
    graph, bundle = build_generation_graph(tokenizer)
    enforcement = discover_generation_enforcement(graph, bundle, on_unsupported="error")
    dfa = enforcement.dfa

    profile = backend_profile(backend, model=model, base_url=base_url, api_key=api_key)
    mock_text = _mock_text(mock_output, custom_mock_output)
    client = build_client(profile, mock_text=mock_text, include_logprobs=request_logprobs)
    adapter = OpenAIResponsesAdapter(client=client, model=profile.model, tokenizer=tokenizer)

    request_params = dict(extra_params or {})
    if request_logprobs:
        request_params.setdefault("logprobs", True)
    result = adapter.generate_and_verify(
        prompt,
        bundle.vocabulary,
        dfa,
        max_output_tokens=max_output_tokens,
        explain=True,
        **request_params,
    )
    logprob_summary = extract_logprob_summary(result.raw)
    constraints = [
        f"{analysis.lc_name}: {'supported' if analysis.supported else analysis.reason}"
        for analysis in analyze_generation_constraints(graph, bundle, on_unsupported="error")
        if analysis.relevant
    ]

    return {
        "graph": graph.name,
        "backend": profile,
        "prompt": prompt,
        "text": result.text,
        "token_ids": result.token_ids,
        "labels": result.labels,
        "accepted": result.accepted,
        "rejection": result.rejection,
        "constraints": constraints,
        "logprobs": logprob_summary,
        "request": getattr(getattr(client, "responses", None), "request", None),
    }


def extract_logprob_summary(raw_response) -> list[dict[str, Any]]:
    """Extract a small backend-dependent logprob summary when present."""
    values = getattr(raw_response, "logprobs", None)
    if not values:
        return []
    summary = []
    for item in values:
        summary.append(
            {
                "token": getattr(item, "token", None),
                "logprob": getattr(item, "logprob", None),
            }
        )
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=sorted(DEFAULT_MODELS), default="mock")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--model")
    parser.add_argument("--prompt", default="Once")
    parser.add_argument("--max-output-tokens", type=int, default=16)
    parser.add_argument("--mock-output", choices=["accepted", "rejected", "custom"], default="accepted")
    parser.add_argument("--custom-mock-output")
    parser.add_argument("--request-logprobs", action="store_true")
    parser.add_argument("--extra-param", action="append", default=[])
    args = parser.parse_args(argv)

    summary = run_generation(
        backend=args.backend,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        prompt=args.prompt,
        max_output_tokens=args.max_output_tokens,
        mock_output=args.mock_output,
        custom_mock_output=args.custom_mock_output,
        request_logprobs=args.request_logprobs,
        extra_params=parse_extra_params(args.extra_param),
    )
    print_summary(summary)
    return 0


def print_summary(summary: dict[str, Any]) -> None:
    """Print a compact human-readable demo summary."""
    profile: BackendProfile = summary["backend"]
    print(f"Backend: {profile.name}")
    print(f"Model: {profile.model}")
    if profile.base_url:
        print(f"Base URL: {profile.base_url}")
    for note in profile.notes:
        print(f"Note: {note}")
    print("Discovered DFA constraints:")
    for constraint in summary["constraints"]:
        print(f" - {constraint}")
    print(f"Prompt: {summary['prompt']!r}")
    print(f"Text: {summary['text']!r}")
    print(f"Token IDs: {summary['token_ids']}")
    print(f"Labels: {summary['labels']}")
    print(f"Accepted: {summary['accepted']}")
    if summary["rejection"]:
        print(f"Rejection: {summary['rejection']}")
    if summary["logprobs"]:
        print("Logprobs:")
        for item in summary["logprobs"]:
            print(f" - {item['token']!r}: {item['logprob']}")


def _mock_text(mock_output: str, custom_mock_output: str | None) -> str:
    if mock_output == "custom":
        if custom_mock_output is None:
            raise ValueError("--custom-mock-output is required when --mock-output custom")
        return custom_mock_output
    return MOCK_OUTPUTS[mock_output]


def _backend_notes(backend: str) -> tuple[str, ...]:
    common = ("OpenAI-compatible generation is verified post hoc; plain API calls do not provide hard DFA masking.",)
    if backend == "mock":
        return ("Offline deterministic mock backend.",) + common
    if backend == "openai":
        return ("Hosted OpenAI Responses API: verification-only in this demo.",) + common
    if backend == "ollama":
        return ("Ollama exposes an OpenAI-compatible endpoint; backend-specific options vary by model/server.",) + common
    if backend == "vllm":
        return ("vLLM may expose logprobs and guided-decoding/server-native controls depending on server config.",) + common
    if backend == "llamacpp":
        return ("llama.cpp server may expose grammar/logprob options beyond the OpenAI-compatible surface.",) + common
    return common


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


if __name__ == "__main__":
    raise SystemExit(main())
