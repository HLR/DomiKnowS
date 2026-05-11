# OpenAI-Compatible Generation Example

This task demonstrates the OpenAI-style path for `domiknows.generation`:
generate text first, then use `OpenAIResponsesAdapter.generate_and_verify(...)`
to encode the output and return `GenerationResult(accepted=...)` from the
graph-discovered DFA.

Unlike the HuggingFace demo, a plain OpenAI-compatible API does not expose a
portable per-token logits hook. That means this demo is verification-only unless
a local backend exposes native guided decoding, grammar controls, or custom
logits processors outside the generic OpenAI surface.

## Backend Capabilities

| Backend | Default URL | Default Model | Constraint Behavior | Extra Signals |
| --- | --- | --- | --- | --- |
| `mock` | none | `mock-openai-compatible` | offline generate-then-verify | fake logprobs for tests |
| `openai` | hosted OpenAI | `gpt-4.1-mini` | generate-then-verify | hosted API metadata only |
| `ollama` | `http://localhost:11434/v1/` | `llama3.2` | generate-then-verify | backend-specific options vary |
| `vllm` | `http://localhost:8000/v1/` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | generate-then-verify unless native guided decoding is wired | logprobs / extra params when supported |
| `llamacpp` | `http://localhost:8080/v1/` | `local-model` | generate-then-verify unless native grammar controls are wired | logprobs / grammar-style options when supported |

## Run Offline

```powershell
uv run --project Tasks/openai_generation python Tasks/openai_generation/run_demo.py
```

Rejected mock output:

```powershell
uv run --project Tasks/openai_generation python Tasks/openai_generation/run_demo.py --mock-output rejected
```

Mock logprob metadata:

```powershell
uv run --project Tasks/openai_generation python Tasks/openai_generation/run_demo.py --request-logprobs
```

## Run Local OpenAI-Compatible Servers

Ollama:

```powershell
uv run --project Tasks/openai_generation python Tasks/openai_generation/run_demo.py --backend ollama --model llama3.2
```

vLLM:

```powershell
uv run --project Tasks/openai_generation python Tasks/openai_generation/run_demo.py --backend vllm --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --request-logprobs
```

llama.cpp server:

```powershell
uv run --project Tasks/openai_generation python Tasks/openai_generation/run_demo.py --backend llamacpp --model local-model --extra-param temperature=0.2
```

Use `--base-url` to point at a non-default server and `--extra-param key=value`
for backend-specific request fields.

## Run Hosted OpenAI

```powershell
$env:OPENAI_API_KEY="..."
uv run --project Tasks/openai_generation python Tasks/openai_generation/run_demo.py --backend openai --model gpt-4.1-mini --prompt "Write a tiny sentence about a cat"
```

The adapter still checks the result after generation. For hard token-level
constraints, use a local model path with an actual logits hook, grammar/guided
decoding, or the HuggingFace DFA decoder.

## Hybrid Reranking

OpenAI-compatible outputs can also be used with the package-level
`HybridController`. In this mode the API model still generates normally, the
DFA verifies the output, and a trained compact DomiKnowS head reranks or
diagnoses candidates:

```python
from domiknows.generation import HybridController

controller = HybridController(
    generator=adapter,
    vocabulary=bundle.vocabulary,
    dfa=dfa,
    scorer_head=trained_head,
    enforcement=enforcement,
    tokenizer=tokenizer,
    constraints=enforcement.dfa_constraints,
)

ranked = controller.generate_verify_rerank("Once", num_candidates=3)
```

This remains generate-then-verify/rerank. It does not provide a portable
per-token hard decoder for hosted or generic OpenAI-compatible APIs.

## Files

- `graph.py`: generation graph and raw DomiKnowS constraints.
- `mock_openai.py`: offline Responses API mock and tiny tokenizer.
- `run_demo.py`: CLI and importable helpers used by tests.
- `pyproject.toml`: task-local uv project.
