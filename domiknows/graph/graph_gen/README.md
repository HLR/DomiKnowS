# DomiKnowS Graph Generator (`graph_gen`)

LLM-powered tool that generates DomiKnowS graph definitions from natural language questions.

## How It Works

1. **Enter questions** — describe your domain through questions
2. **Extract concepts** — LLM identifies entity types and relations
3. **Generate graph** — LLM produces executable DomiKnowS Python code
4. **Encode constraints** — each question is individually converted to an `execute()` constraint
5. **Execute & auto-fix** — code is executed; errors are sent back to the LLM for correction (up to 5 attempts)

## Setup

```bash
uv add flask pyyaml requests
```

Edit `config.yaml` to point to your LLM server:

```yaml
LLM_SERVER_URL: "http://localhost:11434"
MODEL_LLM: "qwen3-coder-next"
```

Any OpenAI-compatible `/api/chat` endpoint works (Ollama, vLLM, etc.).

## Usage

### Web UI

```bash
uv run app.py
# Open http://localhost:5001
```

### CLI — Batch Mode (default)

Reads questions from a JSON dataset file (e.g. CLEVR, CLEVR-X) and outputs a Python file with the generated graph and encoded constraints.

```bash
uv run graph_builder.py dataset.json
```

Each question is processed individually by the LLM. Successfully encoded questions produce an `execute()` call in the output; failed questions are written as comments.

#### Parameters

| Parameter | Default | Description |
|---|---|---|
| `dataset` (positional) | — | Path to the JSON dataset file |
| `--keyword` | `question` | JSON key holding the question text in each record |
| `--list-key` | auto-detect | Top-level JSON key containing the list of records |
| `-o`, `--output` | `output_graph.py` | Output Python file path |
| `--limit` | all | Maximum number of questions to process |
| `--no-validate` | off | Skip execution/validation of the generated output |

#### Supported JSON formats

```jsonc
// Object with a list key (CLEVR-style) — auto-detects "questions", "data", "samples", etc.
{"questions": [{"question": "Is there a red cube?", ...}, ...]}

// Flat array of objects
[{"question": "Is there a red cube?"}, ...]

// Plain string array
["Is there a red cube?", "How many spheres are green?"]
```

#### Examples

```bash
# Process a CLEVR questions file
uv run graph_builder.py CLEVR_val_questions.json

# Custom keyword and limit
uv run graph_builder.py my_data.json --keyword text --limit 100 -o clevr_graph.py

# Explicit list key, skip validation
uv run graph_builder.py dataset.json --list-key data --no-validate
```

### CLI — Interactive Mode

Prompts for questions at the terminal like a REPL. Each question is processed immediately

```bash
uv run graph_builder.py --interactive
```

## Files

| File | Purpose |
|---|---|
| `config.yaml` | LLM server URL and model name |
| `core.py` | Shared logic: LLM calls, prompts, code extraction, execution |
| `app.py` | Flask web interface (imports from `core`) |
| `templates/index.html` | Single-page frontend |
| `graph_builder.py` | CLI: batch mode (default) and interactive mode |
| `prompts/` | LLM prompt templates |