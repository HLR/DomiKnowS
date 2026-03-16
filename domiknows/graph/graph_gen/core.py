"""
Shared logic for DomiKnowS Graph Generator.
"""

import json
import logging
import re
import traceback
from functools import lru_cache
from pathlib import Path

import requests
import yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).parent
_PRIVATE_CONFIG_PATH = _CONFIG_DIR / "config_openai.yaml"
CONFIG_PATH = _PRIVATE_CONFIG_PATH if _PRIVATE_CONFIG_PATH.exists() else _CONFIG_DIR / "config.yaml"
PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

CFG = load_config()

logger = logging.getLogger(__name__)

# Multi-model support
MODELS = CFG["models"]
DEFAULT_MODEL = next(iter(MODELS))
MAX_FIX_ATTEMPTS = 5


def get_available_models() -> list[dict]:
    """Return list of {name, server_url} for all configured models."""
    return [
        {"name": name, "server_url": info.get("server_url", "")}
        for name, info in MODELS.items()
    ]


def _is_openai_responses_endpoint(url: str) -> bool:
    return "/v1/responses" in url


def _messages_to_openai_input(messages: list[dict]) -> list[dict]:
    converted: list[dict] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        converted.append(
            {
                "role": role,
                "content": [{"type": "input_text", "text": str(content)}],
            }
        )
    return converted


def _extract_openai_response_text(response_json: dict) -> str:
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: list[str] = []
    for item in response_json.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)

    if chunks:
        return "\n".join(chunks).strip()

    raise ValueError("OpenAI response did not include any text output.")


def _extract_non_openai_response_text(response_json: dict) -> str:
    # Ollama /api/chat
    message = response_json.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content

    # Ollama /api/generate style
    response_text = response_json.get("response")
    if isinstance(response_text, str) and response_text.strip():
        return response_text

    # OpenAI-compatible chat-completions style
    choices = response_json.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        msg = first.get("message") if isinstance(first, dict) else {}
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content

    # Some providers return explicit error object with HTTP 200
    err = response_json.get("error")
    if err:
        if isinstance(err, dict):
            err_msg = err.get("message") or str(err)
        else:
            err_msg = str(err)
        raise RuntimeError(f"Model provider returned error payload: {err_msg}")

    raise ValueError(
        "Could not extract response text from provider payload. "
        f"Top-level keys: {sorted(response_json.keys())}"
    )


def _resolve_model(model_name: str | None) -> tuple[str, str, str | None]:
    """Return (server_url, model_name, api_key) for the given model, or default."""
    name = model_name or DEFAULT_MODEL
    info = MODELS.get(name)
    if not info:
        name = DEFAULT_MODEL
        info = MODELS[name]
        logger.info("Model '%s' not found in config, falling back to default model '%s'.", model_name, DEFAULT_MODEL)
    else:
        logger.info("Using model '%s' with server URL: %s", name, info.get("server_url", "N/A"))
    return info["server_url"], name, info.get("api_key")

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def get_prompt(name: str, **kwargs) -> str:
    template = _load_prompt(name)
    if kwargs:
        return template.format(**kwargs)
    return template


def get_syntax_description() -> str:
    return _load_prompt("syntax_description.txt")


def get_syntax_description_graph_only() -> str:
    return _load_prompt("syntax_description_graph_only.txt")

# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------


def llm_chat(messages: list[dict], temperature: float = 0.2, model: str | None = None) -> tuple[str, str]:
    """Send messages to the LLM. Returns (response_text, model_name_used)."""
    server_url, model_name, api_key = _resolve_model(model)
    chat_endpoint = f"{server_url}"

    logger.info(
        "Sending request to model '%s' at %s with temperature=%s",
        model_name,
        chat_endpoint,
        temperature,
    )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if _is_openai_responses_endpoint(chat_endpoint):
        payload = {
            "model": model_name,
            "input": _messages_to_openai_input(messages),
            "temperature": temperature,
        }
    else:
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

    resp = requests.post(chat_endpoint, json=payload, headers=headers, timeout=300)
    if not resp.ok:
        detail = resp.text.strip()
        if len(detail) > 1200:
            detail = detail[:1200] + "..."
        raise RuntimeError(
            f"HTTP {resp.status_code} calling {chat_endpoint}: {detail or 'No response body'}"
        )

    try:
        response_json = resp.json()
    except ValueError as e:
        body = resp.text.strip()
        if len(body) > 1200:
            body = body[:1200] + "..."
        raise RuntimeError(
            f"Provider returned non-JSON response from {chat_endpoint}: {body or 'empty body'}"
        ) from e

    if _is_openai_responses_endpoint(chat_endpoint):
        return _extract_openai_response_text(response_json), model_name
    return _extract_non_openai_response_text(response_json), model_name

# ---------------------------------------------------------------------------
# Code / JSON extraction
# ---------------------------------------------------------------------------

def extract_code_block(text: str) -> str | None:
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_json_block(text: str) -> dict | None:
    match = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    brace_start = text.find("{")
    if brace_start != -1:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


_CONSTRAINT_PATTERN = re.compile(
    r'\b(?:ifL|andL|orL|nandL|norL|xorL|notL|equivalenceL|eqL|fixedL|forAllL|'
    r'existsL|atLeastL|atMostL|exactL|existsAL|atLeastAL|atMostAL|exactAL|'
    r'greaterL|greaterEqL|lessL|lessEqL|equalCountsL|sumL|iotaL|queryL)\s*\('
)


def has_logical_constraints(code: str) -> bool:
    return bool(_CONSTRAINT_PATTERN.search(code))


def count_execute_calls(code: str) -> int:
    return len(re.findall(r'\bexecute\s*\(', code))


AUTO_IMPORTS = """\
from domiknows.graph import Graph, Concept, EnumConcept
from domiknows.graph import Relation
from domiknows.graph import (
    ifL, andL, orL, nandL, norL, xorL, notL, equivalenceL,
    eqL, fixedL, forAllL,
    existsL, atLeastL, atMostL, exactL,
    existsAL, atLeastAL, atMostAL, exactAL,
    greaterL, greaterEqL, lessL, lessEqL, equalCountsL,
    sumL, iotaL, queryL,
    execute,
)
from domiknows.graph import Property
from domiknows.graph.relation import disjoint
"""

# ---------------------------------------------------------------------------
# Error enrichment helpers
# ---------------------------------------------------------------------------

def _build_elc_mapping(code: str) -> dict[str, str]:
    mapping = {}
    match = re.search(r'qa_data\s*=\s*\[(.+?)\]', code, re.DOTALL)
    if not match:
        return mapping
    constraints = re.findall(
        r"""['"]constraint['"]\s*:\s*(['"])(.*?)\1""",
        match.group(1),
        re.DOTALL,
    )
    for i, (_, constraint_str) in enumerate(constraints):
        mapping[f"ELC{i}"] = constraint_str.replace('\\"', '"')
    return mapping


def _build_lc_mapping_from_graph(exec_globals: dict) -> dict[str, str]:
    mapping = {}
    try:
        for obj in exec_globals.values():
            if hasattr(obj, 'logicalConstrains') and hasattr(obj, 'executableLCs'):
                for lc_name, lc in obj.logicalConstrains.items():
                    try:
                        mapping[lc_name] = f"{type(lc).__name__}{lc.strEs()}"
                    except Exception:
                        mapping[lc_name] = type(lc).__name__
                for elc_name, elc in obj.executableLCs.items():
                    try:
                        inner = elc.innerLC if hasattr(elc, 'innerLC') else elc
                        mapping[elc_name] = f"{type(inner).__name__}{inner.strEs()}"
                    except Exception:
                        mapping[elc_name] = type(elc).__name__
                break
    except Exception:
        pass
    return mapping


def _enrich_error_with_constraint(error_msg: str, code: str, lc_mapping: dict) -> str:
    elc_mapping = _build_elc_mapping(code)
    combined = {**lc_mapping, **elc_mapping}
    if not combined:
        return error_msg

    def replace_lc(m):
        lc_name = m.group(0)
        if lc_name in combined:
            return f"{lc_name} ({combined[lc_name]})"
        return lc_name

    return re.sub(r'\bE?LC\d+\b', replace_lc, error_msg)

# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------

def execute_graph_code(code: str) -> tuple[bool, str]:
    full_code = AUTO_IMPORTS + "\n" + code
    exec_globals: dict = {}
    try:
        exec(full_code, exec_globals)
        return True, "Graph executed successfully."
    except Exception as e:
        lc_mapping = _build_lc_mapping_from_graph(exec_globals)
        error_msg = f"{type(e).__name__}: {e}"
        return False, _enrich_error_with_constraint(error_msg, code, lc_mapping)

# ---------------------------------------------------------------------------
# Graph validation
# ---------------------------------------------------------------------------

def validate_graph_coverage(code: str, concepts_json: dict) -> tuple[bool, list[str], list[str]]:
    required_concepts = [c["name"] for c in concepts_json.get("concepts", [])]
    required_relations = [r["name"] for r in concepts_json.get("relations", [])]

    missing_concepts = []
    for name in required_concepts:
        pattern = rf"""(?:Concept|EnumConcept)\s*\(\s*['"]{ re.escape(name) }['"]"""
        if not re.search(pattern, code):
            missing_concepts.append(name)

    missing_relations = []
    for name in required_relations:
        pattern = rf"""Concept\s*\(\s*['"]{ re.escape(name) }['"]"""
        if not re.search(pattern, code):
            if name not in code:
                missing_relations.append(name)

    all_present = len(missing_concepts) == 0 and len(missing_relations) == 0
    return all_present, missing_concepts, missing_relations


def merge_executable_into_graph(graph_code: str, exec_snippet: str) -> str:
    indented = "\n".join("    " + line for line in exec_snippet.strip().splitlines())
    return graph_code.rstrip() + "\n\nwith graph:\n" + indented + "\n"

# ---------------------------------------------------------------------------
# High-level pipeline helpers
# ---------------------------------------------------------------------------

def do_extract(questions: list[str], model: str | None = None) -> tuple[str, str, dict | None, str]:
    """Returns (prompt_sent, raw_response, parsed_json_or_None, model_used)."""
    questions_text = "\n".join(f"- {q}" for q in questions)
    prompt = get_prompt("extract_concepts.txt", questions=questions_text)
    raw, model_used = llm_chat([{"role": "user", "content": prompt}], model=model)
    return prompt, raw, extract_json_block(raw), model_used


def do_generate(questions: list[str], concepts_json: dict, model: str | None = None) -> tuple[list[dict], str | None]:
    """Returns (log, extracted_code_or_None)."""
    questions_text = "\n".join(f"- {q}" for q in questions)
    prompt = get_prompt(
        "generate_graph.txt",
        syntax_description_graph_only=get_syntax_description_graph_only(),
        questions=questions_text,
        concepts_json=json.dumps(concepts_json, indent=2),
    )
    messages: list[dict] = [{"role": "user", "content": prompt}]
    raw, model_used = llm_chat(messages, temperature=0.1, model=model)
    code = extract_code_block(raw)
    log: list[dict] = [{"type": "llm_interaction", "iteration": 1, "prompt_sent": prompt, "raw_response": raw, "model": model_used}]

    if code and has_logical_constraints(code):
        log.append({"type": "warning", "message": "Logical constraints detected in generated code — requesting correction"})
        correction = (
            "Your response contains logical constraints (e.g. ifL, andL, existsL, etc.). "
            "At this stage ONLY concepts and relations should be defined — no logical constraints. "
            "Please remove all logical constraint calls and return only the corrected code."
        )
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": correction})
        raw, model_used = llm_chat(messages, temperature=0.1, model=model)
        code = extract_code_block(raw)
        log.append({"type": "llm_interaction", "iteration": 2, "prompt_sent": correction, "raw_response": raw, "model": model_used})

    return log, code


def do_validate_and_fix_coverage(
    code: str, concepts_json: dict, model: str | None = None
) -> tuple[str, str, str | None, str]:
    """Returns (prompt_sent, raw_response, fixed_code, model_used)."""
    required_concepts = [c["name"] for c in concepts_json.get("concepts", [])]
    required_relations = [r["name"] for r in concepts_json.get("relations", [])]
    _, missing_c, missing_r = validate_graph_coverage(code, concepts_json)

    prompt = get_prompt(
        "validate_graph.txt",
        required_concepts=", ".join(required_concepts),
        required_relations=", ".join(required_relations),
        missing_concepts=", ".join(missing_c) if missing_c else "none",
        missing_relations=", ".join(missing_r) if missing_r else "none",
        code=code,
        syntax_description_graph_only=get_syntax_description_graph_only(),
    )
    raw, model_used = llm_chat([{"role": "user", "content": prompt}], temperature=0.1, model=model)
    return prompt, raw, extract_code_block(raw), model_used


def do_generate_executable_single(question: str, graph_code: str, model: str | None = None) -> tuple[str, str, str | None, str]:
    """Returns (prompt_sent, raw_response, single_execute_snippet_or_None, model_used)."""
    prompt = get_prompt(
        "generate_executable.txt",
        graph_code=graph_code,
        question=question,
    )
    raw, model_used = llm_chat([{"role": "user", "content": prompt}], temperature=0.1, model=model)
    return prompt, raw, extract_code_block(raw), model_used


def do_generate_executable(questions: list[str], graph_code: str, model: str | None = None) -> tuple[list[dict], str | None]:
    """Process each question individually and combine results."""
    log: list[dict] = []
    snippets: list[str] = []

    for i, question in enumerate(questions):
        prompt, raw, snippet, model_used = do_generate_executable_single(question, graph_code, model=model)

        entry = {
            "type": "llm_interaction",
            "iteration": i + 1,
            "question": question,
            "prompt_sent": prompt,
            "raw_response": raw,
            "snippet": snippet,
            "model": model_used,
        }
        log.append(entry)

        if snippet:
            snippets.append(f"# Q{i + 1}: {question}\n{snippet.strip()}")
        else:
            log.append({
                "type": "warning",
                "message": f"Could not extract execute() call for question {i + 1}: {question}",
            })

    if not snippets:
        return log, None

    combined = "\n\n".join(snippets)
    return log, combined


def do_fix(
    code: str, error: str, expected_execute_count: int | None = None, model: str | None = None
) -> tuple[list[dict], str | None]:
    """Returns (log, fixed_code_or_None)."""
    prompt = get_prompt(
        "fix_graph.txt",
        error=error,
        code=code,
        syntax_description=get_syntax_description(),
    )
    messages: list[dict] = [{"role": "user", "content": prompt}]
    raw, model_used = llm_chat(messages, temperature=0.1, model=model)
    fixed = extract_code_block(raw)
    log: list[dict] = [{"type": "llm_interaction", "iteration": 1, "prompt_sent": prompt, "raw_response": raw, "model": model_used}]

    if fixed is not None and expected_execute_count is not None:
        actual = count_execute_calls(fixed)
        if actual != expected_execute_count:
            log.append({"type": "warning", "message": f"Execute count mismatch: expected {expected_execute_count} execute() call(s), found {actual} — requesting correction"})
            correction = (
                f"Your fix removed execute() constraints. "
                f"The code must contain exactly {expected_execute_count} execute() call(s) "
                f"but currently has {actual}. "
                "Restore all missing execute() constraints and return the complete corrected code."
            )
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": correction})
            raw, model_used = llm_chat(messages, temperature=0.1, model=model)
            fixed = extract_code_block(raw)
            log.append({"type": "llm_interaction", "iteration": 2, "prompt_sent": correction, "raw_response": raw, "model": model_used})

    return log, fixed