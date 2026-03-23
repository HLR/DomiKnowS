"""
Shared logic for DomiKnowS Graph Generator.
"""

import json
import logging
import re
import threading
import time
import traceback
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
OLLAMA_DISCOVERY_PORTS = (11434, 11435)
OLLAMA_DISCOVERY_INTERVAL_SECONDS = int(CFG.get("ollama_discovery_interval_seconds", 60))

_DISCOVERED_MODELS: dict[str, dict] = {}
_DISCOVERED_MODELS_LOCK = threading.Lock()
_OLLAMA_MONITOR_STARTED = False
_OLLAMA_MONITOR_START_LOCK = threading.Lock()


def _build_ollama_chat_endpoint(port: int) -> str:
    return f"http://localhost:{port}/api/chat"


def _build_ollama_tags_endpoint(port: int) -> str:
    return f"http://localhost:{port}/api/tags"


def _build_model_catalog() -> dict[str, dict]:
    """Return configured models merged with dynamically discovered Ollama models."""
    catalog = {name: dict(info) for name, info in MODELS.items()}

    with _DISCOVERED_MODELS_LOCK:
        discovered = {name: dict(info) for name, info in _DISCOVERED_MODELS.items()}

    for discovered_name, info in discovered.items():
        provider_model_name = info.get("provider_model_name", discovered_name)
        if provider_model_name in MODELS and MODELS[provider_model_name].get("server_url") == info.get("server_url"):
            continue

        existing = catalog.get(discovered_name)
        if existing is None:
            catalog[discovered_name] = info
            continue

        if existing.get("server_url") == info.get("server_url"):
            continue

        port = info.get("port", "unknown")
        qualified_name = f"{discovered_name} @ localhost:{port}"
        suffix = 2
        while qualified_name in catalog:
            qualified_name = f"{discovered_name} @ localhost:{port} ({suffix})"
            suffix += 1
        catalog[qualified_name] = info

    return catalog


def get_available_models() -> list[dict]:
    """Return list of {name, server_url} for configured and discovered models."""
    return [
        {"name": name, "server_url": info.get("server_url", "")}
        for name, info in _build_model_catalog().items()
    ]


def _fetch_ollama_models(port: int) -> dict[str, dict]:
    """Fetch the currently loaded Ollama models for one localhost port."""
    endpoint = _build_ollama_tags_endpoint(port)
    try:
        response = requests.get(endpoint, timeout=3)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        logger.debug("Ollama discovery unavailable on localhost:%s: %s", port, exc)
        return {}
    except ValueError as exc:
        logger.warning("Ollama discovery on localhost:%s returned non-JSON payload: %s", port, exc)
        return {}

    discovered: dict[str, dict] = {}
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue

        model_name = item.get("model") or item.get("name")
        if not isinstance(model_name, str) or not model_name.strip():
            continue

        provider_model_name = model_name.strip()
        display_name = f"{provider_model_name} @ localhost:{port}"
        discovered[display_name] = {
            "server_url": _build_ollama_chat_endpoint(port),
            "api_key": None,
            "provider_model_name": provider_model_name,
            "port": port,
            "source": "ollama",
        }

    return discovered


def refresh_discovered_ollama_models() -> list[dict]:
    """Refresh the dynamic model catalog from Ollama servers on localhost."""
    refreshed: dict[str, dict] = {}
    for port in OLLAMA_DISCOVERY_PORTS:
        refreshed.update(_fetch_ollama_models(port))

    with _DISCOVERED_MODELS_LOCK:
        previous_keys = set(_DISCOVERED_MODELS)
        new_keys = set(refreshed)
        _DISCOVERED_MODELS.clear()
        _DISCOVERED_MODELS.update(refreshed)

    if previous_keys != new_keys:
        logger.info(
            "Updated discovered Ollama models: %s",
            sorted(new_keys) if new_keys else "none",
        )

    return get_available_models()


def _monitor_ollama_models(interval_seconds: int) -> None:
    """Background monitor that periodically refreshes localhost Ollama models."""
    while True:
        try:
            refresh_discovered_ollama_models()
        except Exception:
            logger.exception("Periodic Ollama model discovery failed")
        time.sleep(interval_seconds)


def start_ollama_model_monitor(interval_seconds: int = OLLAMA_DISCOVERY_INTERVAL_SECONDS) -> None:
    """Start a background daemon that refreshes Ollama models on localhost."""
    global _OLLAMA_MONITOR_STARTED

    with _OLLAMA_MONITOR_START_LOCK:
        if _OLLAMA_MONITOR_STARTED:
            return

        thread = threading.Thread(
            target=_monitor_ollama_models,
            args=(interval_seconds,),
            name="ollama-model-monitor",
            daemon=True,
        )
        thread.start()
        _OLLAMA_MONITOR_STARTED = True
        logger.info(
            "Started Ollama model monitor for localhost ports %s (interval=%ss)",
            list(OLLAMA_DISCOVERY_PORTS),
            interval_seconds,
        )


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
    catalog = _build_model_catalog()
    name = model_name or DEFAULT_MODEL
    info = catalog.get(name)
    if not info:
        name = DEFAULT_MODEL
        info = catalog[name]
        logger.info("Model '%s' not found in config, falling back to default model '%s'.", model_name, DEFAULT_MODEL)
    else:
        logger.info("Using model '%s' with server URL: %s", name, info.get("server_url", "N/A"))
    provider_model_name = info.get("provider_model_name", name)
    return info["server_url"], provider_model_name, info.get("api_key")


refresh_discovered_ollama_models()
start_ollama_model_monitor()

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_prompt_cache: dict[str, tuple[float, str]] = {}   # name -> (mtime, content)

def _load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    mtime = path.stat().st_mtime
    cached = _prompt_cache.get(name)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    content = path.read_text(encoding="utf-8")
    _prompt_cache[name] = (mtime, content)
    return content


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
    """Extract the best python code block from LLM output.

    For executable-constraint responses we prefer a block that contains
    an execute() call or a known constraint operator (queryL, existsL, …).
    If the first block is just graph definitions (Concept / object_node)
    we skip it and look for a later block that has the actual constraint.
    """
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        return None

    # Prefer a block that has execute() or a known constraint operator
    for block in blocks:
        stripped = block.strip()
        if "execute(" in stripped or _CONSTRAINT_PATTERN.search(stripped):
            return stripped

    # Fallback: return the last code block (more likely to be the answer)
    return blocks[-1].strip()


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
        graph = _find_graph_in_globals(exec_globals)
        if graph is not None:
            for lc_name, lc in graph.logicalConstrains.items():
                try:
                    mapping[lc_name] = f"{type(lc).__name__}{lc.strEs()}"
                except Exception:
                    mapping[lc_name] = type(lc).__name__
            for elc_name, elc in graph.executableLCs.items():
                try:
                    inner = elc.innerLC if hasattr(elc, 'innerLC') else elc
                    mapping[elc_name] = f"{type(inner).__name__}{inner.strEs()}"
                except Exception:
                    mapping[elc_name] = type(elc).__name__
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

def _find_graph_in_globals(exec_globals: dict):
    """Find the DomiKnowS Graph *instance* in exec_globals after code execution.

    Must skip the Graph *class* itself (which also has ``logicalConstrains``
    and ``executableLCs`` as @property descriptors).  We therefore require the
    candidate to be an instance (not a type) and verify that ``concepts``
    returns something iterable (not a raw property descriptor).
    """
    for obj in exec_globals.values():
        # Skip classes — we want instances only
        if isinstance(obj, type):
            continue
        if hasattr(obj, 'logicalConstrains') and hasattr(obj, 'executableLCs'):
            # Double-check that .concepts is actually accessible (not a property descriptor)
            try:
                concepts = obj.concepts
                if isinstance(concepts, property):
                    continue
                # Verify it's iterable (dict / OrderedDict)
                iter(concepts)
            except TypeError:
                continue
            return obj
    return None


def _describe_graph_structure(graph) -> str:
    """Build a concise summary of the graph's concepts, relations, and structure.

    This is appended to verification errors so the LLM can see what was
    actually defined and spot mismatches (e.g. missing is_a, wrong has_a args).
    """
    lines = ["Graph structure:"]
    try:
        concepts = dict(graph.concepts) if not isinstance(graph.concepts, property) else {}
    except Exception:
        concepts = {}

    if not concepts:
        lines.append("  (no concepts found)")
        return "\n".join(lines)

    from domiknows.graph.concept import EnumConcept

    for cname, concept in concepts.items():
        parts = [f"  {cname}"]
        if isinstance(concept, EnumConcept):
            parts.append(f"(EnumConcept, values={concept.enum})")

        # is_a parents
        parents = [r.dst.name for r in concept._out.get('is_a', [])]
        if parents:
            parts.append(f"is_a({', '.join(parents)})")

        # contains children
        children = [r.dst.name for r in concept._out.get('contains', [])]
        if children:
            parts.append(f"contains({', '.join(children)})")

        # has_a (relation attributes)
        has_a_rels = list(concept.has_a()) if hasattr(concept, 'has_a') else []
        if has_a_rels:
            for rel in has_a_rels:
                parts.append(f"has_a→{rel.dst.name}")

        lines.append(" ".join(parts))

    return "\n".join(lines)


def _verify_constraints(graph, code: str, exec_globals: dict) -> tuple[bool, str]:
    """Verify executable constraints using a dummy DataNode and AnswerSolver.

    Returns (success, error_message).  On success *error_message* is empty.
    On failure the error is enriched with constraint definitions so the LLM
    fix loop can understand which constraint is broken and why.
    """
    import traceback as _tb
    from domiknows.graph.dataNodeDummy import verifyExecutableConstraints

    try:
        all_ok, messages = verifyExecutableConstraints(graph)
    except Exception as e:
        tb_str = _tb.format_exc()
        # Try to get graph structure for context
        try:
            graph_desc = _describe_graph_structure(graph)
        except Exception:
            graph_desc = "  (could not describe graph structure)"

        messages = [
            f"Verification failed: {type(e).__name__}: {e}\n"
            f"  This indicates a problem with the graph definition — "
            f"e.g. missing concepts, broken relations, or incorrect has_a/is_a/contains structure.\n"
            f"\n{graph_desc}\n"
            f"\n  Traceback (last 5 frames):\n{''.join(_tb.format_tb(e.__traceback__)[-5:])}"
        ]
        all_ok = False

    if not all_ok:
        lc_mapping = _build_lc_mapping_from_graph(exec_globals)
        error_parts = ["Constraint verification errors:"]
        for m in messages:
            error_parts.append(f"- {m}")
            # If the message mentions an ELC, show its location in the code
            elc_match = re.search(r'\bELC\d+\b', m)
            if elc_match:
                elc_loc = _find_constraint_in_code(code, elc_match.group(0))
                if elc_loc:
                    error_parts.append(f"  Constraint definition: {elc_loc}")
        error_msg = "\n".join(error_parts)
        return False, _enrich_error_with_constraint(error_msg, code, lc_mapping)

    return True, ""


def _extract_failing_lines(tb_str: str, code: str) -> list[str]:
    """Extract all user code lines referenced in the traceback.

    The exec'd code is ``AUTO_IMPORTS + '\\n' + code``.  Python tracebacks
    for exec'd strings show ``File "<string>"`` with a line number.  We
    subtract the AUTO_IMPORTS offset to map back to the user's code.

    Returns a list of ``"line N: <code>"`` strings (may be empty).
    When the pointed-to line is inside a multiline string (docstring/comment),
    we scan forward to find the first real code line after it.
    """
    import re
    results = []
    code_lines = code.splitlines()
    auto_import_lines = AUTO_IMPORTS.count('\n') + 1  # +1 for the joining \n

    for m in re.finditer(r'File "<string>", line (\d+)', tb_str):
        exec_lineno = int(m.group(1))
        user_lineno = exec_lineno - auto_import_lines
        if not (1 <= user_lineno <= len(code_lines)):
            continue

        line_text = code_lines[user_lineno - 1].strip()

        # If the line is a docstring delimiter or empty, scan for context
        if line_text in ('"""', "'''", '"""\\', "'''\\", ''):
            # Find a meaningful line nearby (up to 5 lines forward)
            for offset in range(1, 6):
                idx = user_lineno - 1 + offset
                if idx < len(code_lines):
                    candidate = code_lines[idx].strip()
                    if candidate and candidate not in ('"""', "'''", ''):
                        line_text = f'(in docstring near) {candidate}'
                        user_lineno += offset
                        break

        entry = f"line {user_lineno}: {line_text}"
        if entry not in results:
            results.append(entry)

    return results


def _find_constraint_in_code(code: str, elc_name: str) -> str | None:
    """Find the execute() call that defines a specific ELC by name or index.

    Searches for patterns like ``execute(existsL(...))`` and returns the
    surrounding code with a line number.  Uses the ELC index to count
    execute() calls in order.
    """
    import re
    m = re.match(r'ELC(\d+)', elc_name)
    if not m:
        return None
    target_idx = int(m.group(1))

    # Find all execute() calls in order
    current_idx = 0
    for line_no, line in enumerate(code.splitlines(), 1):
        if re.search(r'\bexecute\s*\(', line):
            if current_idx == target_idx:
                return f"line {line_no}: {line.strip()}"
            current_idx += 1
    return None


_COMMON_MISTAKE_HINTS = [
    # Pattern: unicode escape errors from Windows paths in strings
    (
        r"unicode.?error.*unicodeescape.*codec can't decode",
        "The code contains a Windows-style backslash path (e.g. C:\\Users\\...) "
        "inside a Python string. Backslashes like \\U, \\N, \\x are interpreted "
        "as Unicode escapes. Fix by: (1) removing the path from the code entirely, "
        "(2) using raw strings r'...', or (3) doubling backslashes C:\\\\Users\\\\...",
    ),
    # Pattern: calling a Concept like a function with a string arg
    # e.g. left_of = pair('left_of')  → returns a list, not a Concept
    (
        r"'list' object has no attribute '(has_a|is_a|contains)'",
        "A variable holds a list instead of a Concept. "
        "This often happens when you call an existing Concept with a string argument "
        "(e.g. `left_of = pair('left_of')`) — Concept.__call__ with a string creates "
        "a logical constraint variable, NOT a new Concept. "
        "To create a new concept, use `left_of = Concept('left_of')` instead.",
    ),
    # Pattern: shadowing built-in 'object'
    (
        r"'(int|str|float|bool)' object has no attribute '(has_a|is_a|contains|name)'",
        "A Python built-in name may have been shadowed. "
        "Avoid using Python reserved names like 'object', 'type', 'list', 'set' "
        "as concept variable names.",
    ),
    # Pattern: calling has_a on something that isn't a Concept
    (
        r"has no attribute 'has_a'",
        "The variable used before .has_a() is not a Concept instance. "
        "Make sure each relation is created with `rel = Concept('rel_name')` "
        "before calling `rel.has_a(src_concept, dst_concept)`.",
    ),
    # Pattern: wrong number of return values from has_a
    (
        r"(not enough|too many) values to unpack",
        "The number of variables on the left side of `=` doesn't match what "
        ".has_a() returns. `concept.has_a(A, B)` returns exactly 2 relation "
        "attributes (source, target).",
    ),
    # Pattern: "Logical Element is incorrect" — vague DomiKnowS error
    (
        r"Logical (Element|Constraint) is incorrect",
        "A logical constraint (existsL, andL, queryL, etc.) received invalid arguments. "
        "Common causes: (1) passing a plain string instead of a concept variable "
        "like concept_name('x'), (2) passing a Python expression like 'y' == 'x' "
        "(which evaluates to True/False) instead of using eqL or separate variables, "
        "(3) wrong nesting — e.g. existsL needs concept predicates, not raw values.",
    ),
    # Pattern: Path validation error from DomiKnowS constraint system
    (
        r"The Path '(\w+)' from the variable .+ is not valid",
        "A relation path in a constraint is invalid. This means a variable "
        "is being used through a relation that doesn't connect to the expected "
        "concept. Check that: (1) the relation was defined with has_a(correct_src, correct_dst), "
        "(2) the variable names in the constraint match the relation's argument order, "
        "(3) the relation connects the right concept types.",
    ),
    # Pattern: concept not found
    (
        r"(KeyError|NameError):.*'(\w+)'",
        "A name is not defined or not found. Make sure all concepts, relations, "
        "and variables are defined before they are used in constraints.",
    ),
]


def _add_execution_hints(error_msg: str, tb_str: str, code: str) -> str:
    """Enrich an execution error with the failing code line and common-mistake hints."""
    import re
    parts = [error_msg]

    # Add the failing lines from user code
    failing_lines = _extract_failing_lines(tb_str, code)
    if failing_lines:
        parts.append("Failing code:")
        for fl in failing_lines:
            parts.append(f"  → {fl}")

    # If the error mentions a specific ELC, also show which execute() call it is
    elc_match = re.search(r'\bELC\d+\b', error_msg)
    if elc_match:
        elc_loc = _find_constraint_in_code(code, elc_match.group(0))
        if elc_loc:
            parts.append(f"Constraint definition: {elc_loc}")

    # Add applicable hints
    for pattern, hint in _COMMON_MISTAKE_HINTS:
        if re.search(pattern, error_msg, re.IGNORECASE):
            parts.append(f"Hint: {hint}")
            break  # one hint is enough

    return "\n".join(parts)


def _sanitize_code_strings(code: str) -> str:
    """Fix common string-literal issues that cause SyntaxErrors.

    1. Windows paths in regular strings (``C:\\Users\\...``) cause
       ``unicodeescape`` errors because ``\\U``, ``\\N``, ``\\x`` etc. are
       interpreted as Unicode escapes.  We strip any leading docstring
       (triple-quoted block at the very start) since it's just a comment
       generated by the LLM and not needed for execution.
    """
    import re

    # Strip a leading triple-quoted docstring (single or double quotes)
    # This is the most common source: LLM puts a description with file paths
    stripped = re.sub(
        r'''^(\s*)("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')''',
        '',
        code,
        count=1,
    )
    return stripped


def execute_graph_code(code: str) -> tuple[bool, str]:
    sanitized_code = _sanitize_code_strings(code)
    full_code = AUTO_IMPORTS + "\n" + sanitized_code
    exec_globals: dict = {}
    try:
        exec(full_code, exec_globals)
    except Exception as e:
        import traceback as _tb
        tb_str = _tb.format_exc()
        lc_mapping = _build_lc_mapping_from_graph(exec_globals)
        error_msg = f"{type(e).__name__}: {e}"
        error_msg = _add_execution_hints(error_msg, tb_str, code)
        return False, _enrich_error_with_constraint(error_msg, code, lc_mapping)

    # Verify executable constraints using dummy DataNode + AnswerSolver
    graph = _find_graph_in_globals(exec_globals)
    if graph is not None and graph.executableLCs:
        ok, err_msg = _verify_constraints(graph, code, exec_globals)
        if not ok:
            return False, err_msg

    return True, "Graph executed successfully."

# ---------------------------------------------------------------------------
# Graph validation
# ---------------------------------------------------------------------------

def validate_graph_coverage(code: str, concepts_json: dict) -> tuple[bool, list[str], list[str]]:
    required_concepts = [c["name"] for c in concepts_json.get("concepts", [])]
    required_relations = [r["name"] for r in concepts_json.get("relations", [])]

    missing_concepts = []
    for name in required_concepts:
        # Check if it appears as a Concept/EnumConcept name
        concept_pattern = rf"""(?:Concept|EnumConcept)\s*\(\s*['"]{ re.escape(name) }['"]"""
        # Check if it appears as a value inside an EnumConcept values list
        enum_value_pattern = rf"""values\s*=\s*\[.*['"]{ re.escape(name) }['"].*\]"""
        if not re.search(concept_pattern, code) and not re.search(enum_value_pattern, code):
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


def validate_single_constraint(graph_code: str, snippet: str) -> tuple[bool, str]:
    """Validate a single executable constraint snippet against the graph.

    Merges the snippet into the graph code inside a ``with graph:`` block
    and executes the combined code.  Returns ``(success, error_message)``.
    """
    combined = merge_executable_into_graph(graph_code, snippet)
    return execute_graph_code(combined)

# ---------------------------------------------------------------------------
# High-level pipeline helpers
# ---------------------------------------------------------------------------

def do_extract(questions: list[str], model: str | None = None) -> tuple[str, str, dict | None, str]:
    """Returns (prompt_sent, raw_response, parsed_json_or_None, model_used)."""
    questions_text = "\n".join(f"- {q}" for q in questions)
    prompt = get_prompt("extract_concepts.txt", questions=questions_text)
    raw, model_used = llm_chat([{"role": "user", "content": prompt}], model=model)
    return prompt, raw, extract_json_block(raw), model_used


def do_continue_extract(
    questions: list[str],
    existing_concepts_json: dict,
    model: str | None = None,
) -> tuple[str, str, dict | None, str]:
    """Continue concept extraction with additional questions.

    Takes the already-extracted concepts/relations and a new batch of questions,
    asks the LLM to add any missing concepts/relations.

    Returns (prompt_sent, raw_response, merged_json_or_None, model_used).
    """
    questions_text = "\n".join(f"- {q}" for q in questions)

    # Format existing concepts and relations for the prompt
    existing_concepts = "\n".join(
        f"- {c['name']}: {c.get('description', '')}"
        for c in existing_concepts_json.get("concepts", [])
    )
    existing_relations = "\n".join(
        f"- {r['name']} ({r.get('source', '')} → {r.get('target', '')}): {r.get('description', '')}"
        for r in existing_concepts_json.get("relations", [])
    )

    prompt = get_prompt(
        "continue_extract_concepts.txt",
        questions=questions_text,
        existing_concepts=existing_concepts or "(none)",
        existing_relations=existing_relations or "(none)",
    )
    raw, model_used = llm_chat([{"role": "user", "content": prompt}], model=model)
    return prompt, raw, extract_json_block(raw), model_used


def do_add_missing_concepts(
    question: str,
    missing_names: list[str],
    existing_concepts_json: dict,
    model: str | None = None,
) -> tuple[str, str, dict | None, str]:
    """Ask the LLM to add missing concept names to the ontology.

    Called when a constraint snippet fails with NameError — the names it
    references don't exist in the graph, so the concept extraction and graph
    need to be updated first.

    Returns (prompt_sent, raw_response, updated_json_or_None, model_used).
    """
    existing_concepts = "\n".join(
        f"- {c['name']}: {c.get('description', '')}"
        for c in existing_concepts_json.get("concepts", [])
    )
    existing_relations = "\n".join(
        f"- {r['name']} ({r.get('source', '')} → {r.get('target', '')}): {r.get('description', '')}"
        for r in existing_concepts_json.get("relations", [])
    )

    prompt = get_prompt(
        "add_missing_concepts.txt",
        question=question,
        missing_names=", ".join(missing_names),
        existing_concepts=existing_concepts or "(none)",
        existing_relations=existing_relations or "(none)",
    )
    raw, model_used = llm_chat([{"role": "user", "content": prompt}], temperature=0.1, model=model)
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


def _is_graph_fragment(snippet: str) -> bool:
    """Return True if the snippet looks like graph definitions instead of a constraint call."""
    if not snippet:
        return False
    # If it has constraint operators or execute(), it's valid
    if "execute(" in snippet or _CONSTRAINT_PATTERN.search(snippet):
        return False
    # Graph-definition markers
    graph_markers = ["Concept(", "object_node(name=", ".has_a(", ".contains(", "Graph("]
    hits = sum(1 for m in graph_markers if m in snippet)
    return hits >= 2


def do_generate_executable_single(question: str, graph_code: str, model: str | None = None) -> tuple[str, str, str | None, str]:
    """Returns (prompt_sent, raw_response, single_execute_snippet_or_None, model_used)."""
    prompt = get_prompt(
        "generate_executable.txt",
        graph_code=graph_code,
        question=question,
    )
    raw, model_used = llm_chat([{"role": "user", "content": prompt}], temperature=0.1, model=model)
    snippet = extract_code_block(raw)

    # Reject snippets that are graph definitions rather than constraint calls
    if snippet and _is_graph_fragment(snippet):
        import logging
        logging.getLogger(__name__).warning(
            "LLM returned graph definitions instead of execute() call — rejecting snippet"
        )
        return prompt, raw, None, model_used

    return prompt, raw, snippet, model_used


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