"""
Flask GUI for DomiKnowS Graph Builder.
"""

import logging
import os

from flask import Flask, render_template, request, jsonify

from core import (
    DEFAULT_MODEL,
    do_extract,
    do_fix,
    do_generate,
    do_generate_executable,
    do_generate_executable_single,
    do_validate_and_fix_coverage,
    execute_graph_code,
    get_available_models,
    merge_executable_into_graph,
    validate_graph_coverage,
)

app = Flask(__name__)
app.secret_key = "domiknows-graph-gen-secret"

# Ensure module-level info logs (e.g., in core.py) are visible when running the Flask app.
_level_name = os.getenv("GRAPH_GEN_LOG_LEVEL", "INFO").upper()
_level = getattr(logging, _level_name, logging.INFO)
logging.basicConfig(
    level=_level,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    force=True,
)
app.logger.setLevel(_level)


def _get_model() -> str | None:
    """Extract model override from the request JSON body (returns None to use default)."""
    data = request.get_json(silent=True) or {}
    return data.get("model") or None

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config", methods=["GET"])
def api_config():
    return jsonify({
        "default_model": DEFAULT_MODEL,
        "models": get_available_models(),
    })


@app.route("/api/load-graph", methods=["POST"])
def load_graph():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".py"):
        return jsonify({"error": "Only .py files are accepted"}), 400
    try:
        code = f.read().decode("utf-8")
    except Exception as e:
        return jsonify({"error": f"Could not read file: {e}"}), 400
    return jsonify({"graph_code": code})

@app.route("/api/extract", methods=["POST"])
def api_extract():
    data = request.get_json()
    questions = data.get("questions", [])
    model = data.get("model")
    if not questions:
        return jsonify({"error": "No questions provided"}), 400

    try:
        prompt, raw, parsed, model_used = do_extract(questions, model=model)
    except Exception as e:
        return jsonify({"error": f"LLM request failed: {e}"}), 502

    return jsonify({"prompt_sent": prompt, "raw_response": raw, "concepts_json": parsed, "model": model_used})


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json()
    questions = data.get("questions", [])
    concepts_json = data.get("concepts_json", {})
    model = data.get("model")

    try:
        interactions, code = do_generate(questions, concepts_json, model=model)
    except Exception as e:
        return jsonify({"error": f"LLM request failed: {e}"}), 502

    return jsonify({"log": interactions, "code": code})


@app.route("/api/validate_coverage", methods=["POST"])
def api_validate_coverage():
    data = request.get_json()
    code = data.get("code", "")
    concepts_json = data.get("concepts_json", {})
    model = data.get("model")

    all_present, missing_c, missing_r = validate_graph_coverage(code, concepts_json)

    if all_present:
        return jsonify({"valid": True, "missing_concepts": [], "missing_relations": []})

    try:
        prompt, raw, fixed_code, model_used = do_validate_and_fix_coverage(code, concepts_json, model=model)
    except Exception as e:
        return jsonify({"error": f"LLM request failed: {e}"}), 502

    return jsonify({
        "valid": False,
        "missing_concepts": missing_c,
        "missing_relations": missing_r,
        "prompt_sent": prompt,
        "raw_response": raw,
        "code": fixed_code if fixed_code else code,
        "model": model_used,
    })


@app.route("/api/execute", methods=["POST"])
def api_execute():
    data = request.get_json()
    code = data.get("code", "")
    if not code:
        return jsonify({"error": "No code provided"}), 400

    success, output = execute_graph_code(code)
    return jsonify({"success": success, "output": output})


@app.route("/api/fix", methods=["POST"])
def api_fix():
    data = request.get_json()
    code = data.get("code", "")
    error = data.get("error", "")
    expected_execute_count = data.get("expected_execute_count")
    model = data.get("model")

    try:
        interactions, fixed_code = do_fix(code, error, expected_execute_count=expected_execute_count, model=model)
    except Exception as e:
        return jsonify({"error": f"LLM request failed: {e}"}), 502

    return jsonify({
        "log": interactions,
        "code": fixed_code if fixed_code else code,
    })


@app.route("/api/generate_executable_single", methods=["POST"])
def api_generate_executable_single():
    data = request.get_json()
    question = data.get("question", "")
    graph_code = data.get("graph_code", "")
    model = data.get("model")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        prompt, raw, snippet, model_used = do_generate_executable_single(question, graph_code, model=model)
    except Exception as e:
        return jsonify({"error": f"LLM request failed: {e}"}), 502

    return jsonify({"prompt_sent": prompt, "raw_response": raw, "snippet": snippet, "model": model_used})


@app.route("/api/generate_executable", methods=["POST"])
def api_generate_executable():
    data = request.get_json()
    questions = data.get("questions", [])
    graph_code = data.get("graph_code", "")
    model = data.get("model")

    try:
        log, combined_snippet = do_generate_executable(questions, graph_code, model=model)
    except Exception as e:
        return jsonify({"error": f"LLM request failed: {e}"}), 502

    return jsonify({"log": log, "snippet": combined_snippet})


@app.route("/api/merge_executable", methods=["POST"])
def api_merge_executable():
    data = request.get_json()
    graph_code = data.get("graph_code", "")
    snippet = data.get("snippet", "")
    merged = merge_executable_into_graph(graph_code, snippet)
    return jsonify({"code": merged})


if __name__ == "__main__":
    app.run(debug=True, port=5001)