"""
CLI interface for DomiKnowS Graph Builder.

Modes:
  (default)      Batch mode — reads questions from a JSON dataset file
  --interactive  Interactive mode — prompts user for questions at the terminal
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from core import (
    MAX_FIX_ATTEMPTS,
    do_extract,
    do_fix,
    do_generate,
    do_generate_executable_single,
    do_validate_and_fix_coverage,
    execute_graph_code,
    merge_executable_into_graph,
    validate_graph_coverage,
)

# ---------------------------------------------------------------------------
# Performance log  (JSON-lines in logs/ subfolder for model comparison)
# ---------------------------------------------------------------------------

LOGS_DIR = Path(__file__).parent / "logs"

def _init_perf_log(dataset_path: str | None, graph_path: str | None, model_hint: str | None, total_questions: int) -> Path:
    """Create a new performance log file and write the header record. Returns the log path."""
    LOGS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = (model_hint or "default").replace("/", "_").replace("\\", "_")
    log_path = LOGS_DIR / f"perf_{model_tag}_{ts}.jsonl"

    header = {
        "type": "run_header",
        "timestamp": datetime.now().isoformat(),
        "dataset": str(dataset_path) if dataset_path else None,
        "graph_path": str(graph_path) if graph_path else None,
        "model_hint": model_hint,
        "total_questions": total_questions,
    }
    log_path.write_text(json.dumps(header) + "\n", encoding="utf-8")
    print(f"[PERF] Log file: {log_path}")
    return log_path


def _log_perf_entry(log_path: Path, entry: dict):
    """Append one JSON-line record to the performance log."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _log_perf_summary(log_path: Path, summary: dict):
    """Append the final summary record to the performance log."""
    summary["type"] = "run_summary"
    _log_perf_entry(log_path, summary)

# ---------------------------------------------------------------------------
# Question collection helpers
# ---------------------------------------------------------------------------

def collect_questions_interactive() -> list[str]:
    print("\n=== DomiKnowS Graph Builder (Interactive) ===")
    print("Enter your domain questions (one per line).")
    print("Type 'done' on a new line when finished.\n")

    questions: list[str] = []
    while True:
        line = input("Q: ").strip()
        if line.lower() == "done":
            break
        if line:
            questions.append(line)

    if not questions:
        print("No questions provided. Exiting.")
        sys.exit(0)

    return questions


def load_questions_from_dataset(
    dataset_path: str,
    keyword: str = "question",
    list_key: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """Load questions from a JSON dataset file.

    Supports:
      - A top-level JSON array of objects:  [{"question": "..."}, ...]
      - A top-level JSON object with a list key: {"questions": [{"question": "..."}, ...]}
      - A top-level JSON array of plain strings: ["Is there a red cube?", ...]

    Args:
        dataset_path: Path to the JSON file.
        keyword: The key inside each record that holds the question text.
        list_key: The top-level key that holds the list of records.
                  Auto-detected if None (tries common keys like "questions", "data").
        limit: Maximum number of questions to load (None = all).
    """
    path = Path(dataset_path)
    if not path.exists():
        print(f"ERROR: Dataset file not found: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Resolve the list of records
    records: list = []
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        if list_key and list_key in data:
            records = data[list_key]
        else:
            # Auto-detect common top-level keys
            for candidate in ["questions", "data", "samples", "items", "records", "dataset"]:
                if candidate in data and isinstance(data[candidate], list):
                    records = data[candidate]
                    print(f"Auto-detected list key: '{candidate}' ({len(records)} records)")
                    break
            if not records:
                print(f"ERROR: Could not find a list of records in {path}. "
                      f"Use --list-key to specify the top-level key.")
                sys.exit(1)
    else:
        print(f"ERROR: Unexpected JSON root type: {type(data).__name__}")
        sys.exit(1)

    if limit is not None:
        records = records[:limit]

    # Extract question strings
    questions: list[str] = []
    for i, rec in enumerate(records):
        if isinstance(rec, str):
            q = rec.strip()
        elif isinstance(rec, dict):
            q = rec.get(keyword, "")
            if not q:
                print(f"WARNING: Record {i} has no '{keyword}' field, skipping.")
                continue
            q = str(q).strip()
        else:
            print(f"WARNING: Record {i} has unexpected type {type(rec).__name__}, skipping.")
            continue
        if q:
            questions.append(q)

    if not questions:
        print("ERROR: No questions extracted from dataset.")
        sys.exit(1)

    return questions

# ---------------------------------------------------------------------------
# Shared pipeline
# ---------------------------------------------------------------------------

def build_graph(questions: list[str], args) -> str | None:
    print(f"\n[BUILD] Starting build_graph with {len(questions)} questions")
    # 1. Extract concepts and relations
    print("\n--- Step 1: Extracting concepts and relations ---")
    print(f"[BUILD] Calling do_extract...")
    prompt, raw, concepts_json = do_extract(questions)
    print(f"[BUILD] do_extract returned — raw length: {len(raw) if raw else 0}")
    print(f"\n[Prompt Sent]\n{prompt}\n")
    print(f"\n[Raw LLM Response]\n{raw}\n")

    if concepts_json is None:
        print("WARNING: Could not parse JSON. Using raw text.")
        concepts_json = {"raw": raw}
    else:
        print(f"Extracted {len(concepts_json.get('concepts', []))} concepts, "
              f"{len(concepts_json.get('relations', []))} relations.")

    # 2. Generate graph code
    print("\n--- Step 2: Generating DomiKnowS graph ---")

    if args.graph_path:
        code = Path(args.graph_path).read_text()
        print(f"[BUILD] Graph loaded from {args.graph_path} ({len(code)} chars) — generation skipped.")
    else:
        print(f"[BUILD] Calling do_generate...")
        log, code = do_generate(questions, concepts_json, model=args.model)
        print(f"[BUILD] do_generate returned — code length: {len(code) if code else 0}")
        for entry in log:
            if entry["type"] == "llm_interaction":
                print(f"\n[Graph Prompt - Iteration {entry['iteration']}]\n{entry['prompt_sent']}\n")
                print(f"\n[Graph Response - Iteration {entry['iteration']}]\n{entry['raw_response']}\n")
            else:
                print(f"\n⚠️  {entry['message']}")

    if not code:
        print("Failed to generate graph code.")
        return None

    # 2b. Validate coverage
    for cov_attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        all_present, missing_c, missing_r = validate_graph_coverage(code, concepts_json)
        if all_present:
            print("✅ All extracted concepts and relations are present in the graph.")
            break

        print(f"\n⚠️  Coverage check failed (attempt {cov_attempt}/{MAX_FIX_ATTEMPTS}):")
        if missing_c:
            print(f"  Missing concepts: {', '.join(missing_c)}")
        if missing_r:
            print(f"  Missing relations: {', '.join(missing_r)}")

        prompt, raw, fixed = do_validate_and_fix_coverage(code, concepts_json)
        print(f"\n[Prompt Sent]\n{prompt}\n")
        print(f"\n[Raw LLM Response]\n{raw}\n")
        if fixed:
            code = fixed
        else:
            print("LLM could not fix coverage.")
            break

    print("=" * 60)
    print("Generated code:")
    print("=" * 60)
    print(code)
    print("=" * 60)

    # 3. Execute and fix loop
    print(f"\n[BUILD] Starting execute+fix loop (max {MAX_FIX_ATTEMPTS} attempts)")
    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        print(f"[BUILD] Execute attempt {attempt}/{MAX_FIX_ATTEMPTS}...")
        success, output = execute_graph_code(code)
        print(f"[BUILD] Execute result: success={success}, output length={len(output) if output else 0}")

        if success:
            print(f"\n✅ Graph created successfully on attempt {attempt}!")
            return code

        print(f"\n❌ Execution failed (attempt {attempt}/{MAX_FIX_ATTEMPTS}):")
        print(output)

        if attempt < MAX_FIX_ATTEMPTS:
            print(f"\n--- Fix attempt {attempt}/{MAX_FIX_ATTEMPTS} ---")
            log, fixed_code = do_fix(code, output)
            for entry in log:
                if entry["type"] == "llm_interaction":
                    print(f"\n[Prompt Sent - Iteration {entry['iteration']}]\n{entry['prompt_sent']}\n")
                    print(f"\n[Raw LLM Response - Iteration {entry['iteration']}]\n{entry['raw_response']}\n")
                else:
                    print(f"\n⚠️  {entry['message']}")

            if fixed_code:
                code = fixed_code
                print("Corrected code:")
                print(code)
                print("=" * 60)

    print(f"\n❌ Failed to produce a valid graph after {MAX_FIX_ATTEMPTS} attempts.")
    return None

# ---------------------------------------------------------------------------
# Interactive mode  (original behaviour)
# ---------------------------------------------------------------------------

def run_interactive(args):
    run_start = time.time()
    questions = collect_questions_interactive()
    print(f"\n[INTERACTIVE] Collected {len(questions)} questions.")
    print(f"[INTERACTIVE] Total questions: {len(questions)}")
    display_limit = min(len(questions), 25)
    for i in range(display_limit):
        print(f"  Q{i + 1}: {questions[i]}")
    if len(questions) > 25:
        print(f"  ... and {len(questions) - 25} more questions")

    # If graph is provided, skip concept extraction and graph generation
    if args.graph_path:
        graph_code = Path(args.graph_path).read_text()
        print(f"\n[INTERACTIVE] Graph loaded from {args.graph_path} ({len(graph_code)} chars) — skipping concept extraction & generation")
        print(f"[INTERACTIVE] Jumping directly to executable constraint generation")
    else:
        print(f"\n[INTERACTIVE] No --graph-path provided, running full pipeline (extract → generate → execute)")
        graph_code = build_graph(questions, args)
    if graph_code is None:
        print(f"[INTERACTIVE] ERROR: graph_code is None — aborting")
        sys.exit(1)

    print(f"\n[INTERACTIVE] Graph code ready ({len(graph_code)} chars)")

    # Initialize performance log
    perf_log = _init_perf_log(
        dataset_path=None,
        graph_path=args.graph_path,
        model_hint=getattr(args, "model", None),
        total_questions=len(questions),
    )

    # Generate executable constraints per question
    print(f"\n--- Step 3: Adding questions as executable constraints ---")
    print(f"[INTERACTIVE] Processing {len(questions)} questions individually...\n")

    snippets: list[str] = []
    for i, question in enumerate(questions):
        print(f"\n[INTERACTIVE] [Question {i + 1}/{len(questions)}: {question}]")
        t0 = time.time()
        prompt, raw, snippet, _model = do_generate_executable_single(question, graph_code)
        elapsed = time.time() - t0
        print(f"[INTERACTIVE]   raw response: {len(raw) if raw else 0} chars, model={_model}, time: {elapsed:.1f}s")
        print(f"\n[Prompt Sent]\n{prompt}\n")
        print(f"\n[Raw LLM Response]\n{raw}\n")
        if snippet:
            snippets.append(f"# Q{i + 1}: {question}\n{snippet.strip()}")
            print(f"  ✅ snippet: {len(snippet)} chars, time: {elapsed:.1f}s")
            print(f"  >> {snippet.strip()}")
        else:
            print(f"  ⚠️  Could not extract execute() call (time: {elapsed:.1f}s)")
            print(f"[INTERACTIVE]   Raw response preview: {(raw or '')[:200]}")

        _log_perf_entry(perf_log, {
            "type": "question",
            "index": i + 1,
            "question": question,
            "success": snippet is not None,
            "error": None,
            "model": _model,
            "time_s": round(elapsed, 2),
            "raw_response_chars": len(raw) if raw else 0,
            "snippet_chars": len(snippet) if snippet else 0,
            "snippet": snippet.strip() if snippet else None,
        })

    total_time = time.time() - run_start
    print(f"\n[INTERACTIVE] Constraint generation complete: {len(snippets)}/{len(questions)} encoded, total time: {total_time:.1f}s")

    _log_perf_summary(perf_log, {
        "total_questions": len(questions),
        "encoded": len(snippets),
        "failed": len(questions) - len(snippets),
        "total_time_s": round(total_time, 2),
        "avg_time_per_question_s": round(total_time / len(questions), 2) if questions else 0,
        "dataset": None,
        "graph_path": str(args.graph_path) if args.graph_path else None,
    })
    print(f"[PERF] Summary written to {perf_log}")

    if not snippets:
        print("[INTERACTIVE] ERROR: No snippets — aborting")
        print("Failed to generate executable constraints. Exiting.")
        sys.exit(1)

    print(f"[INTERACTIVE] Merging {len(snippets)} snippets into graph code...")
    combined = "\n\n".join(snippets)
    code = merge_executable_into_graph(graph_code, combined)
    print(f"[INTERACTIVE] Merged code ready ({len(code)} chars)")
    print("=" * 60)
    print("Full code with executable constraints:")
    print("=" * 60)
    print(code)
    print("=" * 60)

    # Execute and fix loop for full code
    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        success, output = execute_graph_code(code)

        if success:
            print(f"\n✅ Graph with executable constraints created on attempt {attempt}!")
            print("\n--- Final code ---")
            print(code)
            return

        print(f"\n❌ Execution failed (attempt {attempt}/{MAX_FIX_ATTEMPTS}):")
        print(output)

        if attempt < MAX_FIX_ATTEMPTS:
            log, fixed_code = do_fix(code, output, expected_execute_count=len(questions))
            for entry in log:
                if entry["type"] == "llm_interaction":
                    print(f"\n[Prompt Sent - Iteration {entry['iteration']}]\n{entry['prompt_sent']}\n")
                    print(f"\n[Raw LLM Response - Iteration {entry['iteration']}]\n{entry['raw_response']}\n")
                else:
                    print(f"\n⚠️  {entry['message']}")

            if fixed_code:
                code = fixed_code

    print(f"\n❌ Failed after {MAX_FIX_ATTEMPTS} attempts.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Batch mode  (new default — reads from dataset JSON)
# ---------------------------------------------------------------------------

def run_batch(args):
    run_start = time.time()
    questions = load_questions_from_dataset(
        dataset_path=args.dataset,
        keyword=args.keyword,
        list_key=args.list_key,
        limit=args.limit,
    )
    print(f"\n[BATCH] Loaded {len(questions)} questions from {args.dataset}")
    print(f"[BATCH] Total questions: {len(questions)}")
    display_limit = min(len(questions), 25)
    for i in range(display_limit):
        print(f"  Q{i + 1}: {questions[i]}")
    if len(questions) > 25:
        print(f"  ... and {len(questions) - 25} more questions")

    # If graph is provided, skip concept extraction and graph generation
    if args.graph_path:
        graph_code = Path(args.graph_path).read_text()
        print(f"\n[BATCH] Graph loaded from {args.graph_path} ({len(graph_code)} chars) — skipping concept extraction & generation")
        print(f"[BATCH] Jumping directly to executable constraint generation")
    else:
        print(f"\n[BATCH] No --graph-path provided, running full pipeline (extract → generate → execute)")
        graph_code = build_graph(questions, args)
    if graph_code is None:
        print(f"[BATCH] ERROR: graph_code is None — aborting")
        sys.exit(1)

    print(f"\n[BATCH] Graph code ready ({len(graph_code)} chars)")

    # Initialize performance log
    perf_log = _init_perf_log(
        dataset_path=args.dataset,
        graph_path=args.graph_path,
        model_hint=getattr(args, "model", None),
        total_questions=len(questions),
    )

    # Process each question individually and collect results
    print(f"\n--- Step 3: Encoding questions as executable constraints ---")
    print(f"[BATCH] Processing {len(questions)} questions individually...\n")

    results: list[dict] = []  # {"question": str, "snippet": str|None, "success": bool}

    for i, question in enumerate(questions):
        print(f"[BATCH] [{i + 1}/{len(questions)}] {question}")
        t0 = time.time()
        try:
            _prompt, _raw, snippet, _model = do_generate_executable_single(question, graph_code)
            elapsed = time.time() - t0
            print(f"[BATCH]   raw response: {len(_raw) if _raw else 0} chars, model={_model}, time: {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ❌ LLM error (time: {elapsed:.1f}s): {e}")
            _log_perf_entry(perf_log, {
                "type": "question",
                "index": i + 1,
                "question": question,
                "success": False,
                "error": str(e),
                "model": None,
                "time_s": round(elapsed, 2),
                "raw_response_chars": 0,
                "snippet_chars": 0,
                "snippet": None,
            })
            results.append({"question": question, "snippet": None, "success": False})
            continue

        if snippet:
            print(f"  ✅ snippet: {len(snippet)} chars, time: {elapsed:.1f}s")
            print(f"  >> {snippet.strip()}")
            results.append({"question": question, "snippet": snippet.strip(), "success": True})
        else:
            print(f"  ⚠️  Could not extract execute() call (time: {elapsed:.1f}s)")
            print(f"[BATCH]   raw response preview: {(_raw or '')[:200]}")
            results.append({"question": question, "snippet": None, "success": False})

        _log_perf_entry(perf_log, {
            "type": "question",
            "index": i + 1,
            "question": question,
            "success": snippet is not None,
            "error": None,
            "model": _model,
            "time_s": round(elapsed, 2),
            "raw_response_chars": len(_raw) if _raw else 0,
            "snippet_chars": len(snippet) if snippet else 0,
            "snippet": snippet.strip() if snippet else None,
        })

    # Build output Python file
    output_path = Path(args.output)
    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count
    total_time = time.time() - run_start
    print(f"\n[BATCH] Constraint generation complete: {success_count}/{len(results)} encoded, {fail_count} failed, total time: {total_time:.1f}s")
    print(f"[BATCH] Building output file: {output_path}")

    # Write performance summary
    _log_perf_summary(perf_log, {
        "total_questions": len(results),
        "encoded": success_count,
        "failed": fail_count,
        "total_time_s": round(total_time, 2),
        "avg_time_per_question_s": round(total_time / len(results), 2) if results else 0,
        "dataset": str(args.dataset),
        "graph_path": str(args.graph_path) if args.graph_path else None,
        "output_path": str(output_path),
    })
    print(f"[PERF] Summary written to {perf_log}")

    lines: list[str] = []
    lines.append('"""')
    lines.append(f"Auto-generated DomiKnowS graph with executable constraints.")
    lines.append(f"Source dataset: {args.dataset}")
    lines.append(f"Total questions: {len(results)}  |  Encoded: {success_count}  |  Failed: {fail_count}")
    lines.append('"""')
    lines.append("")

    # Graph code (without imports — they go at the top)
    lines.append(graph_code.rstrip())
    lines.append("")
    lines.append("")

    # Executable constraints block
    lines.append("with graph:")

    for i, result in enumerate(results):
        lines.append("")
        if result["success"]:
            lines.append(f"    # Q{i + 1}: {result['question']}")
            for snippet_line in result["snippet"].splitlines():
                lines.append(f"    {snippet_line}")
        else:
            lines.append(f"    # Q{i + 1}: {result['question']}")
            lines.append(f"    # FAILED — could not encode this question")

    lines.append("")

    output_code = "\n".join(lines)

    output_path.write_text(output_code, encoding="utf-8")
    print(f"\n{'=' * 60}")
    print(f"Output written to: {output_path}")
    print(f"  Questions encoded: {success_count}/{len(results)}")
    if fail_count:
        print(f"  Questions failed:  {fail_count} (marked as comments in output)")
    print(f"{'=' * 60}")

    # Optionally validate the full code
    if args.no_validate:
        print("Skipping validation (--no-validate).")
        return

    print("\n--- Step 4: Validating full code ---")
    full_code = output_code
    # Extract only the successful snippets for expected execute count
    expected = success_count

    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        success, output = execute_graph_code(full_code)

        if success:
            print(f"\n✅ Full code validated on attempt {attempt}!")
            # Overwrite with validated code
            output_path.write_text(full_code, encoding="utf-8")
            return

        print(f"\n❌ Execution failed (attempt {attempt}/{MAX_FIX_ATTEMPTS}):")
        print(output)

        if attempt < MAX_FIX_ATTEMPTS:
            log, fixed_code = do_fix(full_code, output, expected_execute_count=expected)
            for entry in log:
                if entry["type"] == "llm_interaction":
                    print(f"\n[Fix Iteration {entry['iteration']}]")
                else:
                    print(f"  ⚠️  {entry['message']}")

            if fixed_code:
                full_code = fixed_code
                output_path.write_text(full_code, encoding="utf-8")

    print(f"\n⚠️  Validation failed after {MAX_FIX_ATTEMPTS} attempts. Output file may contain errors.")

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="DomiKnowS Graph Builder — generate graphs from natural language questions",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run in interactive mode (prompt for questions at the terminal)",
    )

    # Batch mode arguments
    batch = parser.add_argument_group("batch mode (default)")
    
    batch.add_argument(
        "--graph-path",
        metavar="PATH",
        help="Path to an existing graph .py file; skips graph generation step",
    )
    batch.add_argument(
        "dataset", nargs="?", default=None,
        help="Path to the JSON dataset file containing questions",
    )
    batch.add_argument(
        "--keyword", default="question",
        help="JSON key that holds the question text in each record (default: 'question')",
    )
    batch.add_argument(
        "--list-key", default=None, dest="list_key",
        help="Top-level JSON key that holds the list of records (auto-detected if omitted)",
    )
    batch.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of questions to process",
    )
    batch.add_argument(
        "-o", "--output", default="output_graph.py",
        help="Output Python file path (default: output_graph.py)",
    )
    batch.add_argument(
        "--no-validate", action="store_true", dest="no_validate",
        help="Skip execution/validation of the generated output",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.interactive:
        run_interactive(args)
    else:
        if not args.dataset:
            print("ERROR: In batch mode, a dataset file path is required.")
            print("Usage: graph_builder.py <dataset.json> [options]")
            print("       graph_builder.py --interactive")
            sys.exit(1)
        run_batch(args)


if __name__ == "__main__":
    import os, signal

    def _force_exit(*_args):
        """Hard-kill on Ctrl+C — os._exit bypasses blocked socket I/O on Windows."""
        try:
            print("\n\n⚠️  Interrupted by user (Ctrl+C)")
        except Exception:
            pass
        os._exit(130)

    signal.signal(signal.SIGINT, _force_exit)
    try:
        main()
    except KeyboardInterrupt:
        _force_exit()