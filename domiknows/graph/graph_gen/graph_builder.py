"""
CLI interface for DomiKnowS Graph Builder.

Modes:
  (default)      Batch mode — reads questions from a JSON dataset file
  --interactive  Interactive mode — prompts user for questions at the terminal
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from core import (
    CONFIG_PATH,
    DEFAULT_MODEL,
    MODELS,
    MAX_FIX_ATTEMPTS,
    do_add_missing_concepts,
    do_continue_extract,
    do_extract,
    do_fix,
    do_generate,
    do_generate_executable_single,
    do_validate_and_fix_coverage,
    execute_graph_code,
    merge_executable_into_graph,
    validate_graph_coverage,
    validate_single_constraint,
)

EXTRACT_BATCH_SIZE = 20  # default batch size for concept extraction
MAX_GRAPH_REPAIRS = None  # no limit — rebuild graph as many times as needed

# ---------------------------------------------------------------------------
# Snippet extraction helpers
# ---------------------------------------------------------------------------

def _extract_missing_names(val_error: str) -> list[str]:
    """Extract missing concept/variable names from a NameError message.

    Returns a list of names, or an empty list if the error is not a NameError.
    """
    names = re.findall(r"NameError: name '(\w+)' is not defined", val_error)
    return list(dict.fromkeys(names))  # deduplicate, preserve order


def _rebuild_graph(
    questions: list[str],
    concepts_json: dict,
    missing_names: list[str],
    trigger_question: str,
    args,
) -> tuple[str | None, dict]:
    """Update concept extraction and regenerate the graph code.

    1. Ask LLM to add missing names to the concept list.
    2. Regenerate graph code from updated concepts.
    3. Execute+fix the new graph.

    Returns (new_graph_code_or_None, updated_concepts_json).
    """
    print(f"\n{'=' * 60}")
    print(f"[GRAPH-REPAIR] Missing names detected: {', '.join(missing_names)}")
    print(f"[GRAPH-REPAIR] Triggered by question: {trigger_question}")
    print(f"[GRAPH-REPAIR] Updating concept extraction...")

    # 1. Ask LLM to add missing concepts
    prompt, raw, updated_json, _model = do_add_missing_concepts(
        trigger_question, missing_names, concepts_json, model=args.model,
    )
    print(f"[GRAPH-REPAIR] LLM response: {len(raw) if raw else 0} chars")

    if updated_json is None:
        print(f"[GRAPH-REPAIR] ❌ Could not parse updated concepts JSON")
        return None, concepts_json

    n_before = len(concepts_json.get("concepts", []))
    n_after = len(updated_json.get("concepts", []))
    print(f"[GRAPH-REPAIR] Concepts: {n_before} → {n_after} (+{n_after - n_before})")
    concepts_json = updated_json

    # 2. Regenerate graph code
    print(f"[GRAPH-REPAIR] Regenerating graph code...")
    log, code = do_generate(questions, concepts_json, model=args.model)
    for entry in log:
        if entry["type"] == "llm_interaction":
            print(f"[GRAPH-REPAIR] Graph generation iteration {entry['iteration']}")
        else:
            print(f"[GRAPH-REPAIR] ⚠️  {entry['message']}")

    if not code:
        print(f"[GRAPH-REPAIR] ❌ Failed to generate graph code")
        return None, concepts_json

    # 3. Execute and fix loop for the new graph
    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        success, output = execute_graph_code(code)
        if success:
            print(f"[GRAPH-REPAIR] ✅ New graph validated on attempt {attempt}")
            print(f"{'=' * 60}\n")
            return code, concepts_json

        print(f"[GRAPH-REPAIR] ❌ Graph execution failed (attempt {attempt}): {output[:200]}")
        if attempt < MAX_FIX_ATTEMPTS:
            _, fixed_code = do_fix(code, output)
            if fixed_code:
                code = fixed_code
            else:
                break

    print(f"[GRAPH-REPAIR] ❌ Could not produce a valid graph after repairs")
    print(f"{'=' * 60}\n")
    return None, concepts_json

def _extract_snippet_from_fixed(fixed_code: str, graph_code: str) -> str | None:
    """Extract the executable constraint snippet from LLM-fixed full code.

    The fixed code contains the graph definition followed by a
    ``with graph:`` block with the constraint(s).  We need to pull out
    just the constraint lines from inside that block.

    Strategy:
    1. Find the *last* ``with graph:`` line (constraints are appended there).
    2. Collect all indented lines after it as the snippet.
    3. De-indent by the block's indentation level.
    """
    lines = fixed_code.splitlines()

    # Find the last "with graph:" line
    with_graph_idx = None
    for idx, line in enumerate(lines):
        if re.match(r'\s*with\s+graph\s*:', line):
            with_graph_idx = idx

    if with_graph_idx is None:
        return None

    # Collect indented lines after "with graph:"
    snippet_lines = []
    for line in lines[with_graph_idx + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            snippet_lines.append(stripped)
            continue
        # Check if still indented (inside the with block)
        if line and not line[0].isspace():
            break
        # De-indent: remove leading 4 spaces or 1 tab
        if line.startswith('    '):
            snippet_lines.append(line[4:])
        elif line.startswith('\t'):
            snippet_lines.append(line[1:])
        else:
            snippet_lines.append(line.lstrip())

    # Strip leading/trailing blank lines
    while snippet_lines and not snippet_lines[0].strip():
        snippet_lines.pop(0)
    while snippet_lines and not snippet_lines[-1].strip():
        snippet_lines.pop()

    if not snippet_lines:
        return None

    return "\n".join(snippet_lines)


# ---------------------------------------------------------------------------
# Performance log  (JSON-lines in logs/ subfolder for model comparison)
# ---------------------------------------------------------------------------

LOGS_DIR = Path(__file__).parent / "logs"

def _init_perf_log(dataset_path: str | None, graph_path: str | None, model_hint: str | None, total_questions: int) -> Path:
    """Create a new performance log file and write the header record. Returns the log path."""
    LOGS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = re.sub(r'[\\/:*?"<>|]', "_", model_hint or "default")
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
# Output path helpers
# ---------------------------------------------------------------------------

def _derive_output_paths(base_output: str) -> tuple[Path, Path]:
    """Derive concepts JSON and graph-only Python paths from the main output path.

    'output_graph_gpt4_20260323.py' →
      ('output_graph_gpt4_20260323_concepts.json',
       'output_graph_gpt4_20260323_graph.py')
    """
    base = Path(base_output)
    stem = base.stem
    parent = base.parent
    return parent / f"{stem}_concepts.json", parent / f"{stem}_graph.py"


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
# Question pre-filter for continue extraction
# ---------------------------------------------------------------------------

# Common stop-words that never indicate a new domain concept
_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could of in on at to for with by from and or not "
    "no nor but if then else when where how what which who whom whose that this "
    "these those there here than so as it its they them their he she his her we "
    "our you your i my me us any all each every some many much more most few "
    "less least very too also just only still already even about into over after "
    "before between under above through during against same other another such "
    "both either neither while until because since although though whether "
    "number count many equal less greater fewer more than does exist exists "
    "thing things object objects".split()
)


def _build_known_terms(concepts_json: dict) -> set[str]:
    """Build a set of lower-case tokens that are already covered by the ontology."""
    terms: set[str] = set()
    for c in concepts_json.get("concepts", []):
        # Add concept name tokens (e.g. "surface_property" → {"surface", "property"})
        for tok in re.split(r"[_\s]+", c["name"].lower()):
            terms.add(tok)
        # Add description words
        for tok in re.findall(r"[a-z]+", c.get("description", "").lower()):
            terms.add(tok)
        # Add enum values if present
        for val in c.get("values", []):
            for tok in re.split(r"[_\s]+", str(val).lower()):
                terms.add(tok)
    for r in concepts_json.get("relations", []):
        for tok in re.split(r"[_\s]+", r["name"].lower()):
            terms.add(tok)
        for tok in re.findall(r"[a-z]+", r.get("description", "").lower()):
            terms.add(tok)
    return terms


def _question_may_add_concepts(question: str, known_terms: set[str]) -> bool:
    """Return True if the question contains content words not yet in known_terms.

    This is a cheap text-based heuristic: if every meaningful word in the
    question is already covered by the existing ontology, the question is
    unlikely to contribute new concepts and can be skipped.
    """
    words = set(re.findall(r"[a-z]+", question.lower()))
    novel = words - known_terms - _STOP_WORDS
    return len(novel) > 0


# ---------------------------------------------------------------------------
# Shared pipeline
# ---------------------------------------------------------------------------

def build_graph(questions: list[str], args) -> tuple[str | None, dict | None]:
    batch_size = getattr(args, "extract_batch_size", EXTRACT_BATCH_SIZE) or EXTRACT_BATCH_SIZE
    print(f"\n[BUILD] Starting build_graph with {len(questions)} questions (extract batch size: {batch_size})")

    # 1. Extract concepts and relations (first batch)
    first_batch = questions[:batch_size]
    print(f"\n--- Step 1: Extracting concepts and relations (batch 1 — {len(first_batch)} questions) ---")
    print(f"[BUILD] Calling do_extract with first {len(first_batch)} questions...")
    prompt, raw, concepts_json, _model = do_extract(first_batch, model=args.model)
    print(f"[BUILD] do_extract returned — raw length: {len(raw) if raw else 0}")
    print(f"\n[Prompt Sent]\n{prompt}\n")
    print(f"\n[Raw LLM Response]\n{raw}\n")

    if concepts_json is None:
        print("WARNING: Could not parse JSON. Using raw text.")
        concepts_json = {"raw": raw}
    else:
        print(f"Extracted {len(concepts_json.get('concepts', []))} concepts, "
              f"{len(concepts_json.get('relations', []))} relations.")

    # 1b. Continue concept extraction for remaining questions
    #     Pre-filter each question: only include it in a batch if it may
    #     contribute concepts not already in the ontology.
    remaining = questions[batch_size:]
    if remaining and concepts_json and "raw" not in concepts_json:
        print(f"\n--- Step 1b: Continue concept extraction ({len(remaining)} remaining questions) ---")

        # Pre-filter: assess each remaining question
        known_terms = _build_known_terms(concepts_json)
        candidates: list[str] = []
        skipped = 0
        for q in remaining:
            if _question_may_add_concepts(q, known_terms):
                candidates.append(q)
            else:
                skipped += 1

        print(f"[BUILD] Pre-filter: {len(candidates)} questions may contribute new concepts, "
              f"{skipped} skipped (already covered)")

        if candidates:
            batch_num = 1  # continues from initial batch 1
            while candidates:
                # Take the next batch
                batch = candidates[:batch_size]
                candidates = candidates[batch_size:]
                batch_num += 1

                n_concepts_before = len(concepts_json.get("concepts", []))
                n_relations_before = len(concepts_json.get("relations", []))

                print(f"\n[BUILD] Continue extraction batch {batch_num} — {len(batch)} questions")
                prompt, raw, updated_json, _model = do_continue_extract(
                    batch, concepts_json, model=args.model
                )
                print(f"[BUILD] do_continue_extract returned — raw length: {len(raw) if raw else 0}")
                print(f"\n[Prompt Sent]\n{prompt}\n")
                print(f"\n[Raw LLM Response]\n{raw}\n")

                if updated_json is not None:
                    n_concepts_after = len(updated_json.get("concepts", []))
                    n_relations_after = len(updated_json.get("relations", []))
                    new_concepts = n_concepts_after - n_concepts_before
                    new_relations = n_relations_after - n_relations_before
                    print(f"  Batch {batch_num}: +{new_concepts} concepts, +{new_relations} relations "
                          f"(total: {n_concepts_after} concepts, {n_relations_after} relations)")
                    concepts_json = updated_json

                    # Rebuild known terms and re-filter remaining candidates
                    # so questions now covered by newly added concepts are skipped
                    if candidates:
                        known_terms = _build_known_terms(concepts_json)
                        before_refilter = len(candidates)
                        candidates = [q for q in candidates if _question_may_add_concepts(q, known_terms)]
                        newly_skipped = before_refilter - len(candidates)
                        if newly_skipped:
                            print(f"  Re-filter: {newly_skipped} more questions now covered, "
                                  f"{len(candidates)} candidates remaining")
                else:
                    print(f"  ⚠️  Batch {batch_num}: Could not parse response — keeping existing concepts.")

        print(f"\n[BUILD] Continue extraction complete — "
              f"final: {len(concepts_json.get('concepts', []))} concepts, "
              f"{len(concepts_json.get('relations', []))} relations")

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
        return None, concepts_json

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
            return code, concepts_json

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
    return None, concepts_json

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
        concepts_json = None  # extraction was skipped
        print(f"\n[INTERACTIVE] Graph loaded from {args.graph_path} ({len(graph_code)} chars) — skipping concept extraction & generation")
        print(f"[INTERACTIVE] Jumping directly to executable constraint generation")
    else:
        print(f"\n[INTERACTIVE] No --graph-path provided, running full pipeline (extract → generate → execute)")
        graph_code, concepts_json = build_graph(questions, args)
    if graph_code is None:
        print(f"[INTERACTIVE] ERROR: graph_code is None — aborting")
        sys.exit(1)

    print(f"\n[INTERACTIVE] Graph code ready ({len(graph_code)} chars)")

    # Save intermediate files (concepts JSON + graph-only code)
    concepts_path, graph_only_path = _derive_output_paths(args.output)

    if concepts_json is not None:
        concepts_path.write_text(
            json.dumps(concepts_json, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[INTERACTIVE] Concepts/relations saved to: {concepts_path}")

    graph_only_path.write_text(graph_code, encoding="utf-8")
    print(f"[INTERACTIVE] Graph code saved to: {graph_only_path}")
    print(f"[INTERACTIVE]   (usable as --graph-path for subsequent runs)")

    if getattr(args, 'graph_only', False):
        print(f"\n{'=' * 60}")
        print(f"--graph-only: stopping after graph generation.")
        if concepts_json is not None:
            print(f"  Concepts: {concepts_path}")
        print(f"  Graph:    {graph_only_path}")
        print(f"{'=' * 60}")
        return

    # Initialize performance log
    perf_log = _init_perf_log(
        dataset_path=None,
        graph_path=args.graph_path,
        model_hint=args.model,
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

    if getattr(args, 'graph_only', False) and args.graph_path:
        print("WARNING: --graph-only has no effect with --graph-path "
              "(graph already exists, nothing to generate).")

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
        concepts_json = None  # extraction was skipped
        print(f"\n[BATCH] Graph loaded from {args.graph_path} ({len(graph_code)} chars) — skipping concept extraction & generation")
        print(f"[BATCH] Jumping directly to executable constraint generation")
    else:
        print(f"\n[BATCH] No --graph-path provided, running full pipeline (extract → generate → execute)")
        graph_code, concepts_json = build_graph(questions, args)
    if graph_code is None:
        print(f"[BATCH] ERROR: graph_code is None — aborting")
        sys.exit(1)

    print(f"\n[BATCH] Graph code ready ({len(graph_code)} chars)")

    # Save intermediate files (concepts JSON + graph-only code)
    concepts_path, graph_only_path = _derive_output_paths(args.output)

    if concepts_json is not None:
        concepts_path.write_text(
            json.dumps(concepts_json, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[BATCH] Concepts/relations saved to: {concepts_path}")

    graph_only_path.write_text(graph_code, encoding="utf-8")
    print(f"[BATCH] Graph code saved to: {graph_only_path}")
    print(f"[BATCH]   (usable as --graph-path for subsequent runs)")

    if getattr(args, 'graph_only', False):
        print(f"\n{'=' * 60}")
        print(f"--graph-only: stopping after graph generation.")
        if concepts_json is not None:
            print(f"  Concepts: {concepts_path}")
        print(f"  Graph:    {graph_only_path}")
        print(f"{'=' * 60}")
        return

    # Initialize performance log
    perf_log = _init_perf_log(
        dataset_path=args.dataset,
        graph_path=args.graph_path,
        model_hint=args.model,
        total_questions=len(questions),
    )

    # Process each question individually and collect results
    print(f"\n--- Step 3: Encoding questions as executable constraints ---")
    print(f"[BATCH] Processing {len(questions)} questions individually...\n")

    MAX_SNIPPET_FIX = 3  # per-snippet fix attempts
    results: list[dict] = []  # {"question": str, "snippet": str|None, "success": bool}
    graph_repair_count = 0

    i = 0
    while i < len(questions):
        question = questions[i]
        print(f"[BATCH] [{i + 1}/{len(questions)}] {question}")
        t0 = time.time()
        try:
            _prompt, _raw, snippet, _model = do_generate_executable_single(question, graph_code, model=args.model)
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
            i += 1
            continue

        if not snippet:
            print(f"  ⚠️  Could not extract execute() call (time: {elapsed:.1f}s)")
            print(f"[BATCH]   raw response preview: {(_raw or '')[:200]}")
            _log_perf_entry(perf_log, {
                "type": "question",
                "index": i + 1,
                "question": question,
                "success": False,
                "error": "no execute() snippet extracted",
                "model": _model,
                "time_s": round(elapsed, 2),
                "raw_response_chars": len(_raw) if _raw else 0,
                "snippet_chars": 0,
                "snippet": None,
            })
            results.append({"question": question, "snippet": None, "success": False})
            i += 1
            continue

        # Validate the snippet by executing it against the graph
        snippet = snippet.strip()
        print(f"  ✅ snippet: {len(snippet)} chars, time: {elapsed:.1f}s")
        print(f"  >> {snippet}")

        valid, val_error = validate_single_constraint(graph_code, snippet)

        # --- Check for missing concepts (NameError) → graph repair ---
        # NameErrors cannot be fixed by patching the snippet — the graph
        # itself is missing the concept.  We must repair the graph first,
        # then retry this question from scratch.
        if not valid and _extract_missing_names(val_error):
            missing_names = _extract_missing_names(val_error)
            graph_repair_count += 1
            print(f"  ⚠️  Missing concepts in graph: {', '.join(missing_names)}")
            print(f"  [GRAPH-REPAIR #{graph_repair_count}] "
                  f"Rebuilding graph to add missing concepts...")

            new_graph_code, concepts_json = _rebuild_graph(
                questions, concepts_json, missing_names, question, args,
            )
            if new_graph_code is not None:
                graph_code = new_graph_code

                # Save updated intermediates
                concepts_path, graph_only_path = _derive_output_paths(args.output)
                if concepts_json is not None:
                    concepts_path.write_text(
                        json.dumps(concepts_json, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                graph_only_path.write_text(graph_code, encoding="utf-8")
                print(f"[GRAPH-REPAIR] Saved updated graph to: {graph_only_path}")

                # Invalidate all previous results — they were validated
                # against the old graph and must be re-validated
                if results:
                    prev_ok = sum(1 for r in results if r["success"])
                    print(f"[GRAPH-REPAIR] Re-validating {prev_ok} previously encoded constraints...")
                    new_results = []
                    for prev in results:
                        if prev["success"] and prev["snippet"]:
                            ok, _ = validate_single_constraint(graph_code, prev["snippet"])
                            if ok:
                                new_results.append(prev)
                                print(f"  ✅ Q: {prev['question'][:60]}...")
                            else:
                                new_results.append({
                                    "question": prev["question"],
                                    "snippet": None,
                                    "success": False,
                                })
                                print(f"  ❌ Q: {prev['question'][:60]}... (needs re-encode)")
                        else:
                            new_results.append(prev)
                    results = new_results

                # Retry the current question (don't increment i)
                print(f"[GRAPH-REPAIR] Retrying question {i + 1}...")
                continue
            else:
                print(f"  ❌ Graph repair failed — skipping question")
                results.append({"question": question, "snippet": None, "success": False})
                i += 1
                continue

        # --- Normal snippet fix loop (non-NameError failures) ---
        if not valid:
            print(f"  ❌ validation failed: {val_error}")
            for fix_attempt in range(1, MAX_SNIPPET_FIX + 1):
                combined_code = merge_executable_into_graph(graph_code, snippet)
                log, fixed_code = do_fix(combined_code, val_error, expected_execute_count=1, model=args.model)
                for entry in log:
                    if entry["type"] == "llm_interaction":
                        print(f"    [Fix {fix_attempt}/{MAX_SNIPPET_FIX}]")
                    else:
                        print(f"    ⚠️  {entry['message']}")

                if not fixed_code:
                    print(f"    ❌ LLM returned no code")
                    break

                fixed_snippet = _extract_snippet_from_fixed(fixed_code, graph_code)
                if not fixed_snippet:
                    print(f"    ❌ Could not extract snippet from fixed code")
                    break

                snippet = fixed_snippet
                valid, val_error = validate_single_constraint(graph_code, snippet)
                if valid:
                    print(f"    ✅ fixed on attempt {fix_attempt}")
                    print(f"    >> {snippet}")
                    break

                # If the fix introduced a NameError, escalate to graph repair
                new_missing = _extract_missing_names(val_error)
                if new_missing:
                    print(f"    ⚠️  Fix introduced missing concepts: {', '.join(new_missing)}")
                    # Break out of snippet fix loop — the outer while loop
                    # will re-validate and trigger graph repair on next iteration
                    break

                print(f"    ❌ still invalid: {val_error}")

            # After snippet fix loop, if still invalid due to NameError,
            # let the while loop retry (it will hit the graph repair branch)
            if not valid and _extract_missing_names(val_error):
                continue

        if valid:
            print(f"  ✅ validated")

        final_success = valid
        _log_perf_entry(perf_log, {
            "type": "question",
            "index": i + 1,
            "question": question,
            "success": final_success,
            "error": None if final_success else val_error[:500],
            "model": _model,
            "time_s": round(time.time() - t0, 2),
            "raw_response_chars": len(_raw) if _raw else 0,
            "snippet_chars": len(snippet) if snippet else 0,
            "snippet": snippet if final_success else None,
        })
        if final_success:
            results.append({"question": question, "snippet": snippet, "success": True})
        else:
            results.append({"question": question, "snippet": None, "success": False})
        i += 1

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

    # All constraints were individually validated in Step 3 — no Step 4 needed.

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

    # Model selection — available models come from config.yaml + discovered Ollama
    available_models = list(MODELS.keys())
    parser.add_argument(
        "--model", "-m",
        default=None,
        metavar="NAME",
        help=(
            "LLM model to use (default: first in config = '%(default_model)s'). "
            "Configured: %(models)s"
        ) % {"default_model": available_models[0] if available_models else "?",
             "models": ", ".join(available_models) if available_models else "none"},
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
        "-o", "--output", default=None,
        help="Output Python file path (default: output_graph_<model>_<date>.py)",
    )
    batch.add_argument(
        "--no-validate", action="store_true", dest="no_validate",
        help="Skip execution/validation of the generated output",
    )
    batch.add_argument(
        "--extract-batch-size", type=int, default=EXTRACT_BATCH_SIZE, dest="extract_batch_size",
        help=f"Number of questions per concept-extraction batch (default: {EXTRACT_BATCH_SIZE})",
    )
    batch.add_argument(
        "--graph-only", action="store_true", dest="graph_only",
        help="Stop after graph generation; skip executable constraint generation. "
             "Saves concepts JSON and graph code to separate files.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve and validate model selection
    if args.model is not None:
        from core import _build_model_catalog
        catalog = _build_model_catalog()
        if args.model not in catalog:
            print(f"ERROR: Unknown model '{args.model}'.")
            print(f"Available models: {', '.join(catalog.keys())}")
            sys.exit(1)
    selected = args.model or DEFAULT_MODEL
    print(f"[CONFIG] Config file: {CONFIG_PATH}")
    print(f"[CONFIG] Using model: {selected}" + (" (default)" if args.model is None else ""))

    # If user didn't override --output, embed date + model in the default filename
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_tag = re.sub(r'[\\/:*?"<>|]', "_", selected)
        args.output = f"output_graph_{model_tag}_{ts}.py"

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