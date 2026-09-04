#!/usr/bin/env bash
# validate.sh — Validate a generated DomiKnowS graph file using the framework
#
# Usage: ./scripts/validate.sh <python_file>
#
# Requires: domiknows installed (pip install domiknows)
#
# Checks:
#   1. File exists and is non-empty
#   2. Python syntax is valid (AST parse)
#   3. Graph context block and concept definitions present
#   4. No constraint operators leaked into graph-only section
#   5. Full execution with DomiKnowS imports
#   6. DomiKnowS framework validation (checkLcCorrectness):
#      - All concepts in constraints exist in the graph
#      - queryL has multiclass first argument
#      - Counting constraints have valid syntax/cardinality
#      - Relations in constraints have proper has_a structure
#      - Variable definitions and usage are consistent
#   7. Graph introspection: concepts, relations, constraints summary

set -euo pipefail

FILE="${1:?Usage: validate.sh <python_file>}"

if [[ ! -f "$FILE" ]]; then
    echo "❌ FAIL: File not found: $FILE"
    exit 1
fi

if [[ ! -s "$FILE" ]]; then
    echo "❌ FAIL: File is empty: $FILE"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "  DomiKnowS Graph Validator"
echo "  File: $FILE"
echo "═══════════════════════════════════════════════════════════"
echo ""

ERRORS=0
WARNINGS=0

# ---------------------------------------------------------------------------
# Check 1: Python syntax validation (AST parse — no imports needed)
# ---------------------------------------------------------------------------
echo "── Check 1: Python Syntax ──"
if python3 -c "
import ast, sys
try:
    ast.parse(open('$FILE').read())
except SyntaxError as e:
    print(f'  Line {e.lineno}: {e.msg}')
    sys.exit(1)
" 2>&1; then
    echo "  ✅ Syntax valid"
else
    echo "  ❌ Syntax errors found"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# ---------------------------------------------------------------------------
# Check 2: Structural checks (regex — no imports needed)
# ---------------------------------------------------------------------------
echo "── Check 2: Structure ──"

if ! grep -qE "with\s+Graph\s*\(" "$FILE"; then
    echo "  ❌ No 'with Graph(...)' context block found"
    ERRORS=$((ERRORS + 1))
else
    echo "  ✅ Graph context block found"
fi

CONCEPT_COUNT=$(grep -cE "(Concept|EnumConcept)\s*\(" "$FILE" || true)
if [[ "$CONCEPT_COUNT" -eq 0 ]]; then
    echo "  ❌ No Concept/EnumConcept definitions found"
    ERRORS=$((ERRORS + 1))
else
    echo "  ✅ $CONCEPT_COUNT concept definition(s) found"
fi

# Check for constraints leaking into graph definition section
# (between 'with Graph' and 'with graph:' if both exist)
CONSTRAINT_OPS="existsL\|atLeastL\|atMostL\|exactL\|andL\|orL\|notL\|ifL\|nandL\|iotaL\|queryL\|greaterL\|lessL\|equalCountsL"
HAS_EXEC_SECTION=$(grep -c '^\s*with\s\+graph\s*:' "$FILE" || true)
if [[ "$HAS_EXEC_SECTION" -gt 0 ]]; then
    # Extract graph-only section (before 'with graph:')
    GRAPH_SECTION=$(sed -n '1,/^\s*with\s\+graph\s*:/p' "$FILE" | head -n -1)
    BAD_LINES=$(echo "$GRAPH_SECTION" | grep -n "$CONSTRAINT_OPS" || true)
    if [[ -n "$BAD_LINES" ]]; then
        echo "  ⚠️  Constraint operators found in graph definition section:"
        echo "$BAD_LINES" | sed 's/^/      /'
        WARNINGS=$((WARNINGS + 1))
    else
        echo "  ✅ No constraint operators in graph definition section"
    fi
fi

EXEC_COUNT=$(grep -cE '\bexecute\s*\(' "$FILE" || true)
echo "  ℹ️  $EXEC_COUNT execute() call(s) found"
echo ""

# ---------------------------------------------------------------------------
# Check 3: Execute with DomiKnowS and run framework validation
# ---------------------------------------------------------------------------
echo "── Check 3: DomiKnowS Execution & Validation ──"

VALIDATE_SCRIPT=$(cat <<'PYEOF'
import sys
import traceback

# --- Standard imports  ---
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

filepath = sys.argv[1]

# --- Step A: Execute the graph file ---
print("  [A] Executing graph code...")
exec_globals = {}
try:
    with open(filepath, 'r') as f:
        code = f.read()
    exec(code, exec_globals)
    print("  ✅ Execution successful")
except Exception as e:
    print(f"  ❌ Execution failed: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- Step B: Find the graph object ---
graph_obj = None
for name, obj in exec_globals.items():
    if isinstance(obj, Graph):
        graph_obj = obj
        break

if graph_obj is None:
    print("  ❌ No Graph object found in executed globals")
    sys.exit(1)

print(f"  ✅ Graph '{graph_obj.name}' found")

# --- Step C: Introspect graph structure ---
print("")
print("  [B] Graph Structure:")

# Concepts
all_concepts = graph_obj.getAllConceptNames()
print(f"      Concepts ({len(all_concepts)}): {', '.join(all_concepts[:20])}", end="")
if len(all_concepts) > 20:
    print(f" ... and {len(all_concepts) - 20} more")
else:
    print()

# Relations
relations = dict(graph_obj.relations)
if relations:
    rel_names = [r for r in relations.keys() if not r.endswith('reversed') and 'not_a' not in r]
    print(f"      Relations ({len(rel_names)}): {', '.join(rel_names[:15])}", end="")
    if len(rel_names) > 15:
        print(f" ... and {len(rel_names) - 15} more")
    else:
        print()
else:
    print("      Relations: none")

# Logical constraints
lc_count = len(graph_obj.logicalConstrains)
elc_count = len(graph_obj.executableLCs)
print(f"      Logical constraints: {lc_count}")
print(f"      Executable constraints: {elc_count}")

if elc_count > 0:
    print("      Executable constraint details:")
    for elc_name, elc in graph_obj.executableLCs.items():
        try:
            inner = elc.innerLC if hasattr(elc, 'innerLC') else elc
            desc = f"{type(inner).__name__}{inner.strEs()}"
        except Exception:
            desc = type(elc).__name__
        print(f"        {elc_name}: {desc}")

# Labels
labels = graph_obj.executableLCsLabels
if labels:
    print(f"      Executable constraint labels: {labels}")

# Predicates
try:
    predicates = graph_obj.print_predicates()
    if predicates:
        print(f"      Predicates ({len(predicates)}):")
        for p in predicates[:20]:
            print(f"        {p}")
        if len(predicates) > 20:
            print(f"        ... and {len(predicates) - 20} more")
except Exception as e:
    print(f"      ⚠️  Could not generate predicates: {e}")

# --- Step D: Run DomiKnowS framework validation (checkLcCorrectness) ---
print("")
print("  [C] DomiKnowS Framework Validation (checkLcCorrectness):")

if lc_count == 0 and elc_count == 0:
    print("      ℹ️  No constraints to validate (graph-only mode)")
else:
    try:
        from domiknows.graph.lcUtils import checkLcCorrectness
        checkLcCorrectness(graph_obj)
        print("      ✅ All constraint validations passed:")
        print("         - All concepts in constraints exist in the graph")
        print("         - queryL constraints have proper multiclass concepts")
        print("         - Counting constraints have valid syntax and cardinality")
        print("         - Relations in constraints have proper has_a structure")
        print("         - Variable definitions and usage are consistent")
    except Exception as e:
        print(f"      ❌ Validation failed: {e}")
        traceback.print_exc()
        sys.exit(2)

# --- Step E: Test compile_executable if qa_data exists ---
if 'qa_data' in exec_globals:
    print("")
    print("  [D] compile_executable Validation:")
    qa_data = exec_globals['qa_data']
    print(f"      Found qa_data with {len(qa_data)} entries")
    try:
        logic_dataset = graph_obj.compile_executable(
            qa_data,
            logic_keyword='constraint',
            logic_label_keyword='label'
        )
        print(f"      ✅ compile_executable succeeded")
        print(f"         Compiled {len(graph_obj.executableLCs)} executable constraint(s)")
    except Exception as e:
        print(f"      ❌ compile_executable failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(3)

print("")
print("  ✅ All DomiKnowS validations passed")
PYEOF
)

if python3 -c "$VALIDATE_SCRIPT" "$FILE" 2>&1; then
    :
else
    EXIT_CODE=$?
    if [[ "$EXIT_CODE" -eq 1 ]]; then
        echo "  ❌ Graph execution failed"
    elif [[ "$EXIT_CODE" -eq 2 ]]; then
        echo "  ❌ Constraint validation failed (checkLcCorrectness)"
    elif [[ "$EXIT_CODE" -eq 3 ]]; then
        echo "  ❌ compile_executable failed"
    fi
    ERRORS=$((ERRORS + 1))
fi
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "═══════════════════════════════════════════════════════════"
if [[ "$ERRORS" -eq 0 ]]; then
    if [[ "$WARNINGS" -gt 0 ]]; then
        echo "  ✅ Passed with $WARNINGS warning(s)"
    else
        echo "  ✅ All checks passed"
    fi
    exit 0
else
    echo "  ❌ $ERRORS check(s) failed, $WARNINGS warning(s)"
    exit 1
fi