#!/bin/bash
set +e

# Directory containing downloaded artifacts
RESULTS_DIR="$1"
# Debug mode parameter: "true" or "false" (default: "true")
DEBUG_MODE="${2:-true}"

if [ ! -d "$RESULTS_DIR" ]; then
  echo "❌ Results directory not found: $RESULTS_DIR"
  exit 0
fi

# ── Aggregate results from all runners ──

# Directory-level counts (pass/fail/skip per test subfolder)
total_passed=0
total_failed=0
total_skipped=0
overall_failure=0

# Pytest item-level counts (individual test functions)
total_items_passed=0
total_items_failed=0
total_items_skipped=0
total_items_errors=0

declare -A ALL_PASSED_TESTS
declare -A ALL_FAILED_TESTS
declare -A ALL_SKIPPED_TESTS
declare -A RUNNER_STATS
declare -A RUNNER_ITEM_STATS
declare -A TEST_ITEM_COUNTS
declare -A TEST_FAILED_ITEM_NAMES
declare -A TEST_SKIPPED_ITEM_NAMES

# Process each result file
for result_file in "$RESULTS_DIR"/*/test_results_*.txt; do
  if [ ! -f "$result_file" ]; then
    continue
  fi

  # First pass: extract runner_id (written at end of file)
  runner_id=$(grep -oE 'RUNNER_ID=.*' "$result_file" | head -1 | cut -d'=' -f2)
  runner_id="${runner_id:-unknown}"

  runner_passed=0
  runner_failed=0
  runner_skipped=0
  runner_items_passed=0
  runner_items_failed=0
  runner_items_skipped=0
  runner_items_errors=0

  # Second pass: process all entries with runner_id already known
  while IFS=':' read -r status subfolder f3 f4 f5 f6; do
    case "$status" in
      "PASS")
        ((total_passed++))
        ((runner_passed++))
        ALL_PASSED_TESTS["$subfolder"]="$runner_id"
        ;;
      "FAIL")
        ((total_failed++))
        ((runner_failed++))
        ALL_FAILED_TESTS["$subfolder"]="$f3|$runner_id"
        ;;
      "SKIP")
        ((total_skipped++))
        ((runner_skipped++))
        ALL_SKIPPED_TESTS["$subfolder"]="$f3|$runner_id"
        ;;
      "ITEMS")
        # Format: ITEMS:<subfolder>:<passed>:<failed>:<skipped>:<errors>
        ip=${f3:-0}; ifa=${f4:-0}; is=${f5:-0}; ie=${f6:-0}
        ((total_items_passed += ip))
        ((total_items_failed += ifa))
        ((total_items_skipped += is))
        ((total_items_errors += ie))
        ((runner_items_passed += ip))
        ((runner_items_failed += ifa))
        ((runner_items_skipped += is))
        ((runner_items_errors += ie))
        TEST_ITEM_COUNTS["$subfolder"]="${ip}p ${ifa}f ${is}s ${ie}e"
        ;;
      "OVERALL_RESULT"*)
        result=$(echo "$status" | cut -d'=' -f2)
        if [ "$result" != "0" ]; then
          overall_failure=1
        fi
        ;;
    esac
  done < "$result_file"

  RUNNER_STATS["$runner_id"]="✅ $runner_passed | ❌ $runner_failed | ⚠️  $runner_skipped"
  RUNNER_ITEM_STATS["$runner_id"]="✅ $runner_items_passed | ❌ $runner_items_failed | ⏭️  $runner_items_skipped | 💥 $runner_items_errors"

  # Item-names pass: read FAILED_ITEMS/SKIPPED_ITEMS lines without IFS=: splitting
  # so that :: separators inside test node IDs are preserved.
  while IFS= read -r _raw_line; do
    if [[ "$_raw_line" == FAILED_ITEMS:* ]]; then
      _sf=$(echo "$_raw_line" | cut -d: -f2)
      _names=$(echo "$_raw_line" | cut -d: -f3-)
      [ -n "$_names" ] && TEST_FAILED_ITEM_NAMES["$_sf"]="$_names"
    elif [[ "$_raw_line" == SKIPPED_ITEMS:* ]]; then
      _sf=$(echo "$_raw_line" | cut -d: -f2)
      _names=$(echo "$_raw_line" | cut -d: -f3-)
      [ -n "$_names" ] && TEST_SKIPPED_ITEM_NAMES["$_sf"]="$_names"
    fi
  done < "$result_file"
done

total_tests=$((total_passed + total_failed + total_skipped))
total_items=$((total_items_passed + total_items_failed + total_items_skipped + total_items_errors))

# ── Helper: write to both stdout and GITHUB_STEP_SUMMARY ──
# If GITHUB_STEP_SUMMARY is not set (local run), write to stdout only.
SUMMARY_FILE="${GITHUB_STEP_SUMMARY:-/dev/null}"

report() {
  echo "$1"
  echo "$1" >> "$SUMMARY_FILE"
}

# ── Generate report ──

echo "================================="
echo "📊 COMBINED TEST RESULTS SUMMARY"
echo "================================="
echo ""

report "## 📊 Test Results Summary"
report ""

# ── Overall numbers ──

if [ $total_failed -eq 0 ] && [ $total_items_failed -eq 0 ] && [ $total_items_errors -eq 0 ]; then
  report "### ✅ All tests passed"
else
  report "### ❌ Failures detected"
fi
report ""

# Summary table
report "| Metric | Passed | Failed | Skipped | Errors | Total |"
report "|--------|-------:|-------:|--------:|-------:|------:|"
report "| **Test directories** | $total_passed | $total_failed | $total_skipped | — | $total_tests |"
report "| **Pytest items** | $total_items_passed | $total_items_failed | $total_items_skipped | $total_items_errors | $total_items |"
report ""

# ── Per-runner breakdown ──

if [ ${#RUNNER_STATS[@]} -gt 0 ]; then
  report "<details>"
  report "<summary>🖥️ Breakdown by runner</summary>"
  report ""
  report "| Runner | Dirs passed | Dirs failed | Dirs skipped | Items passed | Items failed | Items skipped | Items errors |"
  report "|--------|------------:|------------:|-------------:|-------------:|-------------:|--------------:|-------------:|"
  for runner in "${!RUNNER_STATS[@]}"; do
    # Parse runner stats to get numbers
    dir_stats="${RUNNER_STATS[$runner]}"
    item_stats="${RUNNER_ITEM_STATS[$runner]}"
    # Extract numbers from formatted strings
    rp=$(echo "$dir_stats" | grep -oE '[0-9]+' | sed -n '1p')
    rf=$(echo "$dir_stats" | grep -oE '[0-9]+' | sed -n '2p')
    rs=$(echo "$dir_stats" | grep -oE '[0-9]+' | sed -n '3p')
    rip=$(echo "$item_stats" | grep -oE '[0-9]+' | sed -n '1p')
    rif=$(echo "$item_stats" | grep -oE '[0-9]+' | sed -n '2p')
    ris=$(echo "$item_stats" | grep -oE '[0-9]+' | sed -n '3p')
    rie=$(echo "$item_stats" | grep -oE '[0-9]+' | sed -n '4p')
    report "| $runner | ${rp:-0} | ${rf:-0} | ${rs:-0} | ${rip:-0} | ${rif:-0} | ${ris:-0} | ${rie:-0} |"
  done
  report ""
  report "</details>"
  report ""
fi

# ── Failed tests ──

if [ ${#ALL_FAILED_TESTS[@]} -gt 0 ]; then
  report "### ❌ Failed test directories"
  report ""
  report "| Directory | Runner | Passed | Failed | Skipped | Errors | Failure reason |"
  report "|-----------|--------|-------:|-------:|--------:|-------:|----------------|"
  for subfolder in "${!ALL_FAILED_TESTS[@]}"; do
    info="${ALL_FAILED_TESTS[$subfolder]}"
    failure_reason=$(echo "$info" | cut -d'|' -f1)
    runner=$(echo "$info" | cut -d'|' -f2)
    items="${TEST_ITEM_COUNTS[$subfolder]}"

    # Parse item counts
    ip=0; ifa=0; is=0; ie=0
    if [ -n "$items" ]; then
      ip=$(echo "$items" | grep -oE '[0-9]+p' | grep -oE '[0-9]+')
      ifa=$(echo "$items" | grep -oE '[0-9]+f' | grep -oE '[0-9]+')
      is=$(echo "$items" | grep -oE '[0-9]+s' | grep -oE '[0-9]+')
      ie=$(echo "$items" | grep -oE '[0-9]+e' | grep -oE '[0-9]+')
    fi

    # Truncate failure reason for table cell
    short_reason=""
    if [[ "$failure_reason" == *"Exception:"* ]]; then
      short_reason=$(echo "$failure_reason" | grep "Exception:" | head -1 | cut -c1-80)
    elif [[ "$failure_reason" == *"FAILED"* ]]; then
      short_reason=$(echo "$failure_reason" | grep "FAILED" | head -1 | cut -c1-80)
    else
      short_reason=$(echo "$failure_reason" | head -1 | cut -c1-80)
    fi
    # Escape pipe characters for Markdown table
    short_reason=$(echo "$short_reason" | sed 's/|/\\|/g')

    report "| \`$subfolder\` | $runner | ${ip:-0} | ${ifa:-0} | ${is:-0} | ${ie:-0} | ${short_reason} |"
  done
  report ""
fi

# ── Failed pytest items per directory ──

_has_failed_items=0
for _sf in "${!TEST_FAILED_ITEM_NAMES[@]}"; do
  [ -n "${TEST_FAILED_ITEM_NAMES[$_sf]}" ] && _has_failed_items=1 && break
done

if [ $_has_failed_items -eq 1 ]; then
  _total_fi=0
  for _sf in "${!TEST_FAILED_ITEM_NAMES[@]}"; do
    _n=$(echo "${TEST_FAILED_ITEM_NAMES[$_sf]}" | tr '|' '\n' | grep -c .)
    ((_total_fi += _n)) || true
  done
  report "<details>"
  report "<summary>❌ Failed pytest items per directory ($_total_fi total)</summary>"
  report ""
  for _sf in "${!TEST_FAILED_ITEM_NAMES[@]}"; do
    _items="${TEST_FAILED_ITEM_NAMES[$_sf]}"
    if [ -n "$_items" ]; then
      report "**\`$_sf\`**:"
      while IFS= read -r _item; do
        [ -n "$_item" ] && report "- \`$_item\`"
      done < <(echo "$_items" | tr '|' '\n')
      report ""
    fi
  done
  report "</details>"
  report ""
fi

# ── Skipped/suspended pytest items per directory ──

_has_skipped_items=0
for _sf in "${!TEST_SKIPPED_ITEM_NAMES[@]}"; do
  [ -n "${TEST_SKIPPED_ITEM_NAMES[$_sf]}" ] && _has_skipped_items=1 && break
done

if [ $_has_skipped_items -eq 1 ]; then
  _total_si=0
  for _sf in "${!TEST_SKIPPED_ITEM_NAMES[@]}"; do
    _n=$(echo "${TEST_SKIPPED_ITEM_NAMES[$_sf]}" | tr '|' '\n' | grep -c .)
    ((_total_si += _n)) || true
  done
  report "<details>"
  report "<summary>⏭️ Skipped pytest items per directory ($_total_si total)</summary>"
  report ""
  for _sf in "${!TEST_SKIPPED_ITEM_NAMES[@]}"; do
    _items="${TEST_SKIPPED_ITEM_NAMES[$_sf]}"
    if [ -n "$_items" ]; then
      report "**\`$_sf\`**:"
      while IFS= read -r _item; do
        [ -n "$_item" ] && report "- \`$_item\`"
      done < <(echo "$_items" | tr '|' '\n')
      report ""
    fi
  done
  report "</details>"
  report ""
fi

# ── Passed tests ──

if [ ${#ALL_PASSED_TESTS[@]} -gt 0 ]; then
  report "<details>"
  report "<summary>✅ Passed test directories (${#ALL_PASSED_TESTS[@]})</summary>"
  report ""
  report "| Directory | Runner | Passed | Failed | Skipped | Errors |"
  report "|-----------|--------|-------:|-------:|--------:|-------:|"
  for subfolder in "${!ALL_PASSED_TESTS[@]}"; do
    runner="${ALL_PASSED_TESTS[$subfolder]}"
    items="${TEST_ITEM_COUNTS[$subfolder]}"
    ip=0; ifa=0; is=0; ie=0
    if [ -n "$items" ]; then
      ip=$(echo "$items" | grep -oE '[0-9]+p' | grep -oE '[0-9]+')
      ifa=$(echo "$items" | grep -oE '[0-9]+f' | grep -oE '[0-9]+')
      is=$(echo "$items" | grep -oE '[0-9]+s' | grep -oE '[0-9]+')
      ie=$(echo "$items" | grep -oE '[0-9]+e' | grep -oE '[0-9]+')
    fi
    report "| \`$subfolder\` | $runner | ${ip:-0} | ${ifa:-0} | ${is:-0} | ${ie:-0} |"
  done
  report ""
  report "</details>"
  report ""
fi

# ── Skipped tests ──

if [ ${#ALL_SKIPPED_TESTS[@]} -gt 0 ]; then
  report "<details>"
  report "<summary>⚠️ Skipped test directories (${#ALL_SKIPPED_TESTS[@]})</summary>"
  report ""
  report "| Directory | Runner | Reason |"
  report "|-----------|--------|--------|"
  for subfolder in "${!ALL_SKIPPED_TESTS[@]}"; do
    info="${ALL_SKIPPED_TESTS[$subfolder]}"
    reason=$(echo "$info" | cut -d'|' -f1)
    runner=$(echo "$info" | cut -d'|' -f2)
    reason_escaped=$(echo "$reason" | sed 's/|/\\|/g')
    report "| \`$subfolder\` | $runner | $reason_escaped |"
  done
  report ""
  report "</details>"
  report ""
fi

# ── Debug mode notice ──

if [ "$DEBUG_MODE" = "true" ]; then
  report "> [!NOTE]"
  report "> 🚧 **Debug mode** — CI configured to not fail on test failures."
  report ""
fi

# ── Exit code ──

if [ $total_failed -eq 0 ]; then
  echo ""
  echo "✅ Report completed successfully"
  exit 0
else
  echo ""
  echo "💥 TEST FAILURES DETECTED"

  if [ "$DEBUG_MODE" = "true" ]; then
    echo "🚧 DEBUG MODE: CI configured to not fail on test failures"
    echo "✅ Report completed (debug mode - always passing)"
    exit 0
  else
    echo "🚨 PRODUCTION MODE: CI will fail due to test failures"
    echo "❌ Report completed with failures"
    exit 1
  fi
fi
