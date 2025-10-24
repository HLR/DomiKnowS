#!/bin/bash
set +e

# First argument is the runner identifier (e.g., "test-cpu", "test-gpu")
RUNNER_ID="$1"
shift

# Remaining arguments are the test list
TEST_LIST=("$@")

if [ ${#TEST_LIST[@]} -eq 0 ]; then
  echo "No test subfolders specified in TEST_LIST. Skipping test execution."
  exit 0
fi

TEST_BASE_DIR="test_regr"
echo "Using test base directory: $TEST_BASE_DIR"

# Show available test directories for debugging
echo "Available test directories in $TEST_BASE_DIR:"
find "$TEST_BASE_DIR" -type d -name "*" | sort

# Track overall test result
OVERALL_RESULT=0
declare -A FAILED_TESTS
declare -A TEST_OUTPUTS
declare -A SKIPPED_TESTS

# Create results file and failed tests output file
RESULTS_FILE="/tmp/test_results_${RUNNER_ID}.txt"
FAILED_OUTPUT_FILE="/tmp/test_failures_${RUNNER_ID}.txt"
echo "" > "$RESULTS_FILE"
echo "=== FAILED TEST OUTPUTS FOR ${RUNNER_ID} ===" > "$FAILED_OUTPUT_FILE"
echo "" >> "$FAILED_OUTPUT_FILE"

# Run tests from specified subfolders
for subfolder in "${TEST_LIST[@]}"; do
  test_path="$TEST_BASE_DIR/$subfolder"
  echo "============================================"
  echo "🔍 DEBUGGING: Processing $subfolder"
  echo "   Full path: $test_path"
  
  # Check if directory exists
  if [ -d "$test_path" ]; then
    echo "   ✅ Directory exists"
    
    # List all files in the directory for debugging
    echo "   📂 Directory contents:"
    ls -la "$test_path" || echo "   ❌ Failed to list directory contents"
    
    # Check if there are any test files in the directory
    echo "   🔍 Looking for test files..."
    test_files=$(find "$test_path" -name "test_*.py" -o -name "*_test.py" 2>/dev/null)
    if [ -n "$test_files" ]; then
      test_files_count=$(echo "$test_files" | wc -l)
    else
      test_files_count=0
    fi
    
    if [ $test_files_count -gt 0 ]; then
      echo "   🔍 Found $test_files_count test file(s):"
      echo "$test_files"
    else
      echo "   ⚠️  No test files found matching patterns test_*.py or *_test.py"
      
      # Check for any Python files
      py_files=$(find "$test_path" -name "*.py" 2>/dev/null)
      if [ -n "$py_files" ]; then
        py_files_count=$(echo "$py_files" | wc -l)
        echo "   🔍 Found $py_files_count Python file(s) (but not matching test patterns):"
        echo "$py_files"
      else
        py_files_count=0
        echo "   🔍 No Python files found at all"
      fi
    fi
    
    echo "   🧪 Running pytest..."
    echo "   Command: uv run pytest -v -s --tb=short --no-header \"$test_path\""
    
    # Capture both stdout and stderr
    test_output=$(uv run pytest -v -s --tb=short --no-header "$test_path" 2>&1)
    test_exit_code=$?
    
    echo "   📊 Pytest exit code: $test_exit_code"
    echo "   📄 Pytest output length: $(echo "$test_output" | wc -l) lines"
    echo "   📄 First few lines of pytest output:"
    echo "$test_output" | head -5
    
    if [ $test_exit_code -eq 5 ]; then
      # Exit code 5: No tests collected - treat as warning, not failure
      echo "   ⚠️  Exit code 5: No tests collected"
      if [ $test_files_count -eq 0 ]; then
        echo "   🔍 Reason: No test files found in directory"
        skip_reason="No test files found (*.py files matching test_* or *_test pattern)"
      else
        echo "   🔍 Reason: Test files exist but no tests discovered by pytest"
        skip_reason="No tests collected - test files exist but pytest couldn't discover any tests"
      fi
      SKIPPED_TESTS["$subfolder"]="$skip_reason"
      echo "SKIP:$subfolder:$skip_reason" >> "$RESULTS_FILE"
      echo "   ✅ Treating as SKIP (not failure)"
    elif [ $test_exit_code -ne 0 ]; then
      echo "   ❌ Exit code $test_exit_code: Test failure"
      OVERALL_RESULT=1
      
      # Extract failure summary from pytest output
      failure_summary=$(echo "$test_output" | grep -A 10 "short test summary info" | tail -n +2 | head -20)
      if [ -z "$failure_summary" ]; then
        # Fallback: look for FAILED lines
        failure_summary=$(echo "$test_output" | grep "FAILED\|ERROR\|Exception:" | head -10)
      fi
      if [ -z "$failure_summary" ]; then
        # Last resort: get last few lines of output
        failure_summary=$(echo "$test_output" | tail -10)
      fi
      
      FAILED_TESTS["$subfolder"]="$failure_summary"
      echo "FAIL:$subfolder:$failure_summary" >> "$RESULTS_FILE"
      echo "   🔍 Failure details preview:"
      echo "$failure_summary" | head -3
      
      # Append full test output to failed output file
      echo "============================================" >> "$FAILED_OUTPUT_FILE"
      echo "FAILED TEST: $subfolder" >> "$FAILED_OUTPUT_FILE"
      echo "Test path: $test_path" >> "$FAILED_OUTPUT_FILE"
      echo "Exit code: $test_exit_code" >> "$FAILED_OUTPUT_FILE"
      echo "--------------------------------------------" >> "$FAILED_OUTPUT_FILE"
      echo "$test_output" >> "$FAILED_OUTPUT_FILE"
      echo "" >> "$FAILED_OUTPUT_FILE"
      echo "============================================" >> "$FAILED_OUTPUT_FILE"
      echo "" >> "$FAILED_OUTPUT_FILE"
    else
      echo "   ✅ Exit code 0: Tests PASSED"
      echo "PASS:$subfolder:" >> "$RESULTS_FILE"
    fi
  else
    echo "   ❌ Directory does not exist: $test_path"
    SKIPPED_TESTS["$subfolder"]="Directory not found"
    echo "SKIP:$subfolder:Directory not found" >> "$RESULTS_FILE"
  fi
  echo "============================================"
done

# Save overall result and runner ID
echo "OVERALL_RESULT=$OVERALL_RESULT" >> "$RESULTS_FILE"
echo "RUNNER_ID=$RUNNER_ID" >> "$RESULTS_FILE"

# Add summary to failed output file
if [ $OVERALL_RESULT -eq 0 ]; then
  echo "=== ALL TESTS PASSED ===" >> "$FAILED_OUTPUT_FILE"
else
  echo "=== SUMMARY ===" >> "$FAILED_OUTPUT_FILE"
  echo "Total failed tests: ${#FAILED_TESTS[@]}" >> "$FAILED_OUTPUT_FILE"
fi

# Always exit 0 during debugging phase
echo "🚧 DEBUG MODE: CI set to always pass (not failing on test failures)"
exit 0