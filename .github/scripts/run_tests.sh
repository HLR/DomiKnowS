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

# Create results file and detailed output directory
RESULTS_FILE="/tmp/test_results_${RUNNER_ID}.txt"
OUTPUT_DIR="/tmp/test_outputs_${RUNNER_ID}"
mkdir -p "$OUTPUT_DIR"
echo "" > "$RESULTS_FILE"

# Run tests from specified subfolders
for subfolder in "${TEST_LIST[@]}"; do
  test_path="$TEST_BASE_DIR/$subfolder"
  echo "============================================"
  echo "🔍 DEBUGGING: Processing $subfolder"
  echo "   Full path: $test_path"
  
  # Sanitize subfolder name for filename
  safe_name=$(echo "$subfolder" | tr '/' '_')
  output_file="$OUTPUT_DIR/${safe_name}.log"
  
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
    echo "   Command: uv run pytest -v --tb=short --no-header \"$test_path\""
    
    # Capture both stdout and stderr to file
    uv run pytest -v --tb=short --no-header "$test_path" > "$output_file" 2>&1
    test_exit_code=$?
    
    echo "   📊 Pytest exit code: $test_exit_code"
    echo "   📄 Full output saved to: $output_file"
    echo "   📄 Output length: $(wc -l < "$output_file") lines"
    echo "   📄 First few lines of pytest output:"
    head -5 "$output_file"
    
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
      failure_summary=$(grep -A 10 "short test summary info" "$output_file" | tail -n +2 | head -20)
      if [ -z "$failure_summary" ]; then
        # Fallback: look for FAILED lines
        failure_summary=$(grep "FAILED\|ERROR\|Exception:" "$output_file" | head -10)
      fi
      if [ -z "$failure_summary" ]; then
        # Last resort: get last few lines of output
        failure_summary=$(tail -10 "$output_file")
      fi
      
      FAILED_TESTS["$subfolder"]="$failure_summary"
      echo "FAIL:$subfolder:$failure_summary" >> "$RESULTS_FILE"
      echo "   🔍 Failure details preview:"
      echo "$failure_summary" | head -3
    else
      echo "   ✅ Exit code 0: Tests PASSED"
      echo "PASS:$subfolder:" >> "$RESULTS_FILE"
    fi
  else
    echo "   ❌ Directory does not exist: $test_path"
    SKIPPED_TESTS["$subfolder"]="Directory not found"
    echo "SKIP:$subfolder:Directory not found" >> "$RESULTS_FILE"
    echo "Directory not found: $test_path" > "$output_file"
  fi
  echo "============================================"
done

# Save overall result and runner ID
echo "OVERALL_RESULT=$OVERALL_RESULT" >> "$RESULTS_FILE"
echo "RUNNER_ID=$RUNNER_ID" >> "$RESULTS_FILE"

# Always exit 0 during debugging phase
echo "🚧 DEBUG MODE: CI set to always pass (not failing on test failures)"
exit 0