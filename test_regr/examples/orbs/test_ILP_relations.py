import sys
import os
import pytest
import subprocess
import csp_demo

def run_csp(*args):
    # Run the demo script by its file path to avoid module resolution issues
    script_path = csp_demo.__file__
    return subprocess.run(
        [sys.executable, script_path, *args],
        text=True,
        capture_output=True
    )


@pytest.mark.parametrize("args", [
    ["--constraint", "exactL"],
    ["--constraint", "foreach_bag_existsL"],
])
def test_csp_cases(args):
    r = run_csp(*args)
    # Show both stdout and stderr for easier debugging on failure
    print("stdout:", r.stdout)
    print("stderr:", r.stderr)
    # Ensure the script executed successfully
    assert r.returncode == 0, f"Process failed with code {r.returncode}. stderr: {r.stderr}"
    # Ensure no error markers were printed by the script
    assert "ERROR" not in r.stdout

