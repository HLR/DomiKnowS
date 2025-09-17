import sys
import os
import pytest
import subprocess
import csp_demo

def run_csp(*args):
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
    assert r.returncode == 0, f"Process failed with code {r.returncode}. stderr: {r.stderr}"
    assert "ERROR" not in r.stdout

