import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest

LOG_DIR = Path("test_logs")
LOG_DIR.mkdir(exist_ok=True)

BOOTSTRAP = (
    "import main\n"
    "args = main.parse_arguments()\n"
    "main.main(args)\n"
)

def run_test(params, gpu_id=None):
    args = []
    for key, value in params.items():
        if not str(value) == "False":
            args.extend([f'--{key}', str(value)])

    python_executable = sys.executable
    cmd = [python_executable, "-c", BOOTSTRAP] + args

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")

    if gpu_id is not None:
        if str(gpu_id).lower() == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            env=env,
        )
        return params, True, result.stdout
    except subprocess.CalledProcessError as e:
        full = (e.stdout or "") + ("\n" if e.stdout and e.stderr else "") + (e.stderr or "")
        return params, False, full


PMD_PARAMS = {
    'counting_tnorm': ["G", "P", "SP", "L"],
    'atLeastL': [True, False],
    'atMostL': [True, False],
    'epoch': [1000],
    'expected_atLeastL': [2],
    'expected_atMostL': [5],
    'expected_value': [0, 1],
    'N': [10],
    'M': [8],
    'model': ["PMD"],
}

SAMPLING_PARAMS = {
    'counting_tnorm': ["G"],
    'atLeastL': [True, False],
    'atMostL': [True, False],
    'epoch': [1000],
    'expected_atLeastL': [1, 2, 3],
    'expected_atMostL': [3, 4, 5],
    'expected_value': [0, 1],
    'N': [10],
    'M': [8],
    'model': ["sampling"],
    "sample_size": [10, 20, 50, 100, 200, -1],
}

GUMBEL_PMD_PARAMS = {
    'counting_tnorm': ["G", "P"],
    'atLeastL': [True, False],
    'atMostL': [True, False],
    'epoch': [800],
    'expected_atLeastL': [2],
    'expected_atMostL': [4],
    'expected_value': [0, 1],
    'N': [10],
    'M': [8],
    'model': ["gumbel_pmd"],
    'use_gumbel': [True],
    'initial_temp': [2.0, 3.0],
    'final_temp': [0.1, 0.2],
    'hard_gumbel': [False, True],
}

GUMBEL_SAMPLING_PARAMS = {
    'counting_tnorm': ["G"],
    'atLeastL': [True, False],
    'atMostL': [True, False],
    'epoch': [800],
    'expected_atLeastL': [2, 3],
    'expected_atMostL': [3, 4],
    'expected_value': [0, 1],
    'N': [10],
    'M': [8],
    'model': ["gumbel_sampling"],
    'sample_size': [50, 100, -1],
    'use_gumbel': [True],
    'initial_temp': [2.0],
    'final_temp': [0.1],
    'hard_gumbel': [False, True],
}


def generate_combinations(param_dict):
    keys, values = zip(*param_dict.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


@pytest.fixture
def gpu_id(request):
    return request.config.getoption("--gpu-id", default=None)


@pytest.mark.parametrize("params", generate_combinations(PMD_PARAMS))
def test_pmd_model(params, gpu_id):
    params_dict, success, output = run_test(params, gpu_id)
    passed = ("PASSED" in output) and success
    
    if not passed:
        counting_tnorm = params_dict.get("counting_tnorm", "NA")
        fname = LOG_DIR / f"fail_pmd_{counting_tnorm}_{id(params_dict)}.log"
        try:
            with open(fname, "w", encoding="utf-8", errors="replace") as f:
                f.write("Parameters:\n")
                json.dump(params_dict, f, indent=2)
                f.write("\n\n--- Full Output ---\n")
                f.write(output)
        except Exception:
            pass
    
    assert passed, f"Test failed for params: {params}\nOutput:\n{output[-2000:]}"


@pytest.mark.parametrize("params", generate_combinations(SAMPLING_PARAMS))
def test_sampling_model(params, gpu_id):
    params_dict, success, output = run_test(params, gpu_id)
    passed = ("PASSED" in output) and success
    
    if not passed:
        counting_tnorm = params_dict.get("counting_tnorm", "NA")
        fname = LOG_DIR / f"fail_sampling_{counting_tnorm}_{id(params_dict)}.log"
        try:
            with open(fname, "w", encoding="utf-8", errors="replace") as f:
                f.write("Parameters:\n")
                json.dump(params_dict, f, indent=2)
                f.write("\n\n--- Full Output ---\n")
                f.write(output)
        except Exception:
            pass
    
    assert passed, f"Test failed for params: {params}\nOutput:\n{output[-2000:]}"


@pytest.mark.parametrize("params", generate_combinations(GUMBEL_PMD_PARAMS))
def test_gumbel_pmd_model(params, gpu_id):
    params_dict, success, output = run_test(params, gpu_id)
    passed = ("PASSED" in output) and success
    
    if not passed:
        counting_tnorm = params_dict.get("counting_tnorm", "NA")
        gumbel_config = f"temp{params_dict.get('initial_temp')}to{params_dict.get('final_temp')}_hard{params_dict.get('hard_gumbel')}"
        fname = LOG_DIR / f"fail_gumbel_pmd_{counting_tnorm}_{gumbel_config}_{id(params_dict)}.log"
        try:
            with open(fname, "w", encoding="utf-8", errors="replace") as f:
                f.write("Parameters:\n")
                json.dump(params_dict, f, indent=2)
                f.write("\n\n--- Full Output ---\n")
                f.write(output)
        except Exception:
            pass
    
    assert passed, f"Test failed for params: {params}\nOutput:\n{output[-2000:]}"


@pytest.mark.parametrize("params", generate_combinations(GUMBEL_SAMPLING_PARAMS))
def test_gumbel_sampling_model(params, gpu_id):
    params_dict, success, output = run_test(params, gpu_id)
    passed = ("PASSED" in output) and success
    
    if not passed:
        counting_tnorm = params_dict.get("counting_tnorm", "NA")
        sample_size = params_dict.get("sample_size", "NA")
        gumbel_config = f"temp{params_dict.get('initial_temp')}to{params_dict.get('final_temp')}_hard{params_dict.get('hard_gumbel')}"
        fname = LOG_DIR / f"fail_gumbel_sampling_{counting_tnorm}_s{sample_size}_{gumbel_config}_{id(params_dict)}.log"
        try:
            with open(fname, "w", encoding="utf-8", errors="replace") as f:
                f.write("Parameters:\n")
                json.dump(params_dict, f, indent=2)
                f.write("\n\n--- Full Output ---\n")
                f.write(output)
        except Exception:
            pass
    
    assert passed, f"Test failed for params: {params}\nOutput:\n{output[-2000:]}"


def pytest_addoption(parser):
    parser.addoption("--gpu-id", action="store", default=None, 
                     help="GPU ID to use (e.g., '0', '1', or 'cpu')")