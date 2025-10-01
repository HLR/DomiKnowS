import sys
import pytest
import itertools
import subprocess
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_test(params, gpu_id=None):
    # Convert params to command-line arguments
    args = []
    for key, value in params.items():
        if not str(value) == "False":
            args.extend([f'--{key}', str(value)])

    python_executable = sys.executable
    cmd = [python_executable, "main.py"] + args
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    # Ensure current folder is importable (for debug_utils, main, etc.)
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")

    # Pin this subprocess to a specific GPU (or CPU)
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


def run_tests(param_combinations, gpus=None, max_workers=None, raise_error=False):
    keys, values = zip(*param_combinations.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    total_combinations = len(combinations)

    # Normalize GPUs argument
    gpus = [g.strip() for g in (gpus or []) if g.strip() != ""]
    use_cpu_only = (len(gpus) == 1 and gpus[0].lower() == "cpu")
    if use_cpu_only:
        gpus = ["cpu"]

    # Choose worker count: one per GPU if GPUs are provided; else fallback
    # CI-friendly: limit workers when in CI environment
    is_ci = os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS')
    if max_workers is None:
        if is_ci:
            max_workers = 1  # Single worker for CI stability
        elif gpus:
            max_workers = min(len(gpus), total_combinations)
        else:
            max_workers = min(4, os.cpu_count() or 4)
    
    # Auto-reduce epochs in CI environment
    if is_ci and 'epoch' in combinations[0]:
        original_epochs = combinations[0]['epoch']
        reduced_epochs = 50  # Reduced for CI
        print(f"🔧 CI detected: Reducing epochs from {original_epochs} to {reduced_epochs}")
        for combo in combinations:
            if 'epoch' in combo:
                combo['epoch'] = reduced_epochs
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, params in enumerate(combinations):
            gpu = None
            if gpus:
                gpu = gpus[i % len(gpus)]
            futures.append(executor.submit(run_test, params, gpu))
        for i, fut in enumerate(as_completed(futures), 1):
            params, ok, output = fut.result()
            assert (("PASSED" in output) and ok)
            # assert passed == True


def test_PMD_exactL_Godel(capsys, monkeypatch):
    # , "P", "SP", "L"
    PMD_combinations = {
        'counting_tnorm': ["G"],
        'atLeastL': [False],
        'atMostL': [False],
        'epoch': [1000],
        'expected_atLeastL': [2, 4, 5],
        'expected_atMostL': [8],
        'expected_value': [0, 1],
        'N': [10],
        'M': [8],
        'model': ["PMD"],
        # 'device': ['cuda']  # optional: or rely on main.py --device auto
    }
    run_tests(PMD_combinations, gpus=None, max_workers=None)
    out = capsys.readouterr().out
    assert not "ERROR" in out

def test_PMD_exactL_SimpProduce(capsys, monkeypatch):
    # , "P", "SP", "L"
    PMD_combinations = {
        'counting_tnorm': ["SP"],
        'atLeastL': [False],
        'atMostL': [False],
        'epoch': [1000],
        'expected_atLeastL': [2, 4, 5],
        'expected_atMostL': [8],
        'expected_value': [0, 1],
        'N': [10],
        'M': [8],
        'model': ["PMD"],
        # 'device': ['cuda']  # optional: or rely on main.py --device auto
    }
    # monkeypatch.setattr(sys, "argv", ["csp_demo","--constraint","exactL"])
    run_tests(PMD_combinations, gpus=None, max_workers=None)
    out = capsys.readouterr().out
    assert not "ERROR" in out


def test_PMD_exactL_Product(capsys, monkeypatch):
    # , "P", "SP", "L"
    PMD_combinations = {
        'counting_tnorm': ["P"],
        'atLeastL': [False],
        'atMostL': [False],
        'epoch': [1000],
        'expected_atLeastL': [2, 4, 5],
        'expected_atMostL': [8],
        'expected_value': [0, 1],
        'N': [10],
        'M': [8],
        'model': ["PMD"],
        # 'device': ['cuda']  # optional: or rely on main.py --device auto
    }
    run_tests(PMD_combinations, gpus=None, max_workers=None)
    out = capsys.readouterr().out
    assert not "ERROR" in out

def test_PMD_exactL_Lukas(capsys, monkeypatch):
    # , "P", "SP", "L"
    PMD_combinations = {
        'counting_tnorm': ["L"],
        'atLeastL': [False],
        'atMostL': [False],
        'epoch': [1000],
        'expected_atLeastL': [2, 4, 5],
        'expected_atMostL': [8],
        'expected_value': [0, 1],
        'N': [10],
        'M': [8],
        'model': ["PMD"],
        # 'device': ['cuda']  # optional: or rely on main.py --device auto
    }
    run_tests(PMD_combinations, gpus=None, max_workers=None)
    out = capsys.readouterr().out
    assert not "ERROR" in out

def test_PMD_atLeastL_Godel(capsys, monkeypatch):
    # , "P", "SP", "L"
    PMD_combinations = {
        'counting_tnorm': ["G"],
        'atLeastL': [True],
        'atMostL': [False],
        'epoch': [1000],
        'expected_atLeastL': [2, 4, 5],
        'expected_atMostL': [6],
        'expected_value': [0, 1],
        'N': [10],
        'M': [8],
        'model': ["PMD"],
        # 'device': ['cuda']  # optional: or rely on main.py --device auto
    }
    run_tests(PMD_combinations, gpus=None, max_workers=None)
    out = capsys.readouterr().out
    assert not "ERROR" in out

def test_PMD_atLeastL_SimpProduce(capsys, monkeypatch):
    # , "P", "SP", "L"
    PMD_combinations = {
        'counting_tnorm': ["SP"],
        'atLeastL': [True],
        'atMostL': [False],
        'epoch': [1000],
        'expected_atLeastL': [2, 4, 5],
        'expected_atMostL': [8],
        'expected_value': [0, 1],
        'N': [10],
        'M': [8],
        'model': ["PMD"],
        # 'device': ['cuda']  # optional: or rely on main.py --device auto
    }
    # monkeypatch.setattr(sys, "argv", ["csp_demo","--constraint","exactL"])
    run_tests(PMD_combinations, gpus=None, max_workers=None)
    out = capsys.readouterr().out
    assert not "ERROR" in out

def test_PMD_atLeastL_Product(capsys, monkeypatch):
    # , "P", "SP", "L"
    PMD_combinations = {
        'counting_tnorm': ["P"],
        'atLeastL': [True],
        'atMostL': [False],
        'epoch': [1000],
        'expected_atLeastL': [2, 4, 5],
        'expected_atMostL': [8],
        'expected_value': [0, 1],
        'N': [10],
        'M': [8],
        'model': ["PMD"],
        # 'device': ['cuda']  # optional: or rely on main.py --device auto
    }
    run_tests(PMD_combinations, gpus=None, max_workers=None)
    out = capsys.readouterr().out
    assert not "ERROR" in out


def test_PMD_atLeastL_Lukas(capsys, monkeypatch):
    # , "P", "SP", "L"
    PMD_combinations = {
        'counting_tnorm': ["L"],
        'atLeastL': [True],
        'atMostL': [False],
        'epoch': [1000],
        'expected_atLeastL': [2, 4, 5],
        'expected_atMostL': [8],
        'expected_value': [0, 1],
        'N': [10],
        'M': [8],
        'model': ["PMD"],
        # 'device': ['cuda']  # optional: or rely on main.py --device auto
    }
    run_tests(PMD_combinations, gpus=None, max_workers=None)
    out = capsys.readouterr().out
    assert not "ERROR" in out


def test_PMD_atMostL_Godel(capsys, monkeypatch):
    # , "P", "SP", "L"
    PMD_combinations = {
        'counting_tnorm': ["G"],
        'atLeastL': [False],
        'atMostL': [True],
        'epoch': [1000],
        'expected_atLeastL': [0],
        'expected_atMostL': [2, 4, 5],
        'expected_value': [0, 1],
        'N': [10],
        'M': [8],
        'model': ["PMD"],
        # 'device': ['cuda']  # optional: or rely on main.py --device auto
    }
    run_tests(PMD_combinations, gpus=None, max_workers=None)
    out = capsys.readouterr().out
    assert not "ERROR" in out
def test_PMD_atMostL_SimpProduce(capsys, monkeypatch):
    # , "P", "SP", "L"
    PMD_combinations = {
        'counting_tnorm': ["SP"],
        'atLeastL': [False],
        'atMostL': [True],
        'epoch': [1000],
        'expected_atLeastL': [0],
        'expected_atMostL': [2, 4, 5],
        'expected_value': [0, 1],
        'N': [10],
        'M': [8],
        'model': ["PMD"],
        # 'device': ['cuda']  # optional: or rely on main.py --device auto
    }
    # monkeypatch.setattr(sys, "argv", ["csp_demo","--constraint","exactL"])
    run_tests(PMD_combinations, gpus=None, max_workers=None)
    out = capsys.readouterr().out
    assert not "ERROR" in out
def test_PMD_atMostL_Product(capsys, monkeypatch):
    # , "P", "SP", "L"
    PMD_combinations = {
        'counting_tnorm': ["P"],
        'atLeastL': [False],
        'atMostL': [True],
        'epoch': [1000],
        'expected_atLeastL': [0],
        'expected_atMostL': [2, 4, 5],
        'expected_value': [0, 1],
        'N': [10],
        'M': [8],
        'model': ["PMD"],
        # 'device': ['cuda']  # optional: or rely on main.py --device auto
    }
    run_tests(PMD_combinations, gpus=None, max_workers=None)
    out = capsys.readouterr().out
    assert not "ERROR" in out
def test_PMD_atMostL_Lukas(capsys, monkeypatch):
    # , "P", "SP", "L"
    PMD_combinations = {
        'counting_tnorm': ["L"],
        'atLeastL': [False],
        'atMostL': [True],
        'epoch': [1000],
        'expected_atLeastL': [0],
        'expected_atMostL': [2, 4, 5],
        'expected_value': [0, 1],
        'N': [10],
        'M': [8],
        'model': ["PMD"],
        # 'device': ['cuda']  # optional: or rely on main.py --device auto
    }
    run_tests(PMD_combinations, gpus=None, max_workers=None)
    out = capsys.readouterr().out
    assert not "ERROR" in out
