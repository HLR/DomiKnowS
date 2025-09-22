import argparse
import itertools
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

LOG_DIR = Path("test_logs")
LOG_DIR.mkdir(exist_ok=True)

# Bootstrap: enable debugging + memory profiling, then run main inside a memory meter
BOOTSTRAP = (
    "import debug.debug_utils as DU\n"
    "DU.enable_full_debug()\n"
    "DU.enable_memory_profiling()\n"
    "from debug.debug_utils import memory_meter\n"
    "import main\n"
    "args = main.parse_arguments()\n"
    "with memory_meter('subprocess-main'):\n"
    "    main.main(args)\n"
)

def run_test(params, gpu_id=None):
    # Convert params to command-line arguments
    args = []
    for key, value in params.items():
        if not str(value) == "False":
            args.extend([f'--{key}', str(value)])

    python_executable = sys.executable
    cmd = [python_executable, "-c", BOOTSTRAP] + args

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


def run_tests(param_combinations, gpus=None, max_workers=None):
    keys, values = zip(*param_combinations.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    total_combinations = len(combinations)

    # Normalize GPUs argument
    gpus = [g.strip() for g in (gpus or []) if g.strip() != ""]
    use_cpu_only = (len(gpus) == 1 and gpus[0].lower() == "cpu")
    if use_cpu_only:
        gpus = ["cpu"]

    # Choose worker count: one per GPU if GPUs are provided; else fallback
    if max_workers is None:
        if gpus:
            max_workers = min(len(gpus), total_combinations)
        else:
            max_workers = min(4, os.cpu_count() or 4)

    results = defaultdict(list)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, params in enumerate(combinations):
            gpu = None
            if gpus:
                gpu = gpus[i % len(gpus)]
            futures.append(executor.submit(run_test, params, gpu))

        for i, fut in enumerate(as_completed(futures), 1):
            params, ok, output = fut.result()
            counting_tnorm = params.get("counting_tnorm", "NA")
            passed = ("PASSED" in output) and ok
            results[counting_tnorm].append((params, passed, output))

            print(f"\nCompleted test {i}/{total_combinations}:")
            print(f"Parameters: {params}")
            print(output)  # full output/traceback
            print(f"Passed: {passed}")

            if not passed:
                fname = LOG_DIR / f"fail_{i:03d}_{counting_tnorm}.log"
                try:
                    with open(fname, "w", encoding="utf-8", errors="replace") as f:
                        f.write("Parameters:\n")
                        json.dump(params, f, indent=2)
                        f.write("\n\n--- Full Output ---\n")
                        f.write(output)
                    print(f"[saved] {fname}")
                except Exception as log_err:
                    print(f"[warn] could not write log file: {log_err}")

    # Summary
    print("\n--- Test Summary ---")
    for counting_tnorm, tnorm_results in results.items():
        passed_tests = sum(1 for _, passed, _ in tnorm_results if passed)
        print(f"\nResults for counting_tnorm = {counting_tnorm}:")
        print(f"Passed {passed_tests} out of {len(tnorm_results)} tests.")

        if passed_tests < len(tnorm_results):
            print(f"Failed tests for counting_tnorm = {counting_tnorm}:")
            for params, passed, output in tnorm_results:
                if not passed:
                    print(f"Parameters: {params}")
                    print("Error output (last ~2KB):")
                    print(output[-2000:])


def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default="", help="Comma-separated GPU ids (e.g., '6,5,2'). Use 'cpu' for CPU-only.")
    p.add_argument("--max-workers", type=int, default=None, help="Override ProcessPoolExecutor workers.")
    return p.parse_args()


if __name__ == "__main__":
    # Define the parameter combinations to test PMD
    PMD_combinations = {
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
        # 'device': ['cuda']  # optional: or rely on main.py --device auto
    }
    # run_tests(PMD_combinations, gpus=[...])

    # Define the parameter combinations to test sampling model
    sampling_combinations = {
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
        # 'device': ['cuda']  # optional
    }

    args = parse_cli()
    gpus = []
    if args.gpus.strip():
        gpus = [s.strip() for s in args.gpus.split(",")]

    run_tests(sampling_combinations, gpus=gpus, max_workers=args.max_workers)
