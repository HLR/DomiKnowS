import subprocess
import sys
import os
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from pathlib import Path

LOG_DIR = Path("test_logs")
LOG_DIR.mkdir(exist_ok=True)

# Bootstrap: enable debugging + memory profiling, then run main inside a memory meter
BOOTSTRAP = (
    "try:\n"
    "    import debug_utils as DU\n"
    "    DU.enable_full_debug()\n"
    "    DU.enable_memory_profiling()\n"
    "    from debug_utils import memory_meter\n"
    "    use_debug = True\n"
    "except ImportError:\n"
    "    use_debug = False\n"
    "import main\n"
    "args = main.parse_arguments()\n"
    "if use_debug:\n"
    "    with memory_meter('subprocess-main'):\n"
    "        main.main(args)\n"
    "else:\n"
    "    main.main(args)\n"
)

def run_test(params):
    # Convert params to command-line arguments
    args = []
    for key, value in params.items():
        if not str(value) == "False":
            args.extend([f'--{key}', str(value)])

    python_executable = sys.executable

    # Run main through our bootstrap so debug_utils is guaranteed to be active
    cmd = [python_executable, "-c", BOOTSTRAP] + args

    # Ensure unbuffered output so we see everything in order
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Helpful when CUDA is available (also set inside enable_full_debug; harmless here)
    env.setdefault("CUDA_LAUNCH_BLOCKING", "1")

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
        # Return both stdout and stderr so we don't lose context
        full = (e.stdout or "") + ("\n" if e.stdout and e.stderr else "") + (e.stderr or "")
        return params, False, full


def run_tests(param_combinations):
    keys, values = zip(*param_combinations.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = defaultdict(list)
    total_combinations = len(combinations)

    with ProcessPoolExecutor(max_workers=4) as executor:
        for i, (params, ok, output) in enumerate(executor.map(run_test, combinations), 1):
            counting_tnorm = params["counting_tnorm"]
            passed = ("PASSED" in output) and ok
            results[counting_tnorm].append((params, passed, output))

            print(f"\nCompleted test {i}/{total_combinations}:")
            print(f"Parameters: {params}")
            print(output)  # print FULL output/traceback
            print(f"Passed: {passed}")

            if not passed:
                # Save the full output for post-mortem
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
                    print(output[-2000:])  # quick glance


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
    }
    # run_tests(PMD_combinations)

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
        "sample_size": [10, 20, 50, 100, 200, -1]
    }

    run_tests(sampling_combinations)