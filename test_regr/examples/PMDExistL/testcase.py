import subprocess
import sys
import os
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict


def run_test(params):
    # Convert params to command-line arguments
    args = []
    for key, value in params.items():
        if not str(value) == "False":
            args.extend([f'--{key}', str(value)])

    # Get the path to the Python interpreter in the current virtual environment
    python_executable = sys.executable

    # Construct the command to run the main script
    cmd = [python_executable, 'main.py'] + args

    # Run the command in a subprocess
    try:
        # Use UTF-8 encoding and replace any characters that can't be decoded
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', check=True)
        return params, True, result.stdout
    except subprocess.CalledProcessError as e:
        return params, False, e.stderr


def run_tests(param_combinations):
    """Run tests with different combinations of input arguments."""
    # Generate all combinations of parameters
    keys, values = zip(*param_combinations.items())
    print(keys, values)
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Run tests for each combination using ProcessPoolExecutor with max_workers=4
    results = defaultdict(list)
    total_combinations = len(combinations)

    with ProcessPoolExecutor(max_workers=4) as executor:
        for i, (params, test_passed, output) in enumerate(executor.map(run_test, combinations), 1):
            counting_tnorm = params['counting_tnorm']
            results[counting_tnorm].append((params, 'PASSED' in output, output))
            print(f"\nCompleted test {i}/{total_combinations}:")
            print(f"Parameters: {params}")
            print(output.split("\n")[-2])
            print(f"Passed: {'PASSED' in output}")

    # Print summary of results per counting_tnorm
    print("\n--- Test Summary ---")
    for counting_tnorm, tnorm_results in results.items():
        passed_tests = sum(1 for _, passed, _ in tnorm_results if passed)
        print(f"\nResults for counting_tnorm = {counting_tnorm}:")
        print(f"Passed {passed_tests} out of {len(tnorm_results)} tests.")

        # Print details of failed tests for this counting_tnorm
        if passed_tests < len(tnorm_results):
            print(f"Failed tests for counting_tnorm = {counting_tnorm}:")
            for params, passed, output in tnorm_results:
                if not passed:
                    print(f"Parameters: {params}")
                    print(f"Error output: {output}")


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
        "sample_size": [10, 20, 50, 100, 200, -1]  # maximum 2^8 = 256
    }

    run_tests(sampling_combinations)
