import pytest
import subprocess
import sys
import os


# Marker to distinguish between subprocess and direct tests
USE_SUBPROCESS = os.environ.get('USE_SUBPROCESS', 'false').lower() == 'true'


def run_test(args_list):
    """
    Run test either via subprocess (CI/CD) or direct call (local debugging)
    """
    if USE_SUBPROCESS:
        command = ['python', 'main.py'] + args_list
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        assert result.returncode == 0, result.stderr[result.stderr.find("Traceback"):] if "Traceback" in result.stderr else result.stderr
        return result.returncode
    else:
        # Direct call for debugging
        sys.argv = ['main.py'] + args_list
        from main import main, parse_arguments
        args = parse_arguments()
        result = main(args)
        assert result == 0
        return result


def test_zero_counting_godel():
    run_test(["--epochs", "1", "--train_portion", "zero_counting_YN"])


def test_zero_counting_lukas():
    run_test(["--epochs", "1", "--train_portion", "zero_counting_YN", "--counting_tnorm", "L"])


def test_zero_counting_product():
    run_test(["--epochs", "1", "--train_portion", "zero_counting_YN", "--counting_tnorm", "P"])


def test_zero_counting_simple_product():
    run_test(["--epochs", "1", "--train_portion", "zero_counting_YN", "--counting_tnorm", "SP"])


def test_over_counting_godel():
    run_test(["--epochs", "1", "--train_portion", "over_counting_YN", "--counting_tnorm", "G"])


def test_over_counting_lukas():
    run_test(["--epochs", "1", "--train_portion", "over_counting_YN", "--counting_tnorm", "L"])


def test_over_counting_product():
    run_test(["--epochs", "1", "--train_portion", "over_counting_YN", "--counting_tnorm", "P"])


def test_over_counting_simple_produce():
    run_test(["--epochs", "1", "--train_portion", "over_counting_YN", "--counting_tnorm", "SP"])


def test_general_run():
    run_test(["--epochs", "5", "--checked_acc", "0.8"])