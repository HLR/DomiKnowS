import pytest
import subprocess


def test_zero_counting_godel():
    command = ['python', 'main.py', "--epochs", "1", "--train_size", "10", "--train_portion", "zero_counting_YN"]
    result = subprocess.run(command, capture_output=True, text=True)
    # Checking no assert error in the main command (Accuracy error / General error)
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, result.stderr[result.stderr.find("Traceback"):]

def test_zero_counting_lukas():
    command = ['python', 'main.py', "--epochs", "1", "--train_size", "10", "--train_portion", "zero_counting_YN", "--counting_tnorm", "L"]
    result = subprocess.run(command, capture_output=True, text=True)
    # Checking no assert error in the main command (Accuracy error / General error)
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, result.stderr[result.stderr.find("Traceback"):]

def test_zero_counting_product():
    command = ['python', 'main.py', "--epochs", "1", "--train_size", "10", "--train_portion", "zero_counting_YN", "--counting_tnorm", "P"]
    result = subprocess.run(command, capture_output=True, text=True)
    # Checking no assert error in the main command (Accuracy error / General error)
    assert result.returncode == 0, result.stderr[result.stderr.find("Traceback"):]


def test_zero_counting_simple_product():
    command = ['python', 'main.py', "--epochs", "1", "--train_size", "10", "--train_portion", "zero_counting_YN", "--counting_tnorm", "SP"]
    result = subprocess.run(command, capture_output=True, text=True)
    # Checking no assert error in the main command (Accuracy error / General error)
    assert result.returncode == 0, result.stderr[result.stderr.find("Traceback"):]



def test_over_counting_godel():
    command = ['python', 'main.py', "--epochs", "1", "--train_size", "10", "--train_portion", "over_counting_YN", "--counting_tnorm", "G"]
    result = subprocess.run(command, capture_output=True, text=True)
    # Checking no assert error in the main command (Accuracy error / General error)
    assert result.returncode == 0, result.stderr[result.stderr.find("Traceback"):]

def test_over_counting_lukas():
    command = ['python', 'main.py', "--epochs", "1", "--train_size", "10", "--train_portion", "over_counting_YN", "--counting_tnorm", "L"]
    result = subprocess.run(command, capture_output=True, text=True)
    # Checking no assert error in the main command (Accuracy error / General error)
    assert result.returncode == 0, result.stderr[result.stderr.find("Traceback"):]

def test_over_counting_product():
    command = ['python', 'main.py', "--epochs", "1", "--train_size", "10", "--train_portion", "over_counting_YN", "--counting_tnorm", "P"]
    result = subprocess.run(command, capture_output=True, text=True)
    # Checking no assert error in the main command (Accuracy error / General error)
    assert result.returncode == 0, result.stderr[result.stderr.find("Traceback"):]

def test_over_counting_simple_produce():
    command = ['python', 'main.py', "--epochs", "1", "--train_size", "10", "--train_portion", "over_counting_YN", "--counting_tnorm", "SP"]
    result = subprocess.run(command, capture_output=True, text=True)
    # Checking no assert error in the main command (Accuracy error / General error)
    assert result.returncode == 0, result.stderr[result.stderr.find("Traceback"):]


def test_general_run():
    command = ['python', 'main.py', "--epochs", "5", "--checked_acc", "0.8"]
    result = subprocess.run(command, capture_output=True, text=True)
    # Checking no assert error in the main command (Accuracy error / General error)
    assert result.returncode == 0, result.stderr[result.stderr.find("Traceback"):]
