# test_case.py
import pytest
from .main import run_test_case

test_inputs = list(enumerate([
    [],
    ["--firestations", "4", "--atleastaL", "3"],
    ["--firestations", "4", "5", "6", "7", "--atmostaL", "2"],
    ["--firestations", "1", "2", "6", "--constraint", "ifLnotLexistL"],
    ["--firestations", "1", "--constraint", "orLnotLexistL"],
    ["--firestations", "1", "4", "5", "6", "7", "--constraint", "orLnotLexistL"]
]))

@pytest.mark.parametrize("test_input", test_inputs)
def test_fire_station_case(test_input):
    """Test fire station configurations with different constraints"""
    run_test_case(test_input)

def test_case_0():
    """Test case 0: No fire stations"""
    run_test_case((0, []))

def test_case_1():
    """Test case 1: Fire station at city 4 with at least 3 constraint"""
    run_test_case((1, ["--firestations", "4", "--atleastaL", "3"]))

def test_case_2():
    """Test case 2: Fire stations at cities 4,5,6,7 with at most 2 constraint"""
    run_test_case((2, ["--firestations", "4", "5", "6", "7", "--atmostaL", "2"]))

def test_case_3():
    """Test case 3: Fire stations at cities 1,2,6 with ifLnotLexistL constraint"""
    run_test_case((3, ["--firestations", "1", "2", "6", "--constraint", "ifLnotLexistL"]))

def test_case_4():
    """Test case 4: Fire station at city 1 with orLnotLexistL constraint"""
    run_test_case((4, ["--firestations", "1", "--constraint", "orLnotLexistL"]))

def test_case_5():
    """Test case 5: Fire stations at cities 1,4,5,6,7 with orLnotLexistL constraint"""
    run_test_case((5, ["--firestations", "1", "4", "5", "6", "7", "--constraint", "orLnotLexistL"]))