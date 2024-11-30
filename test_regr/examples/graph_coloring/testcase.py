# run_tests.py
from main import run_test_case
from concurrent.futures import ProcessPoolExecutor

test_inputs = list(enumerate([
    [],
    ["--firestations", "4", "--atleastaL", "3"],
    ["--firestations", "4", "5", "6", "7", "--atmostaL", "2"],
    ["--firestations", "1", "2", "6", "--constraint", "ifLnotLexistL"],
    ["--firestations", "1", "--constraint", "orLnotLexistL"],
    ["--firestations", "1", "4", "5", "6", "7", "--constraint", "orLnotLexistL"]
]))

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=6) as executor:
        executor.map(run_test_case, test_inputs)
