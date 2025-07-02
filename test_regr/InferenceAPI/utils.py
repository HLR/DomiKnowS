import random

def generate_test_case(lenght=5):
    is_positive = random.choice([True, False])
    values1 = [random.random() * (1 if is_positive else -1) for _ in range(lenght)]
    y1 = 1 if all(v > 0 for v in values1) else 0

    is_positive = random.choice([True, False])
    values2 = [random.random() * (1 if is_positive else -1) for _ in range(lenght)]
    y2 = 0 if all(v > 0 for v in values2) else 1

    # Compute z based on y1 and y2
    z = (0 if y1 == 0 else 2) + y2

    return {
        "value1": [values1],
        "value2": [values2],
        "y1": [[y1]],
        "y2": [[y2]],
        "z": [z]
    }