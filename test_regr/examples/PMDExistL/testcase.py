import os.path
import subprocess
import re


def calculate_time():
    with open("logs/regrTimer.log", "r") as f:
        lines = f.read()
    result = re.findall(r"End of Loss Calculation - total internl time: (\d)ms", lines)
    return sum(int(t) for t in result), len(result)


def test_PMD(N = 10):
    prev_time = 0
    prev_count = 0
    if os.path.exists("logs/regrTimer.log"):
        prev_time, prev_count = calculate_time()

    output_count = N
    passed = []
    times = []
    print(f"Test 1: ExactL(1, 3) on {output_count} output")
    result = subprocess.run(["python", "./main.py",
                             "--expected_value", "1",
                             "--M", f"{output_count}"],
                            stdout=subprocess.PIPE, text=True)
    print(result.stdout)

    cur_time, cur_count = calculate_time()
    print("Avg time for calculate constraint loss: {:.4f}ms".format((cur_time - prev_time) / (cur_count - prev_count)))
    passed.append("P" if "PASS" in result.stdout else "X")
    times.append((cur_time - prev_time) / (cur_count - prev_count))
    prev_time, prev_count = cur_time, cur_count
    print("#" * 30)

    print(f"Test 2: AtLeastL(1, 4) on {output_count} output")
    result = subprocess.run(["python", "./main.py",
                             "--expected_value", "1",
                             "--atLeastL", "T",
                             "--expected_atLeastL", "4",
                             "--M", f"{output_count}"],
                            stdout=subprocess.PIPE, text=True)
    print(result.stdout)
    cur_time, cur_count = calculate_time()
    print("Avg time for calculate constraint loss: {:.4f}ms".format((cur_time - prev_time) / (cur_count - prev_count)))
    passed.append("P" if "PASS" in result.stdout else "X")
    times.append((cur_time - prev_time) / (cur_count - prev_count))
    prev_time, prev_count = cur_time, cur_count
    print("#" * 30)

    print(f"Test 3: AtLeastL(1, 10) on {output_count} output")
    result = subprocess.run(["python", "./main.py",
                             "--expected_value", "1",
                             "--atLeastL", "T",
                             "--expected_atLeastL", "10",
                             "--M", f"{output_count}"],
                            stdout=subprocess.PIPE, text=True)
    print(result.stdout)
    cur_time, cur_count = calculate_time()
    print("Avg time for calculate constraint loss: {:.4f}ms".format((cur_time - prev_time) / (cur_count - prev_count)))
    passed.append("P" if "PASS" in result.stdout else "X")
    times.append((cur_time - prev_time) / (cur_count - prev_count))
    prev_time, prev_count = cur_time, cur_count

    print("#" * 30)

    print(f"Test 4: AtMostL(0, 5) on {output_count} output")
    result = subprocess.run(["python", "./main.py",
                             "--expected_value", "0",
                             "--atMostL", "T",
                             "--expected_atMostL", "4",
                             "--M", f"{output_count}"],
                            stdout=subprocess.PIPE, text=True)
    print(result.stdout)
    cur_time, cur_count = calculate_time()
    print("Avg time for calculate constraint loss: {:.4f}ms".format((cur_time - prev_time) / (cur_count - prev_count)))
    passed.append("P" if "PASS" in result.stdout else "X")
    times.append((cur_time - prev_time) / (cur_count - prev_count))
    prev_time, prev_count = cur_time, cur_count

    print("#" * 30)

    print(f"Test 5: AtMostL(0, 0) on {output_count} output")
    result = subprocess.run(["python", "./main.py",
                             "--expected_value", "0",
                             "--atMostL", "T",
                             "--expected_atMostL", "1",
                             "--M", f"{output_count}"],
                            stdout=subprocess.PIPE, text=True)
    print(result.stdout)
    cur_time, cur_count = calculate_time()
    print("Avg time for calculate constraint loss: {:.4f}ms".format((cur_time - prev_time) / (cur_count - prev_count)))
    passed.append("P" if "PASS" in result.stdout else "X")
    times.append((cur_time - prev_time) / (cur_count - prev_count))
    prev_time, prev_count = cur_time, cur_count

    print("#" * 30)

    print(f"Test 6: AtLeastL(1, 4) AtMostL(1, 6) on {output_count} output")
    result = subprocess.run(["python", "./main.py",
                             "--expected_value", "1",
                             "--atLeastL", "T",
                             "--expected_atLeastL", "4",
                             "--atMostL", "T",
                             "--expected_atMostL", "6",
                             "--M", f"{output_count}"],
                            stdout=subprocess.PIPE, text=True)
    print(result.stdout)
    cur_time, cur_count = calculate_time()
    print("Avg time for calculate constraint loss: {:.4f}ms".format((cur_time - prev_time) / (cur_count - prev_count)))
    passed.append("P" if "PASS" in result.stdout else "X")
    times.append((cur_time - prev_time) / (cur_count - prev_count))
    prev_time, prev_count = cur_time, cur_count

    print("#" * 30)

    print(f"Test 7: AtLeastL(1, 5) AtMostL(1, 5) on {output_count} output")
    result = subprocess.run(["python", "./main.py",
                             "--expected_value", "1",
                             "--atLeastL", "T",
                             "--expected_atLeastL", "5",
                             "--atMostL", "T",
                             "--expected_atMostL", "5",
                             "--M", f"{output_count}"],
                            stdout=subprocess.PIPE, text=True)
    print(result.stdout)
    cur_time, cur_count = calculate_time()
    print("Avg time for calculate constraint loss: {:.4f}ms".format((cur_time - prev_time) / (cur_count - prev_count)))
    passed.append("P" if "PASS" in result.stdout else "X")
    times.append((cur_time - prev_time) / (cur_count - prev_count))
    prev_time, prev_count = cur_time, cur_count

    print("#" * 30)

    print(f"Test 8: AtLeastL(0, 2) AtMostL(0, 6) on {output_count} output")
    result = subprocess.run(["python", "./main.py",
                             "--expected_value", "0",
                             "--atLeastL", "T",
                             "--expected_atLeastL", "2",
                             "--atMostL", "T",
                             "--expected_atMostL", "6",
                             "--M", f"{output_count}"],
                            stdout=subprocess.PIPE, text=True)
    print(result.stdout)
    cur_time, cur_count = calculate_time()
    print("Avg time for calculate constraint loss: {:.4f}ms".format((cur_time - prev_time) / (cur_count - prev_count)))
    passed.append("P" if "PASS" in result.stdout else "X")
    times.append((cur_time - prev_time) / (cur_count - prev_count))
    prev_time, prev_count = cur_time, cur_count

    print("#" * 30)

    print(f"Test 9: AtLeastL(0, 5) AtMostL(0, 5) on {output_count} output")
    result = subprocess.run(["python", "./main.py",
                             "--expected_value", "0",
                             "--atLeastL", "T",
                             "--expected_atLeastL", "5",
                             "--atMostL", "T",
                             "--expected_atMostL", "5",
                             "--M", f"{output_count}"],
                            stdout=subprocess.PIPE, text=True)
    print(result.stdout)
    cur_time, cur_count = calculate_time()
    print("Avg time for calculate constraint loss: {:.4f}ms".format((cur_time - prev_time) / (cur_count - prev_count)))
    passed.append("P" if "PASS" in result.stdout else "X")
    times.append((cur_time - prev_time) / (cur_count - prev_count))
    prev_time, prev_count = cur_time, cur_count
    print("#" * 30)

    return "".join(passed), sum(times) / len(times)


if __name__ == "__main__":
    pass_test = []
    time_test = []

    test, time = test_PMD(10)
    pass_test.append(test)
    time_test.append(time)

    test, time = test_PMD(20)
    pass_test.append(test)
    time_test.append(time)

    test, time = test_PMD(50)
    pass_test.append(test)
    time_test.append(time)

    test, time = test_PMD(100)
    pass_test.append(test)
    time_test.append(time)

    print("Pass:", pass_test)
    print("Time:", time_test)
