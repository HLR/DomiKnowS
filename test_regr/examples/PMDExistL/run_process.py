import subprocess
import sys
import itertools
import copy


def run_all_process(contrains_two_existl=False, evaluate=False):
    idx = 1
    procs = []
    param_combinations = {
        "N": [2000],
        "lr": [1e-4, 1e-5],
        "epoch": [50]
    }
    keys, values = zip(*param_combinations.items())
    print(keys, values)
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for run in combinations:
        cmd = ["python", "main_rel.py"]
        for param, values in run.items():
            cmd.extend([f"--{param}", f"{values}"])
        if contrains_two_existl:
            cmd.append("--constraint_2_existL")
        if evaluate:
            cmd.append("--evaluate")

        print("Launching:", " ".join(cmd))

        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # procs.append((idx, p))
        # idx += 1

    # # ── wait for all of them and handle results ─────────────────────────────────
    # for i, p in procs:
    #     out, err = p.communicate()  # blocks until this process exits
    #     print(f"\n[# {i}] exit code {p.returncode}")
    #     if out:
    #         print(f"stdout:\n{out.strip()}")
    #     if err:
    #         print(f"stderr:\n{err.strip()}", file=sys.stderr)


if __name__ == '__main__':
    run_all_process()
    run_all_process(evaluate=True)
    run_all_process(contrains_two_existl=True)
    run_all_process(contrains_two_existl=True, evaluate=True)
