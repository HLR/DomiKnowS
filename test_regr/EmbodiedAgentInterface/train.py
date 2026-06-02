import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MAIN = SCRIPT_DIR / "main.py"
MODEL_DIR = SCRIPT_DIR / "models"


def parse_args():
    parser = argparse.ArgumentParser(description="Simple EAI train/eval driver for main.py.")
    parser.add_argument("--dataset", choices=["all", "behavior", "virtualhome"], default="all")
    parser.add_argument("--program", choices=["all", "solver", "primal-dual"], default="all")
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--train", action="store_true", help="Only train.")
    parser.add_argument("--evaluate", action="store_true", help="Only evaluate saved model.")
    parser.add_argument("--use-dfa", action="store_true", help="Only run DFA evaluation. Default evaluates both modes.")
    parser.add_argument("--constraint-warmup-iters", "--constraint-warmup-iter", dest="constraint_warmup_iters", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    return args, extra


def model_path(model_dir, dataset, program):
    suffix = "normal" if program == "solver" else "pmd"
    return Path(model_dir) / f"eai_{dataset}_{suffix}.pth"


def call_main(cmd, dry_run=False):
    print("$ " + " ".join(str(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main():
    args, extra = parse_args()
    datasets = [args.dataset]
    programs = ["solver", "primal-dual"] if args.program == "all" else [args.program]

    do_train = args.train or not args.evaluate
    do_eval = args.evaluate or not args.train

    for dataset in datasets:
        for program in programs:
            base = [
                sys.executable,
                str(MAIN),
                "--dataset",
                dataset,
                "--program",
                program,
                "--model",
                str(model_path(args.model_dir, dataset, program)),
                *extra,
            ]

            if args.constraint_warmup_iters is not None:
                base.extend(["--constraint-warmup-iters", str(args.constraint_warmup_iters)])

            if do_train:
                call_main([*base, "--train"], dry_run=args.dry_run)

            if do_eval:
                dfa_flags = [True] if args.use_dfa else [False, True]
                for use_dfa in dfa_flags:
                    cmd = [*base, "--evaluate"]
                    if use_dfa:
                        cmd.append("--use-dfa")
                    call_main(cmd, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
