import argparse
import os
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
    parser.add_argument("--cuda-visible-devices", default="4", help="CUDA_VISIBLE_DEVICES value used for main.py subprocesses.")
    parser.add_argument("--small-llm", action="store_true", help="Use the trainable causal-LM baseline.")
    parser.add_argument("--llm-backbone-path", default="Qwen/Qwen2.5-1.5B-Instruct", help="Small causal LM backbone for --small-llm.")
    parser.add_argument("--use-lora", action="store_true", help="Train LoRA adapters for --small-llm.")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=None)
    parser.add_argument("--llm-device-map", default=None, help="Optional Hugging Face device_map for causal LM loading, e.g. auto.")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable causal-LM gradient checkpointing.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model-tag", default=None, help="Optional extra tag appended to generated checkpoint names.")
    parser.add_argument("--train", action="store_true", help="Only train.")
    parser.add_argument("--evaluate", action="store_true", help="Only evaluate saved model.")
    parser.add_argument("--use-dfa", action="store_true", help="Only run DFA evaluation. Default evaluates both modes.")
    parser.add_argument("--constraint-warmup-iters", "--constraint-warmup-iter", dest="constraint_warmup_iters", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    return args, extra


def _safe_name(value):
    text = str(value).replace("/", "-").replace(".", "p")
    keep = []
    for char in text:
        keep.append(char if char.isalnum() or char in {"-", "_"} else "-")
    return "".join(keep).strip("-")


def _float_name(value):
    return _safe_name(f"{float(value):g}")


def model_path(model_dir, dataset, program, args):
    suffix = "normal" if program == "solver" else "pmd"
    model_name = "causal-lm" if args.small_llm else "default"
    parts = [
        f"eai_{dataset}_{suffix}",
        _safe_name(model_name),
        f"lr{_float_name(args.lr)}",
        f"ep{args.epochs}",
        f"ms{args.max_steps}",
        f"hd{args.hidden_dim}",
        f"bs{args.batch_size}",
    ]
    if args.small_llm:
        parts.append(_safe_name(Path(args.llm_backbone_path).name))
    if args.use_lora:
        targets = "-".join(args.lora_target_modules or ["default"] )
        parts.extend([
            f"lora-r{args.lora_r}",
            f"a{args.lora_alpha}",
            f"d{_float_name(args.lora_dropout)}",
            _safe_name(targets),
        ])
    if args.gradient_checkpointing:
        parts.append("gc")
    if args.model_tag:
        parts.append(_safe_name(args.model_tag))
    return Path(model_dir) / ("_".join(parts) + ".pth")


def call_main(cmd, dry_run=False, env=None):
    env_prefix = ""
    if env and env.get("CUDA_VISIBLE_DEVICES"):
        env_prefix = f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
    print("$ " + env_prefix + " ".join(str(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, env=env)


def main():
    args, extra = parse_args()
    datasets = [args.dataset]
    programs = ["solver", "primal-dual"] if args.program == "all" else [args.program]
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

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
                str(model_path(args.model_dir, dataset, program, args)),
                "--epochs",
                str(args.epochs),
                "--max-steps",
                str(args.max_steps),
                "--lr",
                str(args.lr),
                "--hidden-dim",
                str(args.hidden_dim),
                "--batch-size",
                str(args.batch_size),
                *extra,
            ]

            if args.small_llm:
                base.extend(
                    [
                        "--baseline-model",
                        "causal-lm",
                        "--llm-backbone-path",
                        args.llm_backbone_path,
                    ]
                )

            if args.use_lora:
                base.extend(
                    [
                        "--use-lora",
                        "--lora-r",
                        str(args.lora_r),
                        "--lora-alpha",
                        str(args.lora_alpha),
                        "--lora-dropout",
                        str(args.lora_dropout),
                    ]
                )
                if args.lora_target_modules:
                    base.append("--lora-target-modules")
                    base.extend(args.lora_target_modules)

            if args.llm_device_map:
                base.extend(["--llm-device-map", args.llm_device_map])
            if args.gradient_checkpointing:
                base.append("--gradient-checkpointing")

            if args.constraint_warmup_iters is not None:
                base.extend(["--constraint-warmup-iters", str(args.constraint_warmup_iters)])

            if do_train:
                call_main([*base, "--train"], dry_run=args.dry_run, env=env)

            if do_eval:
                dfa_flags = [True] if args.use_dfa else [False, True]
                for use_dfa in dfa_flags:
                    cmd = [*base, "--evaluate"]
                    if use_dfa:
                        cmd.append("--use-dfa")
                    call_main(cmd, dry_run=args.dry_run, env=env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
