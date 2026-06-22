import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

EVAL_CONFIGS = {
    "lr1e-4": {
        "model": "models/eai_all_normal_causal-lm_lr0p0001_ep5_ms135_hd128_bs1_Qwen2p5-1p5B-Instruct_lora-r8_a16_d0p05_q_proj-v_proj_gc.pth",
        "hmm": "models/eai_all_qwen25_ctrlg_hmm.npz",
        "output": "results_eval_qwen15_lr1e-4.txt",
    },
    "lr1e-5": {
        "model": "models/eai_all_normal_causal-lm_lr1e-05_ep5_ms135_hd128_bs1_Qwen2p5-1p5B-Instruct_lora-r8_a16_d0p05_q_proj-v_proj_gc.pth",
        "hmm": "models/eai_all_normal_causal-lm_lr1e-05_ep5_ms135_lora_ctrlg_hmm.npz",
        "output": "results_eval_qwen15_lr1e-5.txt",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run EAI evaluation settings sequentially.")
    parser.add_argument("--run", choices=["all", *EVAL_CONFIGS.keys()], default="all")
    parser.add_argument("--dataset", choices=["all", "behavior", "virtualhome"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--max-steps", type=int, default=135)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--eval-split", choices=["dev", "train", "full"], default="dev")
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--llm-backbone-path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--baseline-model", default="causal-lm")
    parser.add_argument("--lora-target-modules", nargs="*", default=["q_proj", "v_proj"])
    parser.add_argument("--hmm-alpha", type=float, default=1.0)
    parser.add_argument("--hmm-search", choices=["greedy", "beam", "sample"], default="greedy")
    parser.add_argument("--hmm-lookahead-weight", type=float, default=0.0)
    parser.add_argument("--hmm-lookahead-max-steps", type=int, default=8)
    parser.add_argument("--skip-raw-qwen", action="store_true")
    parser.add_argument("--settings", nargs="+", choices=["0", "1", "2", "3", "4", "raw_dfa", "raw", "domiknows", "dfa", "hmm", "nt_domiknows", "nt_dfa", "nt_hmm_dfa", "nt_suite"], default=None)
    parser.add_argument("--constraint-modes", nargs="+", choices=["general", "specific"], default=["general", "specific"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def selected_configs(name):
    if name == "all":
        return EVAL_CONFIGS.items()
    return [(name, EVAL_CONFIGS[name])]


def as_path(value):
    path = Path(value)
    return path if path.is_absolute() else SCRIPT_DIR / path


def main():
    args = parse_args()
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    for name, config in selected_configs(args.run):
        model = as_path(config["model"])
        hmm = as_path(config["hmm"])
        output = as_path(config["output"])
        selected = set(args.settings or ["1", "2", "3", "4"])
        expanded = set(selected)
        if "nt_suite" in expanded:
            expanded.update({"nt_domiknows", "nt_dfa", "nt_hmm_dfa"})
        needs_model = bool(expanded & {"2", "3", "4", "domiknows", "dfa", "hmm"})
        needs_hmm = bool(expanded & {"4", "hmm", "nt_hmm_dfa"})
        required_paths = []
        if needs_model:
            required_paths.append(model)
        if needs_hmm:
            required_paths.append(hmm)
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            message = f"[{name}] missing artifact(s): " + ", ".join(missing)
            if args.allow_missing:
                print(message)
                print(f"[{name}] skipping")
                continue
            raise FileNotFoundError(message)

        cmd = [
            args.python,
            str(SCRIPT_DIR / "evaluate_settings.py"),
            "--dataset", args.dataset,
            "--device", args.device,
            "--max-steps", str(args.max_steps),
            "--eval-split", args.eval_split,
            "--dev-fraction", str(args.dev_fraction),
            "--baseline-model", args.baseline_model,
            "--llm-backbone-path", args.llm_backbone_path,
            "--use-lora",
            "--lora-target-modules", *args.lora_target_modules,
            "--gradient-checkpointing",
            "--hmm-alpha", str(args.hmm_alpha),
            "--hmm-search", args.hmm_search,
            "--hmm-lookahead-weight", str(args.hmm_lookahead_weight),
            "--hmm-lookahead-max-steps", str(args.hmm_lookahead_max_steps),
            "--output", str(output),
        ]
        if needs_model:
            cmd.extend(["--model", str(model)])
        if needs_hmm:
            cmd.extend(["--hmm", str(hmm)])
        if args.constraint_modes:
            cmd.extend(["--constraint-modes", *args.constraint_modes])
        if args.eval_limit is not None:
            cmd.extend(["--eval-limit", str(args.eval_limit)])
        if args.skip_raw_qwen:
            cmd.append("--skip-raw-qwen")
        if args.settings:
            cmd.extend(["--settings", *args.settings])

        print(f"\n===== Running {name} =====")
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, check=True)
        print(f"===== Finished {name}; results: {output} =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
