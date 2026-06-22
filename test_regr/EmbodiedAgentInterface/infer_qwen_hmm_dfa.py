"""Inference-only EAI evaluation: DomiKnowS default Qwen generator + HMM + DFA.

This script does not train or load a trained checkpoint. It follows the normal
`main.py` construction path: load examples, call `build_trainable_program(...)`
to create the graph/bundle/default DomiKnowS generator, compile the graph to a
DFA, and decode with HMM + DFA using that generator's next-label logits.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import evaluate_settings as ev
from main import build_trainable_program

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference-only EAI DomiKnowS-Qwen + HMM + DFA decoding.")
    parser.add_argument("--dataset", choices=["all", "behavior", "virtualhome"], default="all")
    parser.add_argument("--split", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples loaded before building graph/vocabulary.")
    parser.add_argument("--eval-limit", type=int, default=None, help="Limit selected examples scored.")
    parser.add_argument("--eval-split", choices=["dev", "train", "full"], default="full")
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--max-steps", type=int, default=135)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", default=str(SCRIPT_DIR / "results_qwen_hmm_dfa_inference.txt"))
    parser.add_argument("--show", type=int, default=0)

    # Match the graph/generator construction arguments expected by main.py.
    parser.add_argument("--program", choices=["solver", "primal-dual"], default="solver")
    parser.add_argument("--baseline-model", choices=["bert-gru", "tiny-transformer", "causal-lm"], default="causal-lm")
    parser.add_argument("--feature-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--encoder-model-path", default="bert-base-uncased")
    parser.add_argument("--encoder-max-length", type=int, default=256)
    parser.add_argument("--finetune-encoder", action="store_true")
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--llm-backbone-path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--llm-device-map", default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="*", default=None)

    parser.add_argument("--hmm", default=str(SCRIPT_DIR / "models/eai_all_qwen25_ctrlg_hmm.npz"))
    parser.add_argument("--hmm-search", choices=["greedy", "beam", "sample"], default="greedy")
    parser.add_argument("--hmm-beam-size", type=int, default=4)
    parser.add_argument("--hmm-weight", type=float, default=1.0)
    parser.add_argument("--hmm-hf-weight", type=float, default=0.0, help="Optional backend generator label-bias weight. Use 0.0 for Ctrl-G-style HMM+DFA decoding.")
    parser.add_argument("--hmm-lookahead-weight", type=float, default=0.0)
    parser.add_argument("--hmm-lookahead-max-steps", type=int, default=8)
    parser.add_argument("--hmm-keep-rejected", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    all_examples = ev.load_examples(args, args.device)
    if not all_examples:
        raise ValueError("No EAI examples were loaded.")
    examples = ev.select_eval_examples(all_examples, args.eval_split, args.dev_fraction, args.eval_limit)
    if not examples:
        raise ValueError(f"No examples selected for eval_split={args.eval_split!r}.")

    program, bundle = build_trainable_program(args, all_examples, args.device)
    predictions = ev.hmm_dfa_predictions(args, program, bundle, examples)

    from domiknows.generation import constraints_to_dfa_from_graph

    dfa = constraints_to_dfa_from_graph(program.graph, bundle)
    score = ev.score_predictions(
        "DomiKnowS HMM+DFA decoder with Qwen-distilled HMM (no training)",
        predictions,
        examples,
        bundle.vocabulary,
        dfa=dfa,
        show=args.show,
    )

    lines = [
        "EAI inference-only DomiKnowS HMM+DFA decoder with Qwen-distilled HMM",
        f"dataset={args.dataset} eval_split={args.eval_split} examples={len(examples)} loaded_examples={len(all_examples)} max_steps={args.max_steps}",
        f"generator={args.baseline_model} qwen={args.llm_backbone_path}",
        f"hmm={args.hmm}",
        f"hmm_search={args.hmm_search} hmm_weight={args.hmm_weight} backend_generator_weight={args.hmm_hf_weight} lookahead_weight={args.hmm_lookahead_weight}",
        "",
        ev.format_score(score),
        "",
        json.dumps([score], indent=2),
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    for line in lines[:5] + [ev.format_score(score)]:
        print(line)
    print(f"saved_results={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
