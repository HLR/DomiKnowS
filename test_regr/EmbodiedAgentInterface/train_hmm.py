import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def progress_bar(iterable, **kwargs):
    return tqdm(iterable, **kwargs) if tqdm is not None else iterable

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(SCRIPT_DIR.parents[1]))

from eai_hmm_decoder_adapter import build_hmm_from_label_sequences
from main import (
    generation_vocab_from_examples,
    labels_to_actions,
    load_examples,
    load_trained_program,
    object_tokens_from_examples,
    action_tokens_requiring_object_from_examples,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Distill/train a Ctrl-G-style HMM from a trained EAI DomiKnowS generator.")
    parser.add_argument("--dataset", choices=["all", "behavior", "virtualhome"], default="all")
    parser.add_argument("--split", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows used to build/load the EAI dataset. Use the same value as the checkpoint training run, usually None for full-vocab checkpoints.")
    parser.add_argument("--distill-limit", type=int, default=None, help="Limit only the examples used to distill the HMM after the model is loaded.")
    parser.add_argument("--max-steps", type=int, default=135)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", default=None, help="Saved DomiKnowS EAI model checkpoint to load. Defaults to train.py's model-dir convention.")
    parser.add_argument("--model-dir", default=str(SCRIPT_DIR / "models"), help="Directory used by train.py for saved EAI checkpoints.")
    parser.add_argument("--output", default=None, help="Output HMM npz. Defaults to model-dir/eai_{dataset}_{program}_ctrlg_hmm.npz.")

    parser.add_argument("--program", choices=["solver", "primal-dual"], default="solver")
    parser.add_argument("--baseline-model", choices=["tiny-transformer", "bert-gru", "causal-lm"], default="causal-lm")
    parser.add_argument("--llm-backbone-path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--encoder-model-path", default="bert-base-uncased")
    parser.add_argument("--encoder-max-length", type=int, default=256)
    parser.add_argument("--finetune-encoder", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--feature-dim", type=int, default=None)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=None)
    parser.add_argument("--llm-device-map", default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--samples-per-example", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--do-sample", action="store_true", help="Sample from the trained model. Default is greedy distillation.")
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--emission-smoothing", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--show", type=int, default=3, help="Print this many generated distillation samples.")
    return parser.parse_args()


def checkpoint_path(model_dir, dataset, program):
    suffix = "normal" if program == "solver" else "pmd"
    return Path(model_dir) / f"eai_{dataset}_{suffix}.pth"


def hmm_output_path(model_dir, dataset, program):
    suffix = "normal" if program == "solver" else "pmd"
    return Path(model_dir) / f"eai_{dataset}_{suffix}_ctrlg_hmm.npz"


def _model_args(args):
    # load_trained_program expects the same fields used by main.py training.
    return SimpleNamespace(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        encoder_model_path=args.encoder_model_path,
        encoder_max_length=args.encoder_max_length,
        finetune_encoder=args.finetune_encoder,
        max_steps=args.max_steps,
        program=args.program,
        baseline_model=args.baseline_model,
        llm_backbone_path=args.llm_backbone_path,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        llm_device_map=args.llm_device_map,
        gradient_checkpointing=args.gradient_checkpointing,
        model=args.model,
    )


def trim_after_eos(labels, eos_label):
    out = []
    for label in labels:
        label = int(label)
        out.append(label)
        if label == int(eos_label):
            break
    return out or [int(eos_label)]


def sample_next_label(logits, temperature=1.0, top_k=0, do_sample=False):
    logits = logits.detach().float()
    if not do_sample:
        return int(torch.argmax(logits, dim=-1).item())
    temperature = max(float(temperature), 1e-6)
    scores = logits / temperature
    if top_k and top_k > 0 and top_k < scores.numel():
        values, indices = torch.topk(scores, top_k)
        probs = torch.softmax(values, dim=-1)
        return int(indices[torch.multinomial(probs, 1).item()].item())
    probs = torch.softmax(scores, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def generate_sequence(generator, sample, vocabulary, max_steps, temperature=1.0, top_k=0, do_sample=False):
    prefix = [int(vocabulary.eos_label)]
    labels = []
    text = sample.get("text") or sample.get("natural_language_description") or ""
    for _ in range(max_steps):
        logits = generator.next_label_logits(prefix, text=text)
        label = sample_next_label(logits, temperature=temperature, top_k=top_k, do_sample=do_sample)
        labels.append(label)
        prefix.append(label)
        if label == int(vocabulary.eos_label):
            break
    if not labels or labels[-1] != int(vocabulary.eos_label):
        labels.append(int(vocabulary.eos_label))
    return labels[:max_steps]


def save_hmm(path, hmm, vocabulary, sequences, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens = np.asarray(list(vocabulary.tokens), dtype=object)
    lengths = np.asarray([len(seq) for seq in sequences], dtype=np.int64)
    metadata = {
        "dataset": args.dataset,
        "model": args.model,
        "max_steps": args.max_steps,
        "loaded_examples": None if args.limit is None else int(args.limit),
        "distill_examples": None if args.distill_limit is None else int(args.distill_limit),
        "samples_per_example": args.samples_per_example,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "do_sample": args.do_sample,
        "smoothing": args.smoothing,
        "emission_smoothing": args.emission_smoothing,
        "eos_label": int(vocabulary.eos_label),
        "label_count": int(vocabulary.label_count),
    }
    np.savez_compressed(
        path,
        alpha_exp=hmm.alpha_exp,
        beta=hmm.beta,
        gamma=hmm.gamma,
        eos_label=np.asarray([vocabulary.eos_label], dtype=np.int64),
        tokens=tokens,
        sequence_lengths=lengths,
        metadata=np.asarray(json.dumps(metadata), dtype=object),
    )
    return path


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model is None:
        args.model = str(checkpoint_path(args.model_dir, args.dataset, args.program))
    if args.output is None:
        args.output = str(hmm_output_path(args.model_dir, args.dataset, args.program))

    examples = load_examples(args, args.device)
    model_args = _model_args(args)
    program, bundle = load_trained_program(model_args, examples, args.device)
    generator = program.autoregressive_head
    generator.eval()

    distill_examples = examples if args.distill_limit is None else examples[: args.distill_limit]

    sequences = []
    shown = 0
    total_generations = len(distill_examples) * args.samples_per_example
    progress = progress_bar(
        distill_examples,
        total=len(distill_examples),
        desc="Model HMM distillation",
    )
    with torch.no_grad():
        for sample in progress:
            for _ in range(args.samples_per_example):
                labels = generate_sequence(
                    generator,
                    sample,
                    bundle.vocabulary,
                    args.max_steps,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    do_sample=args.do_sample,
                )
                labels = trim_after_eos(labels, bundle.vocabulary.eos_label)
                sequences.append(labels)
                if shown < args.show:
                    print(f"sample_{shown}: {labels_to_actions(labels, bundle.vocabulary)}")
                    shown += 1
                if tqdm is not None and hasattr(progress, "set_postfix"):
                    progress.set_postfix(sequences=len(sequences), total=total_generations)

    hmm = build_hmm_from_label_sequences(
        sequences,
        vocab_size=bundle.vocabulary.label_count,
        smoothing=args.smoothing,
        emission_smoothing=args.emission_smoothing,
        start_label=bundle.vocabulary.eos_label,
        eos_label=bundle.vocabulary.eos_label,
    )
    output = save_hmm(args.output, hmm, bundle.vocabulary, sequences, args)
    print(f"trained_hmm_sequences={len(sequences)}")
    print(f"hmm_hidden_states={hmm.hidden_states} vocab_size={hmm.vocab_size}")
    print(f"saved_hmm={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
