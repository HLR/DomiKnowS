import argparse
import json
import random
import re
import sys
from pathlib import Path

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
from dataset import EOS_TOKEN, dummy_dataset, load_eai_dataset
from main import labels_to_actions
from modules import _prepare_transformers_imports


def parse_args():
    parser = argparse.ArgumentParser(description="Distill a Ctrl-G-style HMM from raw Qwen2.5 text outputs for EAI.")
    parser.add_argument("--dataset", choices=["all", "behavior", "virtualhome"], default="all")
    parser.add_argument("--split", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--distill-limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=135)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--llm-backbone-path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--samples-per-example", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--emission-smoothing", type=float, default=0.01)
    parser.add_argument("--output", default=str(SCRIPT_DIR / "models" / "eai_qwen25_ctrlg_hmm.npz"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--show", type=int, default=3)
    return parser.parse_args()


def load_examples(args):
    if args.dummy:
        return dummy_dataset(device="cpu", max_steps=args.max_steps)
    return load_eai_dataset(
        dataset_name=args.dataset,
        split=args.split,
        limit=args.limit,
        data_path=args.data_path,
        device="cpu",
        max_steps=args.max_steps,
    )


def generation_vocab_from_examples(examples):
    if not examples:
        return (EOS_TOKEN, "other")
    return tuple(examples[0]["generation_vocab"])


def build_prompt(sample, vocab, max_steps):
    non_eos = [token for token in vocab if token != EOS_TOKEN]
    actions = sorted(set(sample.get("action_tokens", ())))
    objects = sorted(set(sample.get("object_tokens", ())))
    action_hint = ", ".join(actions[:80]) if actions else ", ".join(non_eos[:120])
    object_hint = ", ".join(objects[:120]) if objects else ", ".join(non_eos[:120])
    instruction = sample.get("natural_language_description") or sample.get("text") or ""
    goal = sample.get("tl_goal") or ""
    return (
        "You are generating an embodied-agent action plan.\n"
        "Return only space-separated tokens from the allowed vocabulary. Do not add explanations.\n"
        "The sequence should alternate action/object when an action needs an object, and end with <eos>.\n"
        f"Maximum tokens including <eos>: {max_steps}.\n"
        f"Allowed action examples: {action_hint}.\n"
        f"Allowed object examples: {object_hint}.\n"
        f"Instruction: {instruction}\n"
        f"Goal: {goal}\n"
        "Plan tokens:"
    )


def normalize_piece(text):
    text = str(text).strip().lower()
    text = text.strip("`'\".,;:()[]{}<>")
    text = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
    return text


def parse_generated_text(text, vocab, max_steps):
    aliases = {token.lower(): token for token in vocab}
    base_aliases = {}
    for vocab_token in vocab:
        match = re.match(r"^(.+)_\d+$", vocab_token.lower())
        if match and match.group(1) not in aliases:
            base_aliases.setdefault(match.group(1), vocab_token)
    normalized = str(text).lower().replace("<eos>", " <eos> ")
    # Qwen often writes function-like text such as open(cabinet_1).  Split
    # punctuation into token boundaries while preserving underscores and <eos>.
    normalized = re.sub(r"[^a-z0-9_<>]+", " ", normalized)
    labels = []
    for raw in normalized.split():
        token = normalize_piece(raw)
        if not token:
            continue
        if token in {"plan", "tokens", "action", "object"}:
            continue
        if token == "eos":
            token = EOS_TOKEN
        mapped = aliases.get(token) or base_aliases.get(token)
        if mapped is None:
            continue
        labels.append(mapped)
        if mapped == EOS_TOKEN or len(labels) >= max_steps:
            break
    if not labels or labels[-1] != EOS_TOKEN:
        labels.append(EOS_TOKEN)
    return labels[:max_steps]


def labels_to_ids(tokens, vocab):
    label_to_id = {token: idx for idx, token in enumerate(vocab)}
    fallback = label_to_id.get("other", label_to_id.get(EOS_TOKEN, 0))
    return [label_to_id.get(token, fallback) for token in tokens]


def load_qwen(model_path, device):
    _prepare_transformers_imports()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return tokenizer, model


def generate_text(tokenizer, model, prompt, args):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(args.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
            top_p=args.top_p if args.do_sample else None,
            top_k=args.top_k if args.do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def save_hmm(path, hmm, vocab, sequences, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": "raw_qwen_text",
        "dataset": args.dataset,
        "llm_backbone_path": args.llm_backbone_path,
        "max_steps": args.max_steps,
        "loaded_examples": None if args.limit is None else int(args.limit),
        "distill_examples": None if args.distill_limit is None else int(args.distill_limit),
        "samples_per_example": args.samples_per_example,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": args.do_sample,
        "smoothing": args.smoothing,
        "emission_smoothing": args.emission_smoothing,
        "eos_label": int(vocab.index(EOS_TOKEN)),
        "label_count": len(vocab),
    }
    np.savez_compressed(
        path,
        alpha_exp=hmm.alpha_exp,
        beta=hmm.beta,
        gamma=hmm.gamma,
        eos_label=np.asarray([vocab.index(EOS_TOKEN)], dtype=np.int64),
        tokens=np.asarray(list(vocab), dtype=object),
        sequence_lengths=np.asarray([len(seq) for seq in sequences], dtype=np.int64),
        metadata=np.asarray(json.dumps(metadata), dtype=object),
    )
    return path


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    examples = load_examples(args)
    distill_examples = examples if args.distill_limit is None else examples[: args.distill_limit]
    vocab = generation_vocab_from_examples(examples)
    eos_label = int(vocab.index(EOS_TOKEN))
    tokenizer, model = load_qwen(args.llm_backbone_path, args.device)

    sequences = []
    shown = 0
    total_generations = len(distill_examples) * args.samples_per_example
    progress = progress_bar(
        distill_examples,
        total=len(distill_examples),
        desc="Qwen HMM distillation",
    )
    for sample in progress:
        prompt = build_prompt(sample, vocab, args.max_steps)
        for _ in range(args.samples_per_example):
            generated = generate_text(tokenizer, model, prompt, args)
            token_labels = parse_generated_text(generated, vocab, args.max_steps)
            label_ids = labels_to_ids(token_labels, vocab)
            sequences.append(label_ids)
            if shown < args.show:
                print(f"raw_{shown}: {generated}")
                print(f"parsed_{shown}: {token_labels}")
                shown += 1
            if tqdm is not None and hasattr(progress, "set_postfix"):
                progress.set_postfix(sequences=len(sequences), total=total_generations)

    hmm = build_hmm_from_label_sequences(
        sequences,
        vocab_size=len(vocab),
        smoothing=args.smoothing,
        emission_smoothing=args.emission_smoothing,
        start_label=eos_label,
        eos_label=eos_label,
    )
    output = save_hmm(args.output, hmm, vocab, sequences, args)
    print(f"trained_hmm_sequences={len(sequences)}")
    print(f"hmm_hidden_states={hmm.hidden_states} vocab_size={hmm.vocab_size}")
    print(f"saved_hmm={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
