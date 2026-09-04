"""Direct Qwen answer fine-tuning for KB-VQA / VQAR GraphQA.

This baseline does not use image pixels. It trains Qwen from structured scene
facts, bounded KB facts, and the symbolic query to output one object id.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .dataset import DEFAULT_VQAR_ROOT, discover_vqar_dataset, load_kb_facts, load_vqar_tasks, vqar_task_to_graphqa_instance
from .execution import create_query_logic, materialize_bounded_facts
from .oracle import answer_object
from .train_predicate_classifier import choose_default_task_path, filter_kb_facts_for_instance

DEFAULT_MODEL = "/localscratch/premsrit/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DEFAULT_OUTPUT = Path("/egr/research-hlr2/premsrit/GraphQA/models/qwen3_8b_kbvqa_direct_answer_lora.pt")


def progress_iter(iterable, args, desc):
    if not args.progress:
        return iterable
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, desc=desc, dynamic_ncols=True, leave=True, mininterval=1.0)
    except Exception:
        return iterable


def load_instances(args):
    if args.task_path is None:
        task_path = choose_default_task_path(discover_vqar_dataset(args.root))
    else:
        task_path = args.task_path
    tasks = load_vqar_tasks(task_path, limit=args.limit)
    global_kb_facts = [] if args.no_kb else load_kb_facts(kb_dir=args.kb_dir)
    instances = []
    failures = []
    for index, task in enumerate(tasks):
        try:
            instance = vqar_task_to_graphqa_instance(task, kb_facts=[])
            kb_facts = filter_kb_facts_for_instance(
                instance,
                global_kb_facts,
                max_depth=args.kb_depth,
                max_extra_kg=args.max_extra_kg_facts,
            )
            instance["kb_facts"] = kb_facts
            instance["facts"] = materialize_bounded_facts(instance)
            gold = instance.get("expected_answer") or answer_object(instance)
            if gold is None:
                failures.append((index, "NoAnswer", "No unique answer"))
                continue
            instance["expected_answer"] = str(gold)
            instances.append(instance)
        except Exception as exc:
            failures.append((index, type(exc).__name__, str(exc)))
    return task_path, instances, failures


def split_instances(instances, dev_fraction=0.1, seed=13):
    instances = list(instances)
    random.Random(seed).shuffle(instances)
    if len(instances) <= 1 or dev_fraction <= 0:
        return instances, []
    dev_size = min(max(1, int(round(len(instances) * dev_fraction))), len(instances) - 1)
    return instances[dev_size:], instances[:dev_size]


def compact_facts(facts, limit):
    facts = list(facts or [])[: max(0, int(limit))]
    return "; ".join(f"{p}({l},{r})" for p, l, r in facts)


def prompt_for_instance(instance, max_facts=120):
    objects = [str(obj) for obj in instance.get("objects", [])]
    query_logic = create_query_logic(instance)
    facts = compact_facts(instance.get("facts") or materialize_bounded_facts(instance), max_facts)
    # The training/eval code left-truncates long prompts. Keep the executable
    # question and candidate answers next to the supervised answer tokens.
    return "\n".join([
        "Task: answer a KB-VQA question from structured scene and knowledge-base facts.",
        f"Facts: {facts}",
        f"Query: {instance.get('query', {})}",
        f"Executable logic: {query_logic}",
        "Choose exactly one candidate object id from the list below.",
        f"Candidate objects: {', '.join(objects)}",
        "Final answer:",
    ])


class QwenDirectAnswer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device_name = args.device
        self.max_length = int(args.max_length)
        self.max_answer_tokens = int(args.max_answer_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if str(args.device).startswith("cuda"):
            model_kwargs["dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
        if args.lora_r > 0:
            from peft import LoraConfig, TaskType, get_peft_model
            targets = [m.strip() for m in str(args.lora_target_modules).split(",") if m.strip()]
            config = LoraConfig(
                r=int(args.lora_r),
                lora_alpha=int(args.lora_alpha),
                lora_dropout=float(args.lora_dropout),
                target_modules=targets,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, config)
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False
        elif args.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.to(args.device)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def loss_for_batch(self, instances, args):
        rows = []
        label_rows = []
        pad_id = self.tokenizer.pad_token_id
        for instance in instances:
            prompt = prompt_for_instance(instance, args.max_facts)
            answer = " " + str(instance["expected_answer"])
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)[: self.max_answer_tokens]
            keep_prompt = max(1, self.max_length - len(answer_ids))
            ids = prompt_ids[-keep_prompt:] + answer_ids
            labels = [-100] * (len(ids) - len(answer_ids)) + answer_ids
            rows.append(ids)
            label_rows.append(labels)
        max_len = max(len(row) for row in rows)
        input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long, device=self.device_name)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=self.device_name)
        labels = torch.full((len(rows), max_len), -100, dtype=torch.long, device=self.device_name)
        for i, (ids, labs) in enumerate(zip(rows, label_rows)):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long, device=self.device_name)
            attention_mask[i, :n] = 1
            labels[i, :n] = torch.tensor(labs, dtype=torch.long, device=self.device_name)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out.loss

    def score_choices(self, instance, choices, args):
        prompt = prompt_for_instance(instance, args.max_facts)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        rows, masks, label_masks = [], [], []
        pad_id = self.tokenizer.pad_token_id
        for choice in choices:
            ans_ids = self.tokenizer.encode(" " + str(choice), add_special_tokens=False)[: self.max_answer_tokens]
            keep_prompt = max(1, self.max_length - len(ans_ids))
            ids = prompt_ids[-keep_prompt:] + ans_ids
            rows.append(ids)
            masks.append([1] * len(ids))
            label_masks.append([0] * (len(ids) - len(ans_ids)) + [1] * len(ans_ids))
        max_len = max(len(row) for row in rows)
        input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long, device=self.device_name)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=self.device_name)
        label_mask = torch.zeros((len(rows), max_len), dtype=torch.bool, device=self.device_name)
        for i, (ids, mask, lm) in enumerate(zip(rows, masks, label_masks)):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long, device=self.device_name)
            attention_mask[i, :n] = torch.tensor(mask, dtype=torch.long, device=self.device_name)
            label_mask[i, :n] = torch.tensor(lm, dtype=torch.bool, device=self.device_name)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = out.logits[:, :-1, :].log_softmax(dim=-1)
        target_ids = input_ids[:, 1:]
        target_label_mask = label_mask[:, 1:]
        token_scores = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        scores = (token_scores * target_label_mask.float()).sum(dim=-1)
        lengths = target_label_mask.float().sum(dim=-1).clamp_min(1.0)
        return scores / lengths


def run_epoch(model, instances, args, optimizer=None, epoch=None):
    training = optimizer is not None
    model.train(training)
    if args.freeze_backbone and not any(p.requires_grad for p in model.model.parameters()):
        model.model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    iterator = progress_iter(range(0, len(instances), args.batch_size), args, f"{'train' if training else 'eval'} epoch={epoch}")
    for start in iterator:
        batch = instances[start : start + args.batch_size]
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss = model.loss_for_batch(batch, args)
        if optimizer is not None:
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.max_grad_norm)
            optimizer.step()
        total_loss += float(loss.detach().item()) * len(batch)
        total += len(batch)
        with torch.no_grad():
            for instance in batch:
                choices = [str(obj) for obj in instance.get("objects", [])]
                if not choices:
                    continue
                scores = model.score_choices(instance, choices, args)
                pred = choices[int(torch.argmax(scores).item())]
                correct += int(pred == str(instance["expected_answer"]))
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{total_loss / max(total, 1):.4f}", acc=f"{correct / max(total, 1):.3f}", n=total)
    return {"instances": total, "loss": total_loss / total if total else 0.0, "answer_acc": correct / total if total else 0.0}


def save_checkpoint(path, model, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"args": vars(args)}
    if args.lora_r > 0:
        from peft import get_peft_model_state_dict
        state["backbone_lora"] = get_peft_model_state_dict(model.model)
    else:
        state["model_state_dict"] = model.state_dict()
    torch.save(state, path)
    return path


def load_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "backbone_lora" in checkpoint:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model.model, checkpoint["backbone_lora"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Direct Qwen LoRA answer fine-tuning for KB-VQA / VQAR GraphQA.")
    parser.add_argument("--root", type=Path, default=DEFAULT_VQAR_ROOT)
    parser.add_argument("--task-path", type=Path, default=None)
    parser.add_argument("--kb-dir", type=Path, default=None)
    parser.add_argument("--no-kb", action="store_true")
    parser.add_argument("--kb-depth", type=int, default=2)
    parser.add_argument("--max-extra-kg-facts", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-answer-tokens", type=int, default=8)
    parser.add_argument("--max-facts", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    task_path, instances, failures = load_instances(args)
    train, dev = split_instances(instances, args.dev_fraction, args.seed)
    print(f"task_path={task_path}", flush=True)
    print(f"loaded={len(instances)} train={len(train)} dev={len(dev)} failures={len(failures)} device={args.device}", flush=True)
    if failures[:5]:
        print(f"first_failures={failures[:5]}", flush=True)
    model = QwenDirectAnswer(args)
    if args.checkpoint:
        loaded = load_checkpoint(args.checkpoint, model, args.device)
        print(f"loaded_checkpoint={args.checkpoint}", flush=True)
        if "args" in loaded:
            print(f"checkpoint_args={json.dumps(loaded['args'], sort_keys=True, default=str)}", flush=True)
    if args.eval_only:
        with torch.no_grad():
            score = run_epoch(model, instances, args, optimizer=None, epoch="eval_only")
        print(f"eval={json.dumps(score, sort_keys=True)}", flush=True)
        return 0
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = None
    for epoch in range(1, args.epochs + 1):
        train_score = run_epoch(model, train, args, optimizer=optimizer, epoch=epoch)
        with torch.no_grad():
            dev_score = run_epoch(model, dev, args, optimizer=None, epoch=epoch)
        print(f"epoch={epoch} train={json.dumps(train_score, sort_keys=True)}", flush=True)
        print(f"epoch={epoch} dev={json.dumps(dev_score, sort_keys=True)}", flush=True)
        if best is None or dev_score["answer_acc"] >= best:
            best = dev_score["answer_acc"]
            saved = save_checkpoint(args.output, model, args)
            print(f"saved={saved}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
