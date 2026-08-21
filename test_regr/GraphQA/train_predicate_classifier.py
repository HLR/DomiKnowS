import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .dataset import DEFAULT_VQAR_ROOT, discover_vqar_dataset, load_kb_facts, load_vqar_tasks, vqar_task_to_graphqa_instance
from .execution import materialize_bounded_facts
from .graph import canonical_relation, collect_object_relations
from .modules import (
    GraphQAPredicateClassifier,
    NO_RELATION_LABEL,
    create_predicate_examples,
    label_spaces,
    _object_pair_feature_prompt,
    _object_pair_prompt,
    _object_symbol_feature_prompt,
    _object_symbol_prompt,
)


DEFAULT_MODEL = "/localscratch/premsrit/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DEFAULT_OUTPUT = Path("/egr/research-hlr2/premsrit/GraphQA/models/qwen3_8b_graphqa_predicates.pt")


def progress_iter(iterable, args, desc):
    if not args.progress:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, dynamic_ncols=True, leave=True, mininterval=1.0)
    except Exception:
        return iterable


def choose_default_task_path(discovered):
    if not discovered["task_paths"]:
        raise FileNotFoundError(f"No VQAR task files found under {discovered['data_dir']}")
    preferred = [
        "train_tasks.pkl",
        "train_tasks_c2_10000.pkl",
        "train_tasks_c2_1000.pkl",
        "train_tasks_c2_100.pkl",
        "train_tasks_c2_10.pkl",
    ]
    by_name = {path.name: path for path in discovered["task_paths"]}
    for name in preferred:
        if name in by_name:
            return by_name[name]
    train_paths = [path for path in discovered["task_paths"] if path.name.startswith("train_")]
    return train_paths[0] if train_paths else discovered["task_paths"][0]


def filter_kb_facts_for_instance(instance, kb_facts, max_depth=2, max_extra_kg=256):
    needed = set()
    for pred, _obj, symbol in instance.get("visual_facts", []):
        if canonical_relation(pred) in {"Name", "Attribute"} and symbol is not None:
            needed.add(symbol)
    query = instance.get("query", {})
    if query.get("target_type") not in (None, "__any_object__"):
        needed.add(query["target_type"])
    for conditions in [query.get("conditions", [])] + list(query.get("alternatives", [])):
        for pred, _left, right in conditions:
            pred = canonical_relation(pred)
            if pred in {"Name", "Attribute", "ObjectType", "ObjectCategory", "SemanticClass"}:
                needed.add(right)
            elif pred == "KG":
                _rel, dst = right
                needed.add(dst)

    filtered = []
    frontier = set(needed)
    seen = set()
    for _depth in range(max_depth):
        next_frontier = set()
        for pred, left, right in kb_facts:
            pred = canonical_relation(pred)
            fact = (pred, left, right)
            if pred == "TypeOf" and left in frontier and fact not in seen:
                filtered.append(fact)
                seen.add(fact)
                next_frontier.add(right)
        frontier = next_frontier
        needed.update(next_frontier)

    extra_kg = 0
    for pred, left, right in kb_facts:
        pred = canonical_relation(pred)
        if pred == "TypeOf":
            continue
        fact = (pred, left, right)
        if fact in seen:
            continue
        if left in needed or right in needed:
            filtered.append(fact)
            seen.add(fact)
            extra_kg += 1
            if extra_kg >= max_extra_kg:
                break
    return filtered


def load_instances(args):
    if args.task_path is None:
        discovered = discover_vqar_dataset(args.root)
        task_path = choose_default_task_path(discovered)
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
            instance["symbols"] = sorted(set(instance.get("symbols", [])) | {s for _p, l, r in kb_facts for s in (l, r)})
            instance["facts"] = materialize_bounded_facts(instance)
            instances.append(instance)
        except Exception as exc:
            failures.append((index, type(exc).__name__, str(exc)))
    return task_path, instances, failures


def split_instances(instances, dev_fraction=0.1, seed=13):
    instances = list(instances)
    random.Random(seed).shuffle(instances)
    if len(instances) <= 1:
        return instances, []
    dev_size = min(max(1, int(round(len(instances) * dev_fraction))), len(instances) - 1)
    return instances[dev_size:], instances[:dev_size]


def build_examples(instances, args):
    examples = []
    for instance in instances:
        if args.grounding_only:
            examples.extend(create_grounding_predicate_examples(instance, args))
        else:
            examples.extend(create_predicate_examples(instance))
    if args.grounding_only:
        examples = balance_grounding_examples(examples, args)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    return examples


def balance_grounding_examples(examples, args):
    """Keep NoRelation negatives useful without letting them dominate training."""
    max_ratio = getattr(args, "max_grounding_negatives_per_positive", None)
    max_total = getattr(args, "max_grounding_examples", None)
    if max_ratio is None and max_total is None:
        return examples

    rng = random.Random(args.seed)
    positives = {}
    negatives = {}
    for example in examples:
        bucket = negatives if example["label"] == NO_RELATION_LABEL else positives
        bucket.setdefault(example["kind"], []).append(example)

    balanced = []
    for kind in sorted(set(positives) | set(negatives)):
        kind_pos = list(positives.get(kind, []))
        kind_neg = list(negatives.get(kind, []))
        balanced.extend(kind_pos)
        if max_ratio is None:
            keep_neg = kind_neg
        else:
            # If a family has no positives in this split, keep a tiny negative
            # sample only for shape coverage, not for dominating the objective.
            base = max(1, len(kind_pos))
            keep_count = min(len(kind_neg), int(max_ratio * base))
            rng.shuffle(kind_neg)
            keep_neg = kind_neg[:keep_count]
        balanced.extend(keep_neg)

    rng.shuffle(balanced)
    if max_total is not None and len(balanced) > max_total:
        positives_all = [ex for ex in balanced if ex["label"] != NO_RELATION_LABEL]
        negatives_all = [ex for ex in balanced if ex["label"] == NO_RELATION_LABEL]
        rng.shuffle(negatives_all)
        keep_neg = max(0, max_total - len(positives_all))
        balanced = positives_all + negatives_all[:keep_neg]
        rng.shuffle(balanced)
    return balanced


def create_grounding_predicate_examples(instance, args=None):
    """Create visual grounding rows with explicit NoRelation negatives.

    The executor keeps KB/TypeOf symbolic and asks Qwen only for scene atoms:
    Name, Attribute, and object-object relations.  Positive-only supervision is
    dangerous here because every candidate pair at execution time would be
    forced into a positive class; these capped negative rows teach the heads when
    to emit NoRelation.
    """
    examples = []
    query = instance.get("query", {})
    objects = [str(obj) for obj in instance.get("objects", [])]
    symbols = [str(sym) for sym in instance.get("symbols", []) if not str(sym).startswith("__")]
    object_rels = set(collect_object_relations([instance]))
    max_symbols = int(getattr(args, "max_grounding_symbols", 64) if args is not None else 64)
    max_pairs = int(getattr(args, "max_grounding_object_pairs", 128) if args is not None else 128)

    true_object_symbol = {}
    true_object_pair = {}
    needed_symbols = set()
    for pred, left, right in instance.get("visual_facts", []):
        pred = canonical_relation(pred)
        left = str(left)
        right = str(right)
        if pred in {"Name", "Attribute"}:
            true_object_symbol[(left, right)] = pred
            needed_symbols.add(right)
        elif pred in object_rels:
            true_object_pair[(left, right)] = pred

    # Prefer symbols mentioned by the query and positives, then add a bounded
    # slice of the remaining symbols for NoRelation negatives.
    q = instance.get("query", {}) or {}
    for conds in [q.get("conditions", [])] + list(q.get("alternatives", []) or []):
        for pred, _left, right in conds:
            pred = canonical_relation(pred)
            if pred in {"Name", "Attribute", "ObjectType", "ObjectCategory", "SemanticClass"} and right is not None:
                needed_symbols.add(str(right))
    selected_symbols = []
    for sym in symbols:
        if sym in needed_symbols and sym not in selected_symbols:
            selected_symbols.append(sym)
    for sym in symbols:
        if max_symbols > 0 and len(selected_symbols) >= max_symbols:
            break
        if sym not in selected_symbols:
            selected_symbols.append(sym)

    for obj in objects:
        for sym in selected_symbols:
            label = true_object_symbol.get((obj, sym), NO_RELATION_LABEL)
            examples.append({
                "kind": "object_symbol",
                "label": label,
                "prompt": _object_symbol_feature_prompt(instance, obj, sym, query, labels=["Attribute", "Name", NO_RELATION_LABEL]),
            })

    pair_keys = [(src, dst) for src in objects for dst in objects if src != dst]
    if max_pairs > 0:
        positives = [key for key in pair_keys if key in true_object_pair]
        negatives = [key for key in pair_keys if key not in true_object_pair]
        pair_keys = positives + negatives[: max(0, max_pairs - len(positives))]
    for src, dst in pair_keys:
        label = true_object_pair.get((src, dst), NO_RELATION_LABEL)
        examples.append({
            "kind": "object_pair",
            "label": label,
            "prompt": _object_pair_feature_prompt(instance, src, dst, query, labels=collect_object_relations([instance]) + [NO_RELATION_LABEL]),
        })
    return examples


def grounding_label_spaces(instances):
    return {
        "object_symbol": ["Attribute", "Name", NO_RELATION_LABEL],
        # Keep a one-label KG head for checkpoint compatibility, but do not
        # train/evaluate it in --grounding-only mode.  The executor reads KB
        # facts symbolically instead.
        "symbol_pair": ["TypeOf", NO_RELATION_LABEL],
        "object_pair": collect_object_relations(instances) + [NO_RELATION_LABEL],
    }


def run_epoch(model, examples, label_to_index, args, optimizer=None, epoch=None):
    training = optimizer is not None
    model.train(training)
    if args.freeze_backbone and not model.backbone_has_trainable_parameters():
        model.backbone.eval()
    random.shuffle(examples)
    total_loss = 0.0
    total = 0
    correct = 0
    kind_totals = {"object_symbol": 0, "symbol_pair": 0, "object_pair": 0}
    kind_correct = {"object_symbol": 0, "symbol_pair": 0, "object_pair": 0}
    iterator = progress_iter(range(0, len(examples), args.batch_size), args, f"{'train' if training else 'eval'} epoch={epoch}")
    for start in iterator:
        batch = examples[start : start + args.batch_size]
        if not batch:
            continue
        # Keep batches homogeneous because each predicate family has its own head.
        grouped = {}
        for example in batch:
            grouped.setdefault(example["kind"], []).append(example)
        loss_parts = []
        batch_total = 0
        batch_correct = 0
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        for kind, kind_examples in grouped.items():
            logits = model.forward_examples(kind_examples)
            target = torch.tensor(
                [label_to_index[kind][example["label"]] for example in kind_examples],
                dtype=torch.long,
                device=model.device_name,
            )
            loss = F.cross_entropy(logits, target)
            loss_parts.append(loss)
            pred = torch.argmax(logits.detach(), dim=-1)
            hits = int((pred == target).sum().item())
            kind_totals[kind] += len(kind_examples)
            kind_correct[kind] += hits
            batch_total += len(kind_examples)
            batch_correct += hits
        loss = sum(loss_parts) / len(loss_parts)
        if not torch.isfinite(loss):
            print(
                f"skipped_nonfinite_loss kind_batch={sorted(grouped)} start={start} loss={float(loss.detach().cpu())}",
                flush=True,
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            continue
        if optimizer is not None:
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
        total_loss += float(loss.detach().item()) * batch_total
        total += batch_total
        correct += batch_correct
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{total_loss / max(total, 1):.4f}", acc=f"{correct / max(total, 1):.3f}", n=total)
    return {
        "examples": total,
        "loss": total_loss / total if total else 0.0,
        "acc": correct / total if total else 0.0,
        "kind_acc": {kind: kind_correct[kind] / kind_totals[kind] if kind_totals[kind] else 0.0 for kind in kind_totals},
        "kind_total": kind_totals,
    }


def save_checkpoint(path, model, args, spaces):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "args": vars(args),
        "label_spaces": spaces,
        "object_symbol_head": model.object_symbol_head.state_dict(),
        "symbol_pair_head": model.symbol_pair_head.state_dict(),
        "object_pair_head": model.object_pair_head.state_dict(),
    }
    if getattr(model, "lora_enabled", False):
        from peft import get_peft_model_state_dict

        state["backbone_lora"] = get_peft_model_state_dict(model.backbone)
    torch.save(state, path)
    return path


def load_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.object_symbol_head.load_state_dict(checkpoint["object_symbol_head"])
    model.symbol_pair_head.load_state_dict(checkpoint["symbol_pair_head"])
    model.object_pair_head.load_state_dict(checkpoint["object_pair_head"])
    if "backbone_lora" in checkpoint:
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(model.backbone, checkpoint["backbone_lora"])
    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train GraphQA CLEVR-style predicate heads.")
    parser.add_argument("--root", type=Path, default=DEFAULT_VQAR_ROOT)
    parser.add_argument("--task-path", type=Path, default=None)
    parser.add_argument("--kb-dir", type=Path, default=None)
    parser.add_argument("--no-kb", action="store_true")
    parser.add_argument("--kb-depth", type=int, default=2)
    parser.add_argument("--max-extra-kg-facts", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=0)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--grounding-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Train only Qwen visual grounding predicates: Name, Attribute, and object-object relations. "
            "KG/TypeOf facts remain deterministic inputs to the executable graph."
        ),
    )
    parser.add_argument("--save-every-epoch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-grounding-symbols", type=int, default=64)
    parser.add_argument("--max-grounding-object-pairs", type=int, default=128)
    parser.add_argument("--max-grounding-negatives-per-positive", type=float, default=None)
    parser.add_argument("--max-grounding-examples", type=int, default=None)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    task_path, instances, failures = load_instances(args)
    train_instances, dev_instances = split_instances(instances, args.dev_fraction, args.seed)
    train_examples = build_examples(train_instances, args)
    dev_examples = build_examples(dev_instances, args)
    spaces = grounding_label_spaces(instances) if args.grounding_only else label_spaces(instances)
    label_to_index = {kind: {label: index for index, label in enumerate(labels)} for kind, labels in spaces.items()}
    print(f"task_path={task_path}", flush=True)
    print(f"instances={len(instances)} train={len(train_instances)} dev={len(dev_instances)} failures={len(failures)}", flush=True)
    print(f"examples train={len(train_examples)} dev={len(dev_examples)} spaces={json.dumps(spaces, sort_keys=True)}", flush=True)
    if failures[:5]:
        print(f"first_failures={failures[:5]}", flush=True)

    lora_target_modules = [module.strip() for module in args.lora_target_modules.split(",") if module.strip()]
    model = GraphQAPredicateClassifier(
        model_path=args.model_path,
        object_symbol_labels=spaces["object_symbol"],
        symbol_pair_labels=spaces["symbol_pair"],
        object_pair_labels=spaces["object_pair"],
        device=args.device,
        freeze_backbone=args.freeze_backbone,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_target_modules,
        max_length=args.max_length,
    )


    if args.checkpoint is not None:
        loaded = load_checkpoint(args.checkpoint, model, args.device)
        print(f"loaded_checkpoint={args.checkpoint}", flush=True)
        if "args" in loaded:
            print(f"checkpoint_args={json.dumps(loaded['args'], sort_keys=True, default=str)}", flush=True)

    if args.eval_only:
        with torch.no_grad():
            eval_score = run_epoch(
                model,
                build_examples(instances, args),
                label_to_index,
                args,
                optimizer=None,
                epoch="eval_only",
            )
        print(f"eval={json.dumps(eval_score, sort_keys=True)}", flush=True)
        return 0

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    best = None
    for epoch in range(1, args.epochs + 1):
        train_score = run_epoch(model, train_examples, label_to_index, args, optimizer=optimizer, epoch=epoch)
        with torch.no_grad():
            dev_score = run_epoch(model, dev_examples, label_to_index, args, optimizer=None, epoch=epoch)
        print(f"epoch={epoch} train={json.dumps(train_score, sort_keys=True)}", flush=True)
        print(f"epoch={epoch} dev={json.dumps(dev_score, sort_keys=True)}", flush=True)
        current = dev_score["acc"]
        if args.save_every_epoch:
            epoch_path = args.output.with_name(f"{args.output.stem}_epoch{epoch}{args.output.suffix}")
            print(f"saved_epoch={save_checkpoint(epoch_path, model, args, spaces)}", flush=True)
        if best is None or current >= best:
            best = current
            print(f"saved={save_checkpoint(args.output, model, args, spaces)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
