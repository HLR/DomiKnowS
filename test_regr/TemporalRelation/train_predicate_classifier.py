import argparse
from collections import Counter
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .dataset import DEFAULT_TEMPORAL_DATA_ROOT, discover_temporal_datasets, load_temporal_instances
from .graph import TEMPORAL_LABELS, unpack_pair
from .modules import OracleTemporalPredicateClassifier, TemporalPredicateClassifier, predictions_from_logits


DEFAULT_OUTPUT = Path("/egr/research-hlr2/premsrit/TemporalRelation/models/temporal_predicate_heads.pt")


def progress_iter(iterable, args, desc):
    if not getattr(args, "progress", True):
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, dynamic_ncols=True, leave=True, mininterval=1.0)
    except Exception:
        return iterable


def default_matres_file(root):
    discovered = discover_temporal_datasets(root)
    for preferred in ("platinum.txt", "aquaint.txt", "timebank.txt"):
        for path in discovered["matres"]:
            path = Path(path)
            if path.name == preferred and path.is_file():
                return path
    for path in discovered["matres"]:
        path = Path(path)
        if path.is_file():
            return path
    raise FileNotFoundError(f"No MATRES file found under {root}")


def split_instances(instances, dev_fraction=0.2, seed=13):
    instances = list(instances)
    rng = random.Random(seed)
    rng.shuffle(instances)
    if not instances:
        return [], []
    dev_size = int(round(len(instances) * float(dev_fraction)))
    if len(instances) > 1:
        dev_size = min(max(dev_size, 1), len(instances) - 1)
    dev = instances[:dev_size]
    train = instances[dev_size:]
    return train, dev


def query_targets(instance, event_ids, device):
    query_pair = instance.get("query_pair") or (instance.get("event_pairs") or [{}])[0]
    query_e1, query_e2, _ = unpack_pair(query_pair)
    q1 = torch.tensor([1 if event_id == query_e1 else 0 for event_id in event_ids], dtype=torch.long, device=device)
    q2 = torch.tensor([1 if event_id == query_e2 else 0 for event_id in event_ids], dtype=torch.long, device=device)
    event = torch.ones(len(event_ids), dtype=torch.long, device=device)
    return event, q1, q2


def relation_targets(instance, pair_ids, device):
    labels_by_pair = {}
    for pair in instance.get("event_pairs", []):
        e1, e2, label = unpack_pair(pair)
        if label in TEMPORAL_LABELS:
            labels_by_pair[(e1, e2)] = TEMPORAL_LABELS.index(label)
    supervised_indices = []
    target_indices = []
    for index, pair_id in enumerate(pair_ids):
        if pair_id in labels_by_pair:
            supervised_indices.append(index)
            target_indices.append(labels_by_pair[pair_id])
    return (
        torch.tensor(supervised_indices, dtype=torch.long, device=device),
        torch.tensor(target_indices, dtype=torch.long, device=device),
    )


def predicate_loss(batch, instance, args, device):
    event_target, q1_target, q2_target = query_targets(instance, batch.event_ids, device)
    supervised_pair_indices, relation_target = relation_targets(instance, batch.pair_ids, device)

    losses = {}
    if args.event_loss_weight > 0 and len(event_target) > 0:
        losses["event"] = F.cross_entropy(batch.event_logits, event_target) * args.event_loss_weight
    if len(q1_target) > 0:
        losses["query_event1"] = F.cross_entropy(batch.query_event1_logits, q1_target) * args.query_loss_weight
        losses["query_event2"] = F.cross_entropy(batch.query_event2_logits, q2_target) * args.query_loss_weight
    if len(relation_target) > 0:
        relation_logits = batch.temporal_relation_logits.index_select(0, supervised_pair_indices)
        losses["temporal_relation"] = F.cross_entropy(relation_logits, relation_target) * args.relation_loss_weight

    if not losses:
        zero = batch.temporal_relation_logits.sum() * 0.0
        return zero, losses
    total = sum(losses.values())
    return total, losses


class MetricAccumulator:
    def __init__(self):
        self.loss = 0.0
        self.instances = 0
        self.query_event_correct = 0
        self.query_event_total = 0
        self.relation_correct = 0
        self.relation_total = 0
        self.relation_pred_counts = Counter()
        self.relation_gold_counts = Counter()

    def update(self, batch, instance, loss_value, device):
        self.instances += 1
        self.loss += float(loss_value)
        _event_target, q1_target, q2_target = query_targets(instance, batch.event_ids, device)
        if len(q1_target) > 0:
            q1_pred = torch.argmax(batch.query_event1_logits.detach(), dim=-1)
            q2_pred = torch.argmax(batch.query_event2_logits.detach(), dim=-1)
            self.query_event_correct += int((q1_pred == q1_target).sum().item())
            self.query_event_correct += int((q2_pred == q2_target).sum().item())
            self.query_event_total += int(q1_target.numel() + q2_target.numel())

        supervised_pair_indices, relation_target = relation_targets(instance, batch.pair_ids, device)
        if len(relation_target) > 0:
            relation_logits = batch.temporal_relation_logits.detach().index_select(0, supervised_pair_indices)
            relation_pred = torch.argmax(relation_logits, dim=-1)
            self.relation_correct += int((relation_pred == relation_target).sum().item())
            self.relation_total += int(relation_target.numel())
            self.relation_pred_counts.update(TEMPORAL_LABELS[int(index)] for index in relation_pred.detach().cpu().tolist())
            self.relation_gold_counts.update(TEMPORAL_LABELS[int(index)] for index in relation_target.detach().cpu().tolist())

    def summary(self):
        return {
            "instances": self.instances,
            "loss": self.loss / self.instances if self.instances else 0.0,
            "query_event_acc": self.query_event_correct / self.query_event_total if self.query_event_total else 0.0,
            "temporal_relation_acc": self.relation_correct / self.relation_total if self.relation_total else 0.0,
            "relation_total": self.relation_total,
            "relation_pred_counts": dict(self.relation_pred_counts),
            "relation_gold_counts": dict(self.relation_gold_counts),
        }


def run_epoch(model, instances, args, device, optimizer=None, epoch=None):
    training = optimizer is not None
    model.train(training)
    if (
        args.freeze_backbone
        and hasattr(model, "backbone")
        and not getattr(model, "backbone_has_trainable_parameters", lambda: False)()
    ):
        model.backbone.eval()
    metrics = MetricAccumulator()
    phase = "train" if training else "eval"
    desc = f"{phase} epoch={epoch}" if epoch is not None else phase

    iterator = progress_iter(instances, args, desc)
    for instance in iterator:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            batch = model(instance)
            loss, _losses = predicate_loss(batch, instance, args, device)
        if optimizer is not None:
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
        metrics.update(batch, instance, loss.detach().item(), device)
        if hasattr(iterator, "set_postfix"):
            summary = metrics.summary()
            iterator.set_postfix(
                loss=f"{summary['loss']:.4f}",
                q=f"{summary['query_event_acc']:.3f}",
                rel=f"{summary['temporal_relation_acc']:.3f}",
                rel_n=summary["relation_total"],
            )
    return metrics.summary()


def load_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.event_head.load_state_dict(checkpoint["event_head"])
    model.query_event1_head.load_state_dict(checkpoint["query_event1_head"])
    model.query_event2_head.load_state_dict(checkpoint["query_event2_head"])
    model.temporal_relation_head.load_state_dict(checkpoint["temporal_relation_head"])
    if "backbone_lora" in checkpoint:
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(model.backbone, checkpoint["backbone_lora"])
    return checkpoint


def save_checkpoint(path, model, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_path": args.model_path,
        "labels": TEMPORAL_LABELS,
        "freeze_backbone": args.freeze_backbone,
        "event_head": model.event_head.state_dict(),
        "query_event1_head": model.query_event1_head.state_dict(),
        "query_event2_head": model.query_event2_head.state_dict(),
        "temporal_relation_head": model.temporal_relation_head.state_dict(),
        "args": vars(args),
    }
    if getattr(model, "lora_enabled", False):
        from peft import get_peft_model_state_dict

        state["backbone_lora"] = get_peft_model_state_dict(model.backbone)
    if args.save_full_model:
        state["model_state_dict"] = model.state_dict()
    torch.save(state, path)
    return path


def load_instances(args):
    path = args.path or default_matres_file(args.root)
    instances = load_temporal_instances(path, limit=args.limit, group_by_document=True)
    if args.max_events is not None:
        instances = [instance for instance in instances if len(instance.get("events", [])) <= args.max_events]
    return path, instances


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLEVR-style TemporalRelation predicate classifier heads.")
    parser.add_argument("--root", type=Path, default=DEFAULT_TEMPORAL_DATA_ROOT)
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=None, help="Optional cap to avoid huge all-pair documents during smoke runs.")
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)

    parser.add_argument("--model-path", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--encode-batch-size", type=int, default=None)
    parser.add_argument(
        "--supervised-pairs-only",
        action="store_true",
        help="Train/evaluate temporal_relation only on labeled MATRES pairs instead of every ordered event pair.",
    )
    parser.add_argument(
        "--max-pairs-per-instance",
        type=int,
        default=None,
        help="Optional cap on event-pair prompts per document; useful for Qwen3-8B LoRA memory.",
    )
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=0, help="Enable LoRA on the backbone when > 0.")
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target module names for LoRA.",
    )
    parser.add_argument("--oracle", action="store_true", help="Use perfect predicate module; useful for fast smoke checks.")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--event-loss-weight", type=float, default=0.0)
    parser.add_argument("--query-loss-weight", type=float, default=1.0)
    parser.add_argument("--relation-loss-weight", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--save-full-model", action="store_true", help="Also save full backbone state; usually very large.")
    parser.add_argument("--show-predictions", type=int, default=0)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device

    path, instances = load_instances(args)
    train, dev = split_instances(instances, args.dev_fraction, args.seed)
    print(f"dataset={path}", flush=True)
    print(f"loaded={len(instances)} train={len(train)} dev={len(dev)} device={device}", flush=True)
    print(
        f"model_path={args.model_path} oracle={args.oracle} "
        f"freeze_backbone={args.freeze_backbone} lora_r={args.lora_r}",
        flush=True,
    )

    if args.oracle:
        model = OracleTemporalPredicateClassifier(device=device)
        train_score = run_epoch(model, train, args, device, optimizer=None)
        dev_score = run_epoch(model, dev, args, device, optimizer=None)
        print("oracle_train", json.dumps(train_score, sort_keys=True), flush=True)
        print("oracle_dev", json.dumps(dev_score, sort_keys=True), flush=True)
        if args.show_predictions and instances:
            print(json.dumps(predictions_from_logits(model(instances[0])), indent=2))
        return 0

    lora_target_modules = [
        module.strip()
        for module in str(args.lora_target_modules).split(",")
        if module.strip()
    ]
    model = TemporalPredicateClassifier(
        model_path=args.model_path,
        device=device,
        freeze_backbone=args.freeze_backbone,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_target_modules,
        max_length=args.max_length,
        encode_batch_size=args.encode_batch_size,
        supervised_pairs_only=args.supervised_pairs_only,
        max_pairs_per_instance=args.max_pairs_per_instance,
    )
    if args.checkpoint is not None:
        loaded = load_checkpoint(args.checkpoint, model, device)
        print(f"loaded_checkpoint={args.checkpoint}", flush=True)
        if "args" in loaded:
            print(f"checkpoint_args={json.dumps(loaded['args'], sort_keys=True, default=str)}", flush=True)

    if args.eval_only:
        with torch.no_grad():
            eval_score = run_epoch(model, instances, args, device, optimizer=None, epoch="eval_only")
        print(f"eval={json.dumps(eval_score, sort_keys=True)}", flush=True)
        if args.show_predictions and instances:
            with torch.no_grad():
                print(json.dumps(predictions_from_logits(model(instances[0])), indent=2))
        return 0

    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_dev = None
    for epoch in range(1, args.epochs + 1):
        train_score = run_epoch(model, train, args, device, optimizer=optimizer, epoch=epoch)
        with torch.no_grad():
            dev_score = run_epoch(model, dev, args, device, optimizer=None, epoch=epoch)
        print(f"epoch={epoch} train={json.dumps(train_score, sort_keys=True)}", flush=True)
        print(f"epoch={epoch} dev={json.dumps(dev_score, sort_keys=True)}", flush=True)
        current = dev_score["temporal_relation_acc"]
        if best_dev is None or current >= best_dev:
            best_dev = current
            saved = save_checkpoint(args.output, model, args)
            print(f"saved={saved}", flush=True)

    if args.show_predictions and instances:
        with torch.no_grad():
            print(json.dumps(predictions_from_logits(model(instances[0])), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
