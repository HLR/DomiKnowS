"""Train GraphQA Scallop-5.2-style MLP predicates through DomiKnowS execution.

The perception model is M_theta=(M_name, M_attr, M_relation): MLPs over object
feature vectors/bounding boxes and object-pair features.  KG facts remain
symbolic, and the executable miotaL query supplies the answer-set loss.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from tqdm import tqdm

from domiknows.utils import setProductionLogMode

from .graph import canonical_relation
from .evaluate_object_centered_c2 import miota_prediction_records
from .object_centered_pipeline import (
    build_program,
    load_instances,
    _object_feature_dim,
    DynamicActiveConceptsDataset,
)


def split_instances(instances, dev_fraction=0.1, seed=13):
    instances = list(instances)
    random.Random(seed).shuffle(instances)
    if len(instances) <= 1 or dev_fraction <= 0:
        return instances, []
    dev_size = min(max(1, int(round(len(instances) * dev_fraction))), len(instances) - 1)
    return instances[dev_size:], instances[:dev_size]


def vocab_options_for_instances(instances, gqa_info_path):
    with Path(gqa_info_path).open() as stream:
        info = json.load(stream)
    name_indices = {str(value): int(index) for value, index in info["name"]["idx"].items()}
    attr_indices = {str(value): int(index) for value, index in info["attr"]["idx"].items()}
    name_indices.update({
        value.replace(" ", "_"): index for value, index in list(name_indices.items())
    })
    attr_indices.update({
        value.replace(" ", "_"): index for value, index in list(attr_indices.items())
    })
    relation_indices = {}
    for value, index in info["rel"]["idx"].items():
        relation_indices[str(value)] = int(index)
        relation_indices[canonical_relation(value)] = int(index)
    return {
        "scallop_mlp_name_indices": name_indices,
        "scallop_mlp_attr_indices": attr_indices,
        "scallop_mlp_relation_indices": relation_indices,
        "scallop_mlp_name_classes": int(info["name"]["num"]),
        "scallop_mlp_attr_classes": int(info["attr"]["num"]),
        "scallop_mlp_relation_classes": int(info["rel"]["num"]),
        "scallop_mlp_feature_dim": _object_feature_dim(instances),
    }


def evaluate_dev(instances, dataset, program, device, threshold, decode_policy):
    eval_start = time.perf_counter()
    program.model.eval()
    records = miota_prediction_records(
        instances, dataset, program, device,
        threshold=threshold,
        decode_policy=decode_policy,
        show_progress=True,
    )
    wall_seconds = time.perf_counter() - eval_start
    correct = sum(int(record["correct"]) for record in records)
    # Same fix as evaluate_scallop_mlp_c2c6.py: denominator is min(len(gold), k),
    # not len(gold) -- some instances have many valid answers, and dividing by
    # the full gold count caps recall near k/len(gold) even for a perfect
    # top-k. Empty-gold rows are excluded from the recall average (tracked via
    # recall_counts) rather than folded in via max(1, len(gold)), which
    # silently scored them as 0 and deflated the mean.
    recall_sums = {5: 0.0, 10: 0.0}
    recall_counts = {5: 0, 10: 0}
    top1_hits = 0
    for record in records:
        ranked = [
            object_id
            for _score, object_id in sorted(
                zip(record["scores"], record["objects"]), reverse=True
            )
        ]
        gold = set(record["gold_answers"])
        if ranked and ranked[0] in gold:
            top1_hits += 1
        if gold:
            for k in recall_sums:
                recall_sums[k] += len(gold.intersection(ranked[:k])) / min(len(gold), k)
                recall_counts[k] += 1
    examples = len(records)
    return {
        "examples": examples,
        "exact_answer_acc": correct / examples if records else None,
        "top1_gold_hit": top1_hits / examples if records else None,
        "recall_at_5": recall_sums[5] / recall_counts[5] if recall_counts[5] else None,
        "recall_at_10": recall_sums[10] / recall_counts[10] if recall_counts[10] else None,
        "wall_seconds": wall_seconds,
        "seconds_per_example": wall_seconds / examples if examples else None,
    }


def train_epoch(program, dataset, optimizer, device, epoch, lr, batch_size,
                 context=None, raw_instances=None, dynamic_active_concepts=False):
    program.opt = optimizer
    examples = len(dataset)
    if dynamic_active_concepts:
        if context is None or raw_instances is None:
            raise ValueError(
                "--dynamic-active-concepts requires context and raw_instances"
            )
        dataset = DynamicActiveConceptsDataset(context, raw_instances, dataset)
    train_start = time.perf_counter()
    program.train(
        dataset,
        warmup_epochs=0,
        constraint_epochs=1,
        device=device,
        c_lr=lr,
        batch_size=batch_size,
    )
    wall_seconds = time.perf_counter() - train_start
    return {
        "examples": examples,
        "domiknows_train_epoch": epoch,
        "wall_seconds": wall_seconds,
        "seconds_per_example": wall_seconds / examples if examples else None,
    }



def _find_scallop_mlp_modules(program):
    modules = {"name_mlp": None, "attr_mlp": None, "relation_mlp": None}
    object_mlps = []
    for _name, module in program.model.named_modules():
        cls = type(module).__name__
        if cls == "ScallopObjectMLP" and module not in object_mlps:
            object_mlps.append(module)
        elif cls == "ScallopRelationMLP":
            modules["relation_mlp"] = module
    if object_mlps:
        modules["name_mlp"] = object_mlps[0]
    if len(object_mlps) > 1:
        modules["attr_mlp"] = object_mlps[1]
    return modules


def save_checkpoint(
    path, program, context=None, args=None, vocab_options=None, optimizer=None
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    modules = getattr(context, "scallop_mlp_modules", None) or _find_scallop_mlp_modules(program)
    vocab_options = dict(vocab_options or {})
    state = {
        "architecture_version": "vqar_scallop_5_2",
        "args": vars(args) if args is not None else {},
        "indices": {
            "name": dict(vocab_options.get("scallop_mlp_name_indices") or {}),
            "attribute": dict(vocab_options.get("scallop_mlp_attr_indices") or {}),
            "relation": dict(vocab_options.get("scallop_mlp_relation_indices") or {}),
        },
        "num_classes": {
            "name": int(vocab_options.get("scallop_mlp_name_classes", 0)),
            "attribute": int(vocab_options.get("scallop_mlp_attr_classes", 0)),
            "relation": int(vocab_options.get("scallop_mlp_relation_classes", 0)),
        },
        "feature_dim": int(vocab_options.get("scallop_mlp_feature_dim", 0) or 0),
        "hidden_dim": int(getattr(args, "hidden_dim", 1024) if args is not None else 1024),
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    for key in ("name_mlp", "attr_mlp", "relation_mlp"):
        module = modules.get(key) if isinstance(modules, dict) else None
        if module is not None:
            state[key] = module.state_dict()
            first = next(module.parameters(), None)
            if first is not None and not state["feature_dim"]:
                state["feature_dim"] = int(first.shape[1])
    torch.save(state, path)
    return path


def load_released_predicates(predicate_dir, modules):
    """Initialize architecture-compatible heads from released VQAR weights."""

    predicate_dir = Path(predicate_dir)
    files = {
        "name_mlp": predicate_dir / "name_best_epoch.pt",
        "attr_mlp": predicate_dir / "attribute_best_epoch.pt",
        "relation_mlp": predicate_dir / "relation_best_epoch.pt",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing released predicate checkpoints: {missing}")
    for key, path in files.items():
        modules[key].load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-path", type=Path, required=True)
    parser.add_argument("--dev-task-path", type=Path)
    parser.add_argument("--kb-dir", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, default=Path("/egr/research-hlr2/premsrit/VQAR_data/image_cache"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dev-limit", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--gqa-info", type=Path, default=Path("/egr/research-hlr2/premsrit/VQAR_data/data/gqa_info.json"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--scheduler-step", type=int, default=10)
    parser.add_argument("--scheduler-gamma", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-predicate-dir", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--decode-policy", choices=["threshold", "top1", "family-top1"], default="threshold")
    parser.add_argument(
        "--no-global-consistency",
        action="store_true",
        help="Train from executable query loss without KB implication constraints.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Weight applied by InferenceProgram to its compiled constraint loss.",
    )
    parser.add_argument(
        "--execution-loss-weight",
        type=float,
        default=1.0,
        help="Weight of executable query loss inside the combined constraint loss.",
    )
    parser.add_argument(
        "--global-loss-weight",
        type=float,
        default=1.0,
        help="Weight of graph-global consistency loss inside the combined constraint loss.",
    )
    parser.add_argument(
        "--compile-lc",
        action="store_true",
        help=(
            "Use DomiKnowS's compiled (batched-gather) constraint evaluator for the "
            "graph-global constraint loss instead of the default per-constraint "
            "interpreter. Same t-norm math (verified exact-match against the "
            "interpreter on real global-consistency graphs), falls back to the "
            "interpreter per-constraint for unsupported types -- just faster."
        ),
    )
    parser.add_argument(
        "--average-global-constraint-loss",
        action="store_true",
        help=(
            "Average (not sum) the global constraint loss, both within a "
            "constraint's own groundings and across constraint keys. "
            "Measured directly: the raw sum is ~430x larger than "
            "executable_loss in aggregate (up to 75000x per-step) because it "
            "sums unnormalized over every grounded constraint, most "
            "irrelevant to any given instance. Averaging makes global_loss "
            "scale-invariant to graph/instance size, addressing the "
            "imbalance at its source instead of via a fixed --global-loss-"
            "weight scalar that has to be re-tuned by hand and drifts as "
            "training changes how many constraints have nonzero loss."
        ),
    )
    parser.add_argument(
        "--enable-kb-predictor",
        action="store_true",
        help=(
            "Pilot: give a small set of high-coverage KB-hierarchy concepts "
            "(e.g. 'animal') their own learnable predictor instead of the "
            "default deterministic derivation from the Name prediction, so "
            "the global-consistency ifL constraint becomes a real loss "
            "between two independently-parameterized probabilities."
        ),
    )
    parser.add_argument(
        "--kb-predictor-top-k",
        type=int,
        default=6,
        help="Number of highest-coverage KB-hierarchy concepts to give a learnable predictor when --enable-kb-predictor is set.",
    )
    parser.add_argument(
        "--dynamic-active-concepts",
        action="store_true",
        help=(
            "Narrow the graph's active concepts to just each training "
            "instance's own referenced predicates (mirrors evaluate_object_"
            "centered_c2.py's eval-only dynamic-active-concepts mode, applied "
            "to training). Skips compiled/executable constraint evaluation "
            "for concepts irrelevant to that instance, and -- since "
            "_calculate_global_constraint_loss sums unnormalized over every "
            "grounded constraint -- shrinks the raw global-loss magnitude at "
            "its source rather than only scaling it down with a fixed "
            "--global-loss-weight. Off by default; does not change behavior "
            "for runs that don't pass this flag."
        ),
    )
    parser.add_argument(
        "--select-by",
        choices=["exact_answer_acc", "recall_at_5"],
        default="exact_answer_acc",
        help=(
            "Dev metric used to pick the 'best' checkpoint saved to --output. "
            "Defaults to exact_answer_acc (original behavior, unchanged for "
            "runs that don't pass this flag). Pass recall_at_5 to select/"
            "early-stop on Recall@5 instead."
        ),
    )
    parser.add_argument(
        "--max-objects-per-instance",
        type=int,
        default=None,
        help=(
            "Drop instances with more objects than this. create_object_centered_graph "
            "sizes answer_slots (and other per-instance DomiKnowS candidate/constraint "
            "work) to the max object count across the whole loaded batch, so a few "
            "outlier scenes slow down every instance's constraint evaluation, not just "
            "their own. Capping the tail keeps that shared bound small."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setProductionLogMode()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    instances, failures = load_instances(
        args.task_path, args.kb_dir, args.limit,
        single_answer_only=False, offset=args.offset,
        max_objects=args.max_objects_per_instance,
        image_cache=args.image_cache,
    )
    if args.dev_task_path:
        train = instances
        dev, dev_failures = load_instances(
            args.dev_task_path, args.kb_dir, args.dev_limit, single_answer_only=False,
            max_objects=args.max_objects_per_instance,
            image_cache=args.image_cache,
        )
        failures.extend(dev_failures)
    else:
        train, dev = split_instances(instances, args.dev_fraction, args.seed)
    qwen_options = {
        "scallop_mlp_hidden_dim": args.hidden_dim,
        "enable_kb_predictor": args.enable_kb_predictor,
        "kb_dir": args.kb_dir,
        "kb_predictor_top_k": args.kb_predictor_top_k,
        **vocab_options_for_instances(train + dev, args.gqa_info),
    }
    if args.init_checkpoint and args.init_predicate_dir:
        raise ValueError("--init-checkpoint and --init-predicate-dir are mutually exclusive")
    if args.init_checkpoint:
        qwen_options["scallop_mlp_checkpoint"] = str(args.init_checkpoint)
    combined = train + dev
    context, compiled_dataset, program = build_program(
        combined,
        mode="scallop-mlp",
        image_cache=args.image_cache,
        device=args.device,
        answer_mode="miota",
        qwen_options=qwen_options,
        include_global_consistency=not args.no_global_consistency,
        beta=args.beta,
        executable_constraint_loss_weight=args.execution_loss_weight,
        global_constraint_loss_weight=args.global_loss_weight,
        compile_lc=args.compile_lc,
        average_global_constraint_loss=args.average_global_constraint_loss,
    )
    train_dataset = [compiled_dataset[index] for index in range(len(train))]
    dev_dataset = [
        compiled_dataset[index]
        for index in range(len(train), len(compiled_dataset))
    ]
    program.to(args.device)
    modules = context.scallop_mlp_modules
    if args.init_predicate_dir:
        load_released_predicates(args.init_predicate_dir, modules)
    parameter_groups = [
        {"params": list(modules[key].parameters()), "lr": args.lr}
        for key in ("name_mlp", "attr_mlp", "relation_mlp")
    ]
    if not any(group["params"] for group in parameter_groups):
        raise ValueError("No trainable Scallop MLP parameters found")
    optimizer = torch.optim.Adam(parameter_groups)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.scheduler_step,
        gamma=args.scheduler_gamma,
    )
    if args.resume_optimizer:
        if not args.init_checkpoint:
            raise ValueError("--resume-optimizer requires --init-checkpoint")
        init_state = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        if "optimizer_state" not in init_state:
            raise ValueError(
                f"Checkpoint {args.init_checkpoint} has no optimizer state"
            )
        optimizer.load_state_dict(init_state["optimizer_state"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
    module_params = {
        key: (sum(param.numel() for param in module.parameters()) if module is not None else 0)
        for key, module in modules.items()
        if key in {"name_mlp", "attr_mlp", "relation_mlp"}
    }
    print(json.dumps({
        "loaded": len(instances),
        "train": len(train),
        "dev": len(dev),
        "failures": len(failures),
        "device": args.device,
        "feature_dim": modules["feature_dim"],
        "num_classes": modules["num_classes"],
        "module_params": module_params,
        "global_consistency": not args.no_global_consistency,
        "beta": args.beta,
        "execution_loss_weight": args.execution_loss_weight,
        "global_loss_weight": args.global_loss_weight,
    }), flush=True)
    if args.eval_only:
        dev_score = evaluate_dev(
            dev, dev_dataset, program, args.device,
            args.prediction_threshold, args.decode_policy,
        )
        print(f"eval={json.dumps(dev_score, sort_keys=True)}", flush=True)
        save_checkpoint(args.output, program, context=context, args=args, vocab_options=qwen_options)
        return 0
    select_key = args.select_by
    best_dev = -1.0
    for epoch in range(1, args.epochs + 1):
        train_score = train_epoch(
            program, train_dataset, optimizer, args.device, epoch, args.lr, args.batch_size,
            context=context, raw_instances=train,
            dynamic_active_concepts=args.dynamic_active_concepts,
        )
        scheduler.step()
        dev_score = (
            evaluate_dev(
                dev, dev_dataset, program, args.device,
                args.prediction_threshold, args.decode_policy,
            )
            if dev_dataset else None
        )
        ckpt = args.output.with_name(f"{args.output.stem}_epoch{epoch}{args.output.suffix}")
        saved = save_checkpoint(
            ckpt, program, context=context, args=args,
            vocab_options=qwen_options, optimizer=optimizer,
        )
        if dev_score is None or dev_score[select_key] > best_dev:
            best_dev = dev_score[select_key] if dev_score is not None else 0.0
            save_checkpoint(
                args.output, program, context=context, args=args,
                vocab_options=qwen_options, optimizer=optimizer,
            )
        print(
            f"epoch={epoch} train={json.dumps(train_score, sort_keys=True)} "
            f"dev={json.dumps(dev_score, sort_keys=True)} saved_epoch={saved}",
            flush=True,
        )
    print(f"saved_best={args.output} best_dev_{select_key}={best_dev}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
