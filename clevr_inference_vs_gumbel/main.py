from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn

RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
for path in (RUN_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from domiknows import setProductionLogMode

setProductionLogMode(True)

from domiknows.graph import Concept, Graph, Relation
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor import Sensor
from domiknows.sensor.pytorch import EdgeSensor, ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.sensors import FunctionalReaderSensor, FunctionalSensor, ReaderSensor

from clevr_constraints import g_attribute_concepts, g_relational_concepts, prepare_logic_fields
from graph import create_graph


@dataclass
class BuiltProgram:
    name: str
    program: Any
    train_dataset: Any
    eval_dataset: Any


@dataclass
class EvalStats:
    accuracy: float
    constraint_loss: float
    executable_loss: float
    global_loss: float
    gumbel_sample_accuracy: float | None = None


class SoftmaxClassifier(nn.Module):
    """Small trainable classifier returning two-class probabilities."""

    def __init__(self, input_size: int, hidden_size: int = 24):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.layers(features.float()), dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare deterministic InferenceProgram with InferenceProgram(use_gumbel=True)."
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--train-items",
        type=int,
        default=None,
        help="Number of examples to train on. Defaults to 80%% of the compact dataset.",
    )
    parser.add_argument(
        "--eval-items",
        type=int,
        default=None,
        help="Number of examples to evaluate on. Defaults to the remaining compact dataset.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"])
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--tnorm", default="G")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--global-constraint-loss-weight", type=float, default=0.1)
    parser.add_argument("--executable-constraint-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--disable-global-constraint-loss",
        action="store_true",
        help="Train only executable query constraints and ignore graph-global visual constraints.",
    )
    parser.add_argument("--gumbel-temp-start", type=float, default=1.0)
    parser.add_argument("--gumbel-temp-end", type=float, default=0.3)
    parser.add_argument("--hard-gumbel", action="store_true")
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return "cuda:0"
    return device


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_items() -> list[dict[str, Any]]:
    with (RUN_DIR / "data" / "clevr_20_programs.json").open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return [_make_data_item(item) for item in raw["items"]]


def _make_data_item(item: dict[str, Any]) -> dict[str, Any]:
    scene = item["scene"]
    objects = scene["objects"]
    relationships = scene.get("relationships", {})
    object_features = _object_features(objects)
    pair_features = _pair_features(objects, relationships)
    return {
        "question": item["question"],
        "question_raw": item["question"],
        "answer": item["answer"],
        "program": item["program"],
        "image_id": item["image_index"],
        "image_index": item["image_index"],
        "image_filename": item.get("image_filename"),
        "pil_image": None,
        "all_objects": objects,
        "objects_raw": _dummy_boxes(len(objects)),
        "object_features": object_features,
        "pair_features": pair_features,
        "pair_rev_features": pair_features,
    }


def _dummy_boxes(count: int) -> list[list[float]]:
    return [[float(i), 0.0, 1.0, 1.0] for i in range(count)]


def _object_features(objects: list[dict[str, Any]]) -> list[list[float]]:
    values = [value for group in g_attribute_concepts.values() for value in group]
    features = []
    for obj in objects:
        row = []
        for value in values:
            row.append(1.0 if value in obj.values() else 0.0)
        features.append(row)
    return features


def _pair_features(objects: list[dict[str, Any]], relationships: dict[str, list[list[int]]]) -> list[list[float]]:
    relation_names = g_relational_concepts["spatial_relation"]
    same_attrs = list(g_attribute_concepts.keys())
    count = len(objects)
    relation_mats = {name: np.zeros((count, count), dtype=np.float32) for name in relation_names}
    for name in relation_names:
        rel_rows = relationships.get(name, [])
        for obj_i, related_indices in enumerate(rel_rows):
            for obj_j in related_indices:
                if 0 <= obj_i < count and 0 <= obj_j < count:
                    relation_mats[name][obj_j, obj_i] = 1.0

    features = []
    for row_idx in range(count):
        for col_idx in range(count):
            row = [float(relation_mats[name][row_idx, col_idx]) for name in relation_names]
            for attr in same_attrs:
                row.append(float(row_idx != col_idx and objects[row_idx].get(attr) == objects[col_idx].get(attr)))
            features.append(row)
    return features


def filter_relation(property=None, arg1=None, arg2=None, **kwargs) -> bool:
    del property
    if arg1 is not None and arg2 is not None:
        return arg1.getAttribute("image_id") == arg2.getAttribute("image_id")
    values = [v for v in kwargs.values() if v is not None and hasattr(v, "getAttribute")]
    return len(values) < 2 or values[0].getAttribute("image_id") == values[1].getAttribute("image_id")


def build_program(
    name: str,
    program_cls: type,
    items: list[dict[str, Any]],
    args: argparse.Namespace,
    device: str,
    *,
    use_gumbel: bool = False,
) -> BuiltProgram:
    set_seed(args.seed)
    Graph.clear()
    Concept.clear()
    Relation.clear()
    Sensor.clear()

    dataset = copy.deepcopy(items)
    results = create_graph(
        dataset,
        include_query_questions=True,
        relation_syntax="legacy",
    )
    executions, graph, image, objects, image_object_contains = results[:5]
    obj1, obj2, pair_forward, attribute_names_dict = results[5:9]
    query_types = results[9]
    obj1_rev = results[10] if len(results) > 10 else None
    obj2_rev = results[11] if len(results) > 11 else None
    pair_reverse = results[12] if len(results) > 12 else None

    prepare_logic_fields(
        dataset,
        device=device,
        executions=executions,
        query_types=query_types,
        pin_query_answers=False,
    )
    _connect_sensors(
        image=image,
        objects=objects,
        image_object_contains=image_object_contains,
        obj1=obj1,
        obj2=obj2,
        pair_forward=pair_forward,
        obj1_rev=obj1_rev,
        obj2_rev=obj2_rev,
        pair_reverse=pair_reverse,
        attribute_names_dict=attribute_names_dict,
        device=device,
    )
    graph.constraint["label"] = ReaderSensor(keyword="logic_label", label=True)
    compiled = graph.compile_executable(
        dataset,
        logic_keyword="logic_str",
        logic_label_keyword="logic_label",
        extra_namespace_values=attribute_names_dict,
    )

    if args.train_items is None:
        train_count = max(1, int(round(len(compiled) * 0.8)))
        train_count = min(train_count, max(0, len(compiled) - 1))
    else:
        train_count = max(0, min(args.train_items, len(compiled)))

    remaining = max(0, len(compiled) - train_count)
    if args.eval_items is None:
        eval_count = remaining
    else:
        eval_count = max(0, min(args.eval_items, remaining))
    train_dataset = [compiled[i] for i in range(train_count)]
    eval_dataset = [
        compiled[i]
        for i in range(train_count, train_count + eval_count)
    ] or [compiled[i] for i in range(min(args.eval_items or 1, len(compiled)))]

    poi = [image, objects, *attribute_names_dict.values(), graph.constraint, pair_forward]
    if pair_reverse is not None:
        poi.append(pair_reverse)

    disable_global_constraint_loss = getattr(args, "disable_global_constraint_loss", False)
    global_constraint_loss_weight = getattr(args, "global_constraint_loss_weight", 0.1)
    executable_constraint_loss_weight = getattr(args, "executable_constraint_loss_weight", 1.0)

    program_kwargs = {
        "loss": NBCrossEntropyLoss,
        "poi": poi,
        "device": device,
        "tnorm": args.tnorm,
        "inferTypes": ["local/softmax", "local/argmax"],
        "include_global_constraint_loss": not disable_global_constraint_loss,
        "global_constraint_loss_weight": global_constraint_loss_weight,
        "executable_constraint_loss_weight": executable_constraint_loss_weight,
    }
    if use_gumbel:
        program_kwargs.update({
            "use_gumbel": True,
            "initial_temp": args.gumbel_temp_start,
            "final_temp": args.gumbel_temp_end,
            "anneal_epochs": args.epochs,
            "hard_gumbel": args.hard_gumbel,
        })

    program = program_cls(graph, SolverModel, **program_kwargs)
    return BuiltProgram(name=name, program=program, train_dataset=train_dataset, eval_dataset=eval_dataset)


def _connect_sensors(
    *,
    image,
    objects,
    image_object_contains,
    obj1,
    obj2,
    pair_forward,
    obj1_rev,
    obj2_rev,
    pair_reverse,
    attribute_names_dict,
    device: str,
) -> None:
    object_feature_size = sum(len(values) for values in g_attribute_concepts.values())
    pair_feature_size = len(g_relational_concepts["spatial_relation"]) + len(g_attribute_concepts)

    image["pil_image"] = FunctionalReaderSensor(keyword="pil_image", forward=lambda data: data)
    image["image_id"] = FunctionalReaderSensor(keyword="image_id", forward=lambda data: [data])
    objects["bounding_boxes"] = FunctionalReaderSensor(
        keyword="objects_raw",
        forward=lambda data: torch.tensor(data, dtype=torch.float32, device=device),
    )
    objects["features"] = FunctionalReaderSensor(
        keyword="object_features",
        forward=lambda data: torch.tensor(data, dtype=torch.float32, device=device),
    )
    objects["image_id"] = FunctionalSensor(
        image["image_id"],
        "bounding_boxes",
        forward=lambda image_id, boxes: image_id * len(boxes),
    )
    objects[image_object_contains] = EdgeSensor(
        objects["bounding_boxes"],
        image["pil_image"],
        relation=image_object_contains,
        forward=lambda boxes, _image: torch.ones(len(boxes), 1),
    )

    pair_forward[obj1.reversed, obj2.reversed] = CompositionCandidateSensor(
        objects["image_id"],
        relations=(obj1.reversed, obj2.reversed),
        forward=filter_relation,
    )
    pair_forward["features"] = FunctionalReaderSensor(
        keyword="pair_features",
        forward=lambda data: torch.tensor(data, dtype=torch.float32, device=device),
    )
    if pair_reverse is not None and obj1_rev is not None and obj2_rev is not None:
        pair_reverse[obj1_rev.reversed, obj2_rev.reversed] = CompositionCandidateSensor(
            objects["image_id"],
            relations=(obj1_rev.reversed, obj2_rev.reversed),
            forward=filter_relation,
        )
        pair_reverse["features"] = FunctionalReaderSensor(
            keyword="pair_rev_features",
            forward=lambda data: torch.tensor(data, dtype=torch.float32, device=device),
        )

    spatial = set(g_relational_concepts["spatial_relation"])
    spatial_rev = {f"{name}_rev" for name in spatial}
    for attr_name, attr_variable in attribute_names_dict.items():
        if attr_name in spatial:
            pair_forward[attr_variable] = ModuleLearner(
                "features",
                module=SoftmaxClassifier(pair_feature_size).to(device),
                device=device,
            )
        elif attr_name in spatial_rev and pair_reverse is not None:
            pair_reverse[attr_variable] = ModuleLearner(
                "features",
                module=SoftmaxClassifier(pair_feature_size).to(device),
                device=device,
            )
        elif attr_name.startswith("same_"):
            pair_forward[attr_variable] = ModuleLearner(
                "features",
                module=SoftmaxClassifier(pair_feature_size).to(device),
                device=device,
            )
        else:
            objects[attr_variable] = ModuleLearner(
                "features",
                module=SoftmaxClassifier(object_feature_size).to(device),
                device=device,
            )


def _constraint_forward(program: Any, builder: Any) -> torch.Tensor:
    if getattr(program, "use_gumbel", False) and hasattr(program, "_call_cmodel_with_gumbel"):
        loss, *_ = program._call_cmodel_with_gumbel(builder)
    else:
        loss, *_ = program.cmodel(builder)
    return loss


def average_constraint_losses(built: BuiltProgram) -> tuple[float, float, float]:
    losses = []
    executable_losses = []
    global_losses = []
    built.program.model.eval()
    built.program.cmodel.eval()
    with torch.no_grad():
        for data in built.eval_dataset:
            _mloss, _metric, _datanode, builder = built.program.model(data)
            loss = _constraint_forward(built.program, builder)
            if torch.is_tensor(loss):
                losses.append(float(loss.detach().cpu()))
            executable_loss = getattr(built.program.cmodel, "last_executable_loss", None)
            if torch.is_tensor(executable_loss):
                executable_losses.append(float(executable_loss.detach().cpu()))
            global_loss = getattr(built.program.cmodel, "last_global_loss", None)
            if torch.is_tensor(global_loss):
                global_losses.append(float(global_loss.detach().cpu()))
    return (
        float(np.mean(losses)) if losses else float("nan"),
        float(np.mean(executable_losses)) if executable_losses else float("nan"),
        float(np.mean(global_losses)) if global_losses else float("nan"),
    )


def evaluate(built: BuiltProgram, device: str) -> EvalStats:
    print(f"{built.name} deterministic evaluation:")
    raw_accuracy = built.program.evaluate_condition(built.eval_dataset, device=device)
    if isinstance(raw_accuracy, dict):
        accuracy = float(raw_accuracy.get("accuracy", raw_accuracy.get("satisfaction", 0.0)))
    else:
        accuracy = float(raw_accuracy)

    gumbel_sample_accuracy = None
    if getattr(built.program, "use_gumbel", False):
        print(f"{built.name} sampled Gumbel evaluation:")
        sampled_accuracy = built.program.evaluate_condition(
            built.eval_dataset,
            device=device,
            use_gumbel=True,
            temperature=getattr(built.program, "current_temp", None),
            hard_gumbel=getattr(built.program, "hard_gumbel", False),
        )
        if isinstance(sampled_accuracy, dict):
            gumbel_sample_accuracy = float(
                sampled_accuracy.get("accuracy", sampled_accuracy.get("satisfaction", 0.0))
            )
        else:
            gumbel_sample_accuracy = float(sampled_accuracy)

    constraint_loss, executable_loss, global_loss = average_constraint_losses(built)
    return EvalStats(
        accuracy=accuracy,
        constraint_loss=constraint_loss,
        executable_loss=executable_loss,
        global_loss=global_loss,
        gumbel_sample_accuracy=gumbel_sample_accuracy,
    )


def gradient_report(built: BuiltProgram) -> None:
    program = built.program
    loss = None
    grad_rows = []
    total_sq = 0.0
    max_abs = 0.0
    for data in built.train_dataset:
        for opt in (getattr(program, "opt", None), getattr(program, "copt", None)):
            if opt is not None:
                opt.zero_grad()
        program.model.zero_grad(set_to_none=True)
        program.cmodel.zero_grad(set_to_none=True)

        _mloss, _metric, _datanode, builder = program.model(data)
        loss = _constraint_forward(program, builder)
        if torch.is_tensor(loss) and loss.requires_grad:
            loss.backward()

        grad_rows = []
        total_sq = 0.0
        max_abs = 0.0
        for name, param in program.model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            norm = float(grad.norm().cpu())
            total_sq += norm * norm
            max_abs = max(max_abs, float(grad.abs().max().cpu()))
            grad_rows.append((name, norm))
        grad_rows.sort(key=lambda row: row[1], reverse=True)
        if grad_rows and total_sq > 0:
            break

    loss_value = float(loss.detach().cpu()) if torch.is_tensor(loss) else float("nan")
    print(
        f"{built.name} gradient: loss={loss_value:.4f}, "
        f"l2={total_sq ** 0.5:.4f}, max={max_abs:.4f}, params={len(grad_rows)}"
    )
    for name, norm in grad_rows[:3]:
        print(f"  {name}: {norm:.4f}")

    program.model.zero_grad(set_to_none=True)
    program.cmodel.zero_grad(set_to_none=True)


def train_program(built: BuiltProgram, args: argparse.Namespace, device: str) -> None:
    optimizer_factory = lambda params: torch.optim.Adam(params, lr=args.lr)
    built.program.train(
        built.train_dataset,
        train_epoch_num=args.epochs,
        Optim=optimizer_factory,
        c_lr=args.lr,
        device=device,
    )


def print_comparison(before: dict[str, EvalStats], after: dict[str, EvalStats]) -> None:
    print("\nComparison")
    for name in after:
        delta = after[name].accuracy - before[name].accuracy
        sampled_part = ""
        if after[name].gumbel_sample_accuracy is not None:
            sampled_part = f", sampled Gumbel accuracy {after[name].gumbel_sample_accuracy:.2f}"
        print(
            f"{name}: accuracy {before[name].accuracy:.2f} -> {after[name].accuracy:.2f} "
            f"(delta {delta:+.2f}), constraint loss {after[name].constraint_loss:.4f}"
            f" (exec={after[name].executable_loss:.4f}, global={after[name].global_loss:.4f})"
            f"{sampled_part}"
        )
    names = list(after)
    best = max(
        names,
        key=lambda name: (
            after[name].accuracy - before[name].accuracy,
            after[name].accuracy,
            -after[name].constraint_loss,
        ),
    )
    print(
        f"Winner: {best}. It had the best accuracy improvement/final accuracy tradeoff; "
        "constraint loss is used as the tie-breaker."
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    items = load_items()
    print(f"Loaded {len(items)} compact CLEVR examples on {device}.")
    print(
        "Gumbel settings: "
        f"start={args.gumbel_temp_start}, end={args.gumbel_temp_end}, hard={args.hard_gumbel}"
    )
    print(
        "Constraint loss settings: "
        f"global_enabled={not args.disable_global_constraint_loss}, "
        f"global_weight={args.global_constraint_loss_weight}, "
        f"executable_weight={args.executable_constraint_loss_weight}"
    )

    inference = build_program("InferenceProgram", InferenceProgram, items, args, device)
    gumbel = build_program(
        "InferenceProgram(use_gumbel=True)",
        InferenceProgram,
        items,
        args,
        device,
        use_gumbel=True,
    )
    programs = [inference, gumbel]
    graph = inference.program.graph
    print(
        "Graph constraints: "
        f"global={len(getattr(graph, 'logicalConstrains', {}))}, "
        f"executable={len(getattr(graph, 'executableLCs', {}))}"
    )

    print("\nBefore training")
    before = {built.name: evaluate(built, device) for built in programs}
    for built in programs:
        stats = before[built.name]
        sampled_part = ""
        if stats.gumbel_sample_accuracy is not None:
            sampled_part = f", sampled_gumbel_accuracy={stats.gumbel_sample_accuracy:.2f}"
        print(
            f"{built.name}: accuracy={stats.accuracy:.2f}, "
            f"constraint_loss={stats.constraint_loss:.4f} "
            f"(exec={stats.executable_loss:.4f}, global={stats.global_loss:.4f})"
            f"{sampled_part}"
        )
        gradient_report(built)

    print("\nTraining")
    for built in programs:
        print(f"Training {built.name}...")
        train_program(built, args, device)

    print("\nAfter training")
    after = {built.name: evaluate(built, device) for built in programs}
    for built in programs:
        stats = after[built.name]
        sampled_part = ""
        if stats.gumbel_sample_accuracy is not None:
            sampled_part = f", sampled_gumbel_accuracy={stats.gumbel_sample_accuracy:.2f}"
        print(
            f"{built.name}: accuracy={stats.accuracy:.2f}, "
            f"constraint_loss={stats.constraint_loss:.4f} "
            f"(exec={stats.executable_loss:.4f}, global={stats.global_loss:.4f})"
            f"{sampled_part}"
        )

    print_comparison(before, after)


if __name__ == "__main__":
    main()
