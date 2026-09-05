from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter
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
    query_namespace: dict[str, Any]
    ilp_benchmark_sample: Any | None = None
    ilp_benchmark_samples: tuple[Any, ...] = ()


@dataclass
class EvalStats:
    accuracy: float
    constraint_loss: float
    executable_loss: float
    global_loss: float
    gumbel_sample_accuracy: float | None = None


@dataclass
class AdHocComparison:
    results: dict[str, dict[str, Any]]
    answers_agree: bool
    types_agree: bool


@dataclass(frozen=True)
class ILPTiming:
    durations_seconds: tuple[float, ...]
    median_seconds: float
    answer: Any
    result_type: str
    active_concepts: tuple[str, ...]
    predicate_count: int


@dataclass(frozen=True)
class ILPGraphPerformance:
    sample: dict[str, Any]
    requested_concepts: tuple[str, ...]
    full: ILPTiming
    dynamic: ILPTiming
    milliseconds_saved: float
    reduction_percent: float
    speedup: float
    answers_agree: bool


@dataclass(frozen=True)
class ILPBenchmarkFailure:
    sample: dict[str, Any]
    question_type: str
    error_type: str
    error: str


@dataclass(frozen=True)
class ILPQuestionTypePerformance:
    question_type: str
    attempted: int
    succeeded: int
    failed: int
    full_average_seconds: float | None
    dynamic_average_seconds: float | None
    milliseconds_saved: float | None
    reduction_percent: float | None
    speedup: float | None
    answers_agree: bool | None
    full_average_predicates: float | None
    dynamic_average_predicates: float | None


@dataclass(frozen=True)
class ILPBenchmarkReport:
    comparisons: tuple[ILPGraphPerformance, ...]
    failures: tuple[ILPBenchmarkFailure, ...]
    question_types: tuple[ILPQuestionTypePerformance, ...]
    attempted: int
    full_workload_seconds: float
    dynamic_workload_seconds: float
    milliseconds_saved: float
    reduction_percent: float
    speedup: float
    answers_agree: bool


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
    parser.add_argument(
        "--ilp-benchmark-warmup",
        type=int,
        default=1,
        help="Number of discarded full/dynamic ILP timing pairs.",
    )
    parser.add_argument(
        "--ilp-benchmark-repeats",
        type=int,
        default=3,
        help="Number of measured full/dynamic ILP timing pairs.",
    )
    parser.add_argument(
        "--ilp-benchmark-items",
        type=int,
        default=0,
        help=(
            "Number of compiled questions to benchmark in dataset order; "
            "use 0 for all questions (default: 0)."
        ),
    )
    parser.add_argument(
        "--ilp-benchmark-only",
        action="store_true",
        help=(
            "Train only the standard InferenceProgram and run only the "
            "full/dynamic ILP benchmark."
        ),
    )
    args = parser.parse_args()
    if args.ilp_benchmark_warmup < 0:
        parser.error("--ilp-benchmark-warmup must be non-negative")
    if args.ilp_benchmark_repeats < 1:
        parser.error("--ilp-benchmark-repeats must be positive")
    if args.ilp_benchmark_items < 0:
        parser.error("--ilp-benchmark-items must be non-negative")
    return args


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


def _select_relation_free_count_samples(
    dataset: list[dict[str, Any]],
    compiled: Any,
) -> tuple[dict[str, Any], ...]:
    """Select deterministic count steps that do not traverse scene relations."""
    selected = []
    for index, item in enumerate(dataset):
        functions = [
            step.get("function", step.get("type"))
            for step in item.get("program", ())
        ]
        if not functions or functions[-1] != "count":
            continue
        if all(
            function in {"scene", "unique", "count", "union", "intersect"}
            or (isinstance(function, str) and function.startswith("filter_"))
            for function in functions
        ):
            selected.append(compiled[index])
    if not selected:
        raise ValueError(
            "The ILP graph benchmark requires a relation-free count question"
        )
    return tuple(selected)


def _select_simple_count_sample(
    dataset: list[dict[str, Any]],
    compiled: Any,
) -> dict[str, Any]:
    """Backward-compatible accessor for the first benchmark count sample."""
    return _select_relation_free_count_samples(dataset, compiled)[0]


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
    ilp_benchmark_samples = tuple(compiled)
    ilp_benchmark_sample = _select_simple_count_sample(dataset, compiled)

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
        "loss": torch.nn.BCELoss,
        "query_loss": NBCrossEntropyLoss,
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
    query_namespace = {
        **attribute_names_dict,
        "obj": objects,
        "obj1": obj1,
        "obj2": obj2,
        "pair_forward": pair_forward,
    }
    if obj1_rev is not None:
        query_namespace["obj1_rev"] = obj1_rev
    if obj2_rev is not None:
        query_namespace["obj2_rev"] = obj2_rev
    if pair_reverse is not None:
        query_namespace["pair_reverse"] = pair_reverse

    return BuiltProgram(
        name=name,
        program=program,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        query_namespace=query_namespace,
        ilp_benchmark_sample=ilp_benchmark_sample,
        ilp_benchmark_samples=ilp_benchmark_samples,
    )


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


def infer_ad_hoc_example(
    built: BuiltProgram,
    device: str,
) -> tuple[dict[str, Any], AdHocComparison]:
    """Run and compare t-norm, circuit, and ILP ad hoc inference."""
    if built.ilp_benchmark_sample is not None:
        sample = built.ilp_benchmark_sample
    elif built.eval_dataset:
        sample = built.eval_dataset[0]
    else:
        raise ValueError(f"{built.name} has no evaluation sample for an ad hoc query")
    query = sample.get("logic_str")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("The ad hoc CLEVR example requires a non-empty logic_str")

    built.program.model.eval()
    built.program.cmodel.eval()
    with torch.no_grad():
        _mloss, _metric, _datanode, builder = built.program.model(sample)
        datanode = builder.getDataNode(device=device)
        if datanode is None:
            raise RuntimeError("The learned model did not produce a DataNode")

        mode_results = {}
        for mode in ("tnorm", "circuit", "ilp"):
            results = datanode.inferExecutableResults(
                mode=mode,
                tnorm=built.program.cmodel.tnorm,
                queries=query,
                queryNamespace=built.query_namespace,
                populate=False,
                compiled=False,
            )
            mode_results[mode] = results["ADHOC0"]

    answers = [result["answer"] for result in mode_results.values()]
    result_types = [result["type"] for result in mode_results.values()]
    comparison = AdHocComparison(
        results=mode_results,
        answers_agree=all(answer == answers[0] for answer in answers[1:]),
        types_agree=all(
            result_type == result_types[0]
            for result_type in result_types[1:]
        ),
    )
    return sample, comparison


def _benchmark_executable_name(
    built: BuiltProgram,
    sample: dict[str, Any],
) -> str:
    names = [
        key.removeprefix("_constraint_")
        for key in sample
        if isinstance(key, str) and key.startswith("_constraint_ELC")
    ]
    registered = [
        name for name in names
        if name in built.program.graph.executableLCs
    ]
    if len(registered) != 1:
        raise ValueError(
            "The ILP graph benchmark sample must activate exactly one "
            f"registered executable constraint, found {registered}"
        )
    return registered[0]


def _synchronize_device(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def _question_type(sample: dict[str, Any]) -> str:
    """Return the terminal CLEVR operation used for benchmark grouping."""
    program = sample.get("program") or ()
    if not program:
        return "unknown"
    final_step = program[-1]
    value = final_step.get("function", final_step.get("type"))
    return value if isinstance(value, str) and value else "unknown"


def _executable_result_type(executable: Any) -> str:
    from domiknows.graph.logicalConstrain import miotaL, queryL, sumL

    lc = executable.innerLC
    if isinstance(lc, sumL):
        return "count"
    if isinstance(lc, queryL):
        return "multi_query" if lc.is_multi_answer else "query"
    if isinstance(lc, miotaL):
        return "selection"
    return "boolean"


def _measure_ilp_configuration(
    built: BuiltProgram,
    sample: dict[str, Any],
    device: str,
    executable_name: str,
    requested_concepts: tuple[str, ...] | None,
) -> tuple[float, dict[str, Any], tuple[str, ...], int]:
    """Build a fresh DataNode, then time ``inferILPResults`` directly."""
    graph = built.program.graph
    graph.set_active_concepts(requested_concepts)

    with torch.no_grad():
        _mloss, _metric, _datanode, builder = built.program.model(sample)
        datanode = builder.getDataNode(device=device)
    if datanode is None:
        raise RuntimeError("The learned model did not produce a DataNode")

    constraint_dn = datanode._getExecutableConstraintDataNode()
    if constraint_dn is None:
        raise RuntimeError("The ILP benchmark sample has no constraint DataNode")
    target_label = f"{executable_name}/label"
    for key in list(constraint_dn.attributes):
        if isinstance(key, str) and key.endswith("/label") and key != target_label:
            del constraint_dn.attributes[key]
    if target_label not in constraint_dn.attributes:
        constraint_dn.attributes[target_label] = torch.tensor(
            1, device=datanode.current_device
        )

    active_concepts = graph.get_active_concepts()
    active_names = tuple(concept.name for concept in active_concepts)
    _synchronize_device(device)
    started = perf_counter()
    datanode.inferILPResults()
    _synchronize_device(device)
    elapsed = perf_counter() - started
    answer_key = f"{executable_name}/answer"
    if answer_key not in constraint_dn.attributes:
        raise RuntimeError(
            f"inferILPResults did not persist an answer for {executable_name}"
        )
    result = {
        "type": _executable_result_type(
            built.program.graph.executableLCs[executable_name]
        ),
        "answer": constraint_dn.attributes[answer_key],
    }
    predicate_count = len(datanode.collectedConceptsAndRelations or ())
    return elapsed, result, active_names, predicate_count


def _summarize_ilp_timings(
    measured: list[tuple[float, dict[str, Any], tuple[str, ...], int]],
) -> ILPTiming:
    durations = tuple(row[0] for row in measured)
    answers = [row[1]["answer"] for row in measured]
    result_types = [row[1]["type"] for row in measured]
    active_concepts = [row[2] for row in measured]
    predicate_counts = [row[3] for row in measured]
    if any(answer != answers[0] for answer in answers[1:]):
        raise RuntimeError("ILP benchmark answers changed between repetitions")
    if any(result_type != result_types[0] for result_type in result_types[1:]):
        raise RuntimeError("ILP benchmark result types changed between repetitions")
    if any(names != active_concepts[0] for names in active_concepts[1:]):
        raise RuntimeError("ILP benchmark active concepts changed between repetitions")
    if any(count != predicate_counts[0] for count in predicate_counts[1:]):
        raise RuntimeError("ILP benchmark predicate count changed between repetitions")
    return ILPTiming(
        durations_seconds=durations,
        median_seconds=float(median(durations)),
        answer=answers[0],
        result_type=result_types[0],
        active_concepts=active_concepts[0],
        predicate_count=predicate_counts[0],
    )


def benchmark_ilp_graph_activation(
    built: BuiltProgram,
    device: str,
    *,
    warmup: int = 1,
    repeats: int = 3,
    sample: dict[str, Any] | None = None,
) -> ILPGraphPerformance:
    """Compare full and query-scoped ILP inference on one learned model."""
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    sample = sample if sample is not None else built.ilp_benchmark_sample
    if sample is None:
        raise ValueError(f"{built.name} has no ILP benchmark sample")
    query = sample.get("logic_str")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("The ILP graph benchmark requires a non-empty logic_str")

    executable_name = _benchmark_executable_name(built, sample)
    executable = built.program.graph.executableLCs[executable_name]
    requested_concepts = tuple(sorted(executable.innerLC.getLcConcepts()))
    if not requested_concepts:
        raise ValueError("The ILP graph benchmark query references no concepts")

    built.program.model.eval()
    built.program.cmodel.eval()
    full_measurements = []
    dynamic_measurements = []
    try:
        for iteration in range(warmup + repeats):
            full = _measure_ilp_configuration(
                built,
                sample,
                device,
                executable_name,
                requested_concepts=None,
            )
            dynamic = _measure_ilp_configuration(
                built,
                sample,
                device,
                executable_name,
                requested_concepts=requested_concepts,
            )
            if iteration >= warmup:
                full_measurements.append(full)
                dynamic_measurements.append(dynamic)
    finally:
        built.program.graph.set_active_concepts(None)

    full_timing = _summarize_ilp_timings(full_measurements)
    dynamic_timing = _summarize_ilp_timings(dynamic_measurements)
    saved_seconds = full_timing.median_seconds - dynamic_timing.median_seconds
    reduction_percent = (
        100.0 * saved_seconds / full_timing.median_seconds
        if full_timing.median_seconds
        else 0.0
    )
    speedup = (
        full_timing.median_seconds / dynamic_timing.median_seconds
        if dynamic_timing.median_seconds
        else float("inf")
    )
    return ILPGraphPerformance(
        sample=sample,
        requested_concepts=requested_concepts,
        full=full_timing,
        dynamic=dynamic_timing,
        milliseconds_saved=saved_seconds * 1000.0,
        reduction_percent=reduction_percent,
        speedup=speedup,
        answers_agree=full_timing.answer == dynamic_timing.answer,
    )


def benchmark_ilp_graph_activations(
    built: BuiltProgram,
    device: str,
    *,
    warmup: int = 1,
    repeats: int = 3,
    items: int = 0,
) -> ILPBenchmarkReport:
    """Benchmark every selected question and aggregate results by CLEVR type."""
    if items < 0:
        raise ValueError("items must be non-negative")
    samples = built.ilp_benchmark_samples
    if not samples and built.ilp_benchmark_sample is not None:
        samples = (built.ilp_benchmark_sample,)
    if not samples:
        raise ValueError(f"{built.name} has no ILP benchmark samples")
    selected = samples if items == 0 else samples[:items]
    completed = []
    failed = []
    for sample in selected:
        try:
            completed.append(
                benchmark_ilp_graph_activation(
                    built,
                    device,
                    warmup=warmup,
                    repeats=repeats,
                    sample=sample,
                )
            )
        except Exception as error:
            failed.append(
                ILPBenchmarkFailure(
                    sample=sample,
                    question_type=_question_type(sample),
                    error_type=type(error).__name__,
                    error=str(error),
                )
            )
    comparisons = tuple(completed)
    failures = tuple(failed)

    type_names = tuple(dict.fromkeys(_question_type(sample) for sample in selected))
    type_summaries = []
    for question_type in type_names:
        type_samples = [
            sample for sample in selected
            if _question_type(sample) == question_type
        ]
        type_rows = [
            row for row in comparisons
            if _question_type(row.sample) == question_type
        ]
        type_failures = [
            row for row in failures if row.question_type == question_type
        ]
        if type_rows:
            full_average = mean(row.full.median_seconds for row in type_rows)
            dynamic_average = mean(
                row.dynamic.median_seconds for row in type_rows
            )
            saved = full_average - dynamic_average
            reduction = (
                100.0 * saved / full_average if full_average else 0.0
            )
            speedup = (
                full_average / dynamic_average
                if dynamic_average else float("inf")
            )
            answers_agree = all(row.answers_agree for row in type_rows)
            full_predicates = mean(
                row.full.predicate_count for row in type_rows
            )
            dynamic_predicates = mean(
                row.dynamic.predicate_count for row in type_rows
            )
        else:
            full_average = None
            dynamic_average = None
            saved = None
            reduction = None
            speedup = None
            answers_agree = None
            full_predicates = None
            dynamic_predicates = None
        type_summaries.append(
            ILPQuestionTypePerformance(
                question_type=question_type,
                attempted=len(type_samples),
                succeeded=len(type_rows),
                failed=len(type_failures),
                full_average_seconds=full_average,
                dynamic_average_seconds=dynamic_average,
                milliseconds_saved=(
                    saved * 1000.0 if saved is not None else None
                ),
                reduction_percent=reduction,
                speedup=speedup,
                answers_agree=answers_agree,
                full_average_predicates=full_predicates,
                dynamic_average_predicates=dynamic_predicates,
            )
        )
    full_seconds = sum(row.full.median_seconds for row in comparisons)
    dynamic_seconds = sum(row.dynamic.median_seconds for row in comparisons)
    saved_seconds = full_seconds - dynamic_seconds
    reduction_percent = (
        100.0 * saved_seconds / full_seconds if full_seconds else 0.0
    )
    speedup = full_seconds / dynamic_seconds if dynamic_seconds else float("inf")
    return ILPBenchmarkReport(
        comparisons=comparisons,
        failures=failures,
        question_types=tuple(type_summaries),
        attempted=len(selected),
        full_workload_seconds=full_seconds,
        dynamic_workload_seconds=dynamic_seconds,
        milliseconds_saved=saved_seconds * 1000.0,
        reduction_percent=reduction_percent,
        speedup=speedup,
        answers_agree=all(row.answers_agree for row in comparisons),
    )


def _is_gurobi_license_limit(failure: ILPBenchmarkFailure) -> bool:
    """Return whether a benchmark failure is only the free-license size cap."""
    message = failure.error.lower()
    return (
        failure.error_type == "GurobiError"
        and "size-limited license" in message
        and "model too large" in message
    )


def _question_type_table(report: ILPBenchmarkReport) -> tuple[str, ...]:
    """Format the per-question-type benchmark summary as a console table."""
    headers = (
        "Question type",
        "Success/total",
        "Full avg.",
        "Dynamic avg.",
        "Speedup",
    )
    rows = []
    for summary in report.question_types:
        ignored = sum(
            1
            for failure in report.failures
            if failure.question_type == summary.question_type
            and _is_gurobi_license_limit(failure)
        )
        effective_total = summary.attempted - ignored
        if summary.succeeded:
            full_average = f"{summary.full_average_seconds * 1000.0:.2f} ms"
            dynamic_average = (
                f"{summary.dynamic_average_seconds * 1000.0:.2f} ms"
            )
            speedup = f"{summary.speedup:.2f}\N{MULTIPLICATION SIGN}"
        else:
            full_average = dynamic_average = speedup = "n/a"
        rows.append(
            (
                summary.question_type,
                f"{summary.succeeded}/{effective_total}",
                full_average,
                dynamic_average,
                speedup,
            )
        )

    successful = len(report.comparisons)
    aggregate_row = None
    if successful:
        aggregate_row = (
            "Successful-question aggregate",
            f"{successful}/{successful}",
            f"{report.full_workload_seconds * 1000.0 / successful:.2f} ms",
            f"{report.dynamic_workload_seconds * 1000.0 / successful:.2f} ms",
            f"{report.speedup:.2f}\N{MULTIPLICATION SIGN}",
        )

    width_rows = rows + ([aggregate_row] if aggregate_row else [])
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in width_rows))
        for column in range(len(headers))
    ]

    def format_row(row: tuple[str, ...]) -> str:
        return "  ".join(
            value.ljust(widths[column])
            for column, value in enumerate(row)
        ).rstrip()

    separator = format_row(tuple("-" * width for width in widths))
    lines = [
        format_row(headers),
        separator,
        *(format_row(row) for row in rows),
    ]
    if aggregate_row:
        lines.extend((separator, format_row(aggregate_row)))
    return tuple(lines)


def print_post_training_ilp_benchmark(
    built: BuiltProgram,
    device: str,
    *,
    warmup: int,
    repeats: int,
    items: int = 0,
) -> ILPBenchmarkReport:
    """Run and print the standard-model multi-question ILP comparison."""
    report = benchmark_ilp_graph_activations(
        built,
        device,
        warmup=warmup,
        repeats=repeats,
        items=items,
    )
    license_skips = tuple(
        failure
        for failure in report.failures
        if _is_gurobi_license_limit(failure)
    )
    inference_failures = tuple(
        failure
        for failure in report.failures
        if not _is_gurobi_license_limit(failure)
    )
    print(
        "\nPost-training ILP full-graph vs dynamic-graph benchmark "
        f"({report.attempted} questions: "
        f"{len(report.comparisons)} succeeded, "
        f"{len(license_skips)} skipped by Gurobi license, "
        f"{len(inference_failures)} failed)"
    )
    print(f"{built.name}:")
    for index, comparison in enumerate(report.comparisons, start=1):
        print(f"  completed question {index}/{len(report.comparisons)}={comparison.sample.get('question')!r}")
        print(f"    expected={comparison.sample.get('answer')!r}")
        print(f"    requested_concepts={list(comparison.requested_concepts)!r}")
        print(
            f"    full: median={comparison.full.median_seconds * 1000.0:.2f}ms, "
            f"active_concepts={len(comparison.full.active_concepts)}, "
            f"ilp_predicates={comparison.full.predicate_count}, "
            f"answer={comparison.full.answer!r}"
        )
        print(
            f"    dynamic: median={comparison.dynamic.median_seconds * 1000.0:.2f}ms, "
            f"active_concepts={len(comparison.dynamic.active_concepts)}, "
            f"ilp_predicates={comparison.dynamic.predicate_count}, "
            f"answer={comparison.dynamic.answer!r}"
        )
        print(
            f"    difference: saved={comparison.milliseconds_saved:.2f}ms, "
            f"reduction={comparison.reduction_percent:.2f}%, "
            f"speedup={comparison.speedup:.2f}x, "
            f"answers_agree={comparison.answers_agree}"
        )
    for index, failure in enumerate(license_skips, start=1):
        print(
            f"  license-skipped question {index}/{len(license_skips)}="
            f"{failure.sample.get('question')!r}"
        )
        print(f"    type={failure.question_type}, reason={failure.error}")
    for index, failure in enumerate(inference_failures, start=1):
        print(
            f"  failed question {index}/{len(inference_failures)}="
            f"{failure.sample.get('question')!r}"
        )
        print(
            f"    type={failure.question_type}, "
            f"error={failure.error_type}: {failure.error}"
        )
    print("  averages by question type (successful questions):")
    for line in _question_type_table(report):
        print(f"    {line}")
    return report


def print_post_training_ad_hoc_results(
    programs: list[BuiltProgram],
    device: str,
) -> None:
    """Print one return-only ad hoc answer from every learned model."""
    print("\nPost-training ad hoc executable query")
    for built in programs:
        sample, comparison = infer_ad_hoc_example(built, device)
        print(f"{built.name}:")
        print(f"  question={sample.get('question')!r}")
        print(f"  expected={sample.get('answer')!r}")
        for mode, result in comparison.results.items():
            distribution = result["distribution"]
            distribution_text = (
                "None" if distribution is None else str(distribution.tolist())
            )
            print(
                f"  {mode}: type={result['type']}, "
                f"answer={result['answer']!r}, "
                f"probability={result['probability']}, "
                f"distribution={distribution_text}"
            )
        print(
            f"  comparison: answers_agree={comparison.answers_agree}, "
            f"types_agree={comparison.types_agree}"
        )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    items = load_items()
    print(f"Loaded {len(items)} compact CLEVR examples on {device}.")
    print(
        "Constraint loss settings: "
        f"global_enabled={not args.disable_global_constraint_loss}, "
        f"global_weight={args.global_constraint_loss_weight}, "
        f"executable_weight={args.executable_constraint_loss_weight}"
    )

    inference = build_program("InferenceProgram", InferenceProgram, items, args, device)
    if args.ilp_benchmark_only:
        print("\nTraining InferenceProgram for ILP benchmark...")
        train_program(inference, args, device)
        print_post_training_ilp_benchmark(
            inference,
            device,
            warmup=args.ilp_benchmark_warmup,
            repeats=args.ilp_benchmark_repeats,
            items=args.ilp_benchmark_items,
        )
        return

    print(
        "Gumbel settings: "
        f"start={args.gumbel_temp_start}, end={args.gumbel_temp_end}, hard={args.hard_gumbel}"
    )
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
    print_post_training_ad_hoc_results(programs, device)
    print_post_training_ilp_benchmark(
        inference,
        device,
        warmup=args.ilp_benchmark_warmup,
        repeats=args.ilp_benchmark_repeats,
        items=args.ilp_benchmark_items,
    )


if __name__ == "__main__":
    main()
