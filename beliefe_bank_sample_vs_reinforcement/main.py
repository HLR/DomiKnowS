from __future__ import annotations

import argparse
import gc
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

RUN_DIR = Path(__file__).resolve().parent
TASKS_DIR = RUN_DIR.parent
REPO_ROOT = TASKS_DIR.parent
BELIEF_BANK_DIR = TASKS_DIR / "beliefe_bank"
for path in (RUN_DIR, BELIEF_BANK_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from domiknows import setProductionLogMode

setProductionLogMode(True)

from domiknows.graph import Concept, Graph, Relation, andL, existsL, ifL, notL
from domiknows.program import ReinforcementProgram
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import GumbelSampleLossProgram, SampleLossProgram
from domiknows.program.metric import MacroAverageTracker
from domiknows.program.model.base import Mode
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor import Sensor
from domiknows.sensor.pytorch import ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ReaderSensor

from reward import make_beliefbank_reward_function
from utils import BBRobert, RobertaTokenizer, label_reader, make_facts

logging.basicConfig(level=logging.WARNING)


@dataclass
class BeliefBankData:
    train: list[dict[str, Any]]
    eval: list[dict[str, Any]]
    constraints_yes: dict[str, set[str]]
    constraints_no: dict[str, set[str]]


@dataclass
class BuiltProgram:
    graph: Graph
    subject: Concept
    facts: Concept
    fact_check: Concept
    implication: Concept
    nimplication: Concept
    program: Any


@dataclass
class EvalStats:
    reward: float
    accuracy: float
    constraint_satisfaction: float
    predicted_yes: float
    gold_yes: float
    items: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare BeliefBank SampleLossProgram and ReinforcementProgram."
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-items", type=int, default=8)
    parser.add_argument("--eval-items", type=int, default=8)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"],
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
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


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_constraints(raw: dict[str, Any]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    constraints_yes: dict[str, set[str]] = {}
    constraints_no: dict[str, set[str]] = {}
    for node in raw["nodes"]:
        constraints_yes[node["id"]] = set()
        constraints_no[node["id"]] = set()

    for link in raw["links"]:
        source = link["source"]
        target = link["target"]
        direction = link.get("direction")
        weight = link.get("weight")
        if weight == "yes_yes":
            if direction == "forward":
                constraints_yes[source].add(target)
            else:
                constraints_yes[target].add(source)
        elif (direction == "forward" and weight == "yes_no") or (
            direction == "back" and weight == "no_yes"
        ):
            constraints_no[source].add(target)
        else:
            constraints_no[target].add(source)
    return constraints_yes, constraints_no


def _make_items(
    raw_facts: dict[str, dict[str, str]],
    constraints_yes: dict[str, set[str]],
    constraints_no: dict[str, set[str]],
    batch_size: int,
    item_limit: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for subject_name, facts_by_label in raw_facts.items():
        facts: list[str] = []
        labels: list[str] = []
        for fact, label in facts_by_label.items():
            facts.append(fact)
            labels.append(label)
            if len(facts) == batch_size:
                items.append(_make_item(subject_name, facts, labels, constraints_yes, constraints_no))
                facts, labels = [], []
                if 0 < item_limit <= len(items):
                    return items
        if facts:
            items.append(_make_item(subject_name, facts, labels, constraints_yes, constraints_no))
            if 0 < item_limit <= len(items):
                return items
    return items


def _make_item(
    subject_name: str,
    facts: list[str],
    labels: list[str],
    constraints_yes: dict[str, set[str]],
    constraints_no: dict[str, set[str]],
) -> dict[str, Any]:
    fact_set = set(facts)
    positive_edges = [
        (source, target)
        for source in facts
        for target in constraints_yes.get(source, set())
        if target in fact_set
    ]
    negative_edges = [
        (source, target)
        for source in facts
        for target in constraints_no.get(source, set())
        if target in fact_set
    ]
    return {
        "name": subject_name,
        "facts": [list(facts)],
        "labels": [list(labels)],
        "logic_str": f"BeliefBank({subject_name}): dense label + implication reward",
        "logic_label": {
            "yes": sum(1 for label in labels if label == "yes"),
            "no": sum(1 for label in labels if label == "no"),
            "positive_edges": len(positive_edges),
            "negative_edges": len(negative_edges),
        },
        "positive_edges": positive_edges,
        "negative_edges": negative_edges,
        "reward_function": make_beliefbank_reward_function(
            subject_name, list(facts), list(labels), positive_edges, negative_edges
        ),
    }


def load_beliefbank_data(batch_size: int, train_items: int, eval_items: int) -> BeliefBankData:
    data_dir = BELIEF_BANK_DIR / "data"
    calibration = _read_json(data_dir / "calibration_facts.json")
    silver = _read_json(data_dir / "silver_facts.json")
    raw_constraints = _read_json(data_dir / "constraints_v2.json")
    constraints_yes, constraints_no = _normalize_constraints(raw_constraints)
    return BeliefBankData(
        train=_make_items(calibration, constraints_yes, constraints_no, batch_size, train_items),
        eval=_make_items(silver, constraints_yes, constraints_no, batch_size, eval_items),
        constraints_yes=constraints_yes,
        constraints_no=constraints_no,
    )


def build_graph(
    constraints_yes: dict[str, set[str]],
    constraints_no: dict[str, set[str]],
    device: str,
) -> tuple[Graph, Concept, Concept, Concept, Concept, Concept]:
    Graph.clear()
    Concept.clear()
    Relation.clear()
    Sensor.clear()

    with Graph("belief_bank_sample_vs_reinforcement") as graph:
        subject = Concept(name="subject")
        facts = Concept(name="facts")
        (subject_facts_contains,) = subject.contains(facts)
        tokenizer = RobertaTokenizer()
        facts["token_ids"] = JointSensor(
            "name",
            "sentence",
            forward=lambda name, sentence: tokenizer(name, sentence)[0],
            device=device,
        )
        facts["Mask"] = JointSensor(
            "name",
            "sentence",
            forward=lambda name, sentence: tokenizer(name, sentence)[1],
            device=device,
        )

        fact_check = facts(name="fact_check")
        implication = Concept(name="implication")
        i_arg1, i_arg2 = implication.has_a(arg1=facts, arg2=facts)

        nimplication = Concept(name="nimplication")
        ni_arg1, ni_arg2 = nimplication.has_a(narg1=facts, narg2=facts)

        ifL(
            andL(fact_check("x"), existsL(implication("s", path=("x", i_arg1.reversed)))),
            fact_check(path=("s", i_arg2)),
        )
        ifL(
            andL(fact_check("x"), existsL(nimplication("s", path=("x", ni_arg1.reversed)))),
            notL(fact_check(path=("s", ni_arg2))),
        )

    def guess_pair_yes(sentence, arg1, arg2):
        if len(sentence) < 2 or arg1 == arg2:
            return False
        sentence1 = arg1.getAttribute("sentence")
        sentence2 = arg2.getAttribute("sentence")
        return sentence2 in constraints_yes.get(sentence1, set())

    def guess_pair_no(sentence, narg1, narg2):
        if len(sentence) < 2 or narg1 == narg2:
            return False
        sentence1 = narg1.getAttribute("sentence")
        sentence2 = narg2.getAttribute("sentence")
        return sentence2 in constraints_no.get(sentence1, set())

    subject["name"] = ReaderSensor(keyword="name")
    subject["facts"] = ReaderSensor(keyword="facts")
    subject["labels"] = ReaderSensor(keyword="labels")

    facts[subject_facts_contains, "name", "sentence", "label"] = JointSensor(
        subject["name"],
        subject["facts"],
        subject["labels"],
        forward=make_facts,
        device=device,
    )
    facts[fact_check] = FunctionalSensor(
        subject_facts_contains,
        "label",
        forward=label_reader,
        label=True,
        device=device,
    )

    implication[i_arg1.reversed, i_arg2.reversed] = CompositionCandidateSensor(
        facts["sentence"],
        relations=(i_arg1.reversed, i_arg2.reversed),
        forward=guess_pair_yes,
        device=device,
    )
    nimplication[ni_arg1.reversed, ni_arg2.reversed] = CompositionCandidateSensor(
        facts["sentence"],
        relations=(ni_arg1.reversed, ni_arg2.reversed),
        forward=guess_pair_no,
        device=device,
    )

    facts[fact_check] = ModuleLearner(
        "token_ids", "Mask", module=BBRobert(), device=device
    )
    return graph, subject, facts, fact_check, implication, nimplication


def build_sample_loss_program(
    args: argparse.Namespace,
    data: BeliefBankData,
    device: str,
) -> BuiltProgram:
    set_seed(args.seed)
    graph, subject, facts, fact_check, implication, nimplication = build_graph(
        data.constraints_yes, data.constraints_no, device
    )
    program = SampleLossProgram(
        graph,
        SolverModel,
        poi=[facts[fact_check], implication, nimplication],
        inferTypes=["local/argmax"],
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        sample=True,
        sampleSize=args.sample_size,
        sampleGlobalLoss=False,
        beta=args.beta,
        device=device,
    )
    return BuiltProgram(graph, subject, facts, fact_check, implication, nimplication, program)


def build_gumbel_sample_loss_program(
    args: argparse.Namespace,
    data: BeliefBankData,
    device: str,
) -> BuiltProgram:
    set_seed(args.seed)
    graph, subject, facts, fact_check, implication, nimplication = build_graph(
        data.constraints_yes, data.constraints_no, device
    )
    program = GumbelSampleLossProgram(
        graph,
        SolverModel,
        poi=[facts[fact_check], implication, nimplication],
        inferTypes=["local/argmax"],
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        sample=True,
        sampleSize=args.sample_size,
        sampleGlobalLoss=False,
        beta=args.beta,
        device=device,
        use_gumbel=True,
        initial_temp=args.gumbel_temp_start,
        final_temp=args.gumbel_temp_end,
        hard_gumbel=args.hard_gumbel,
    )
    return BuiltProgram(graph, subject, facts, fact_check, implication, nimplication, program)


def build_reinforcement_program(
    args: argparse.Namespace,
    data: BeliefBankData,
    device: str,
) -> BuiltProgram:
    set_seed(args.seed)
    graph, subject, facts, fact_check, implication, nimplication = build_graph(
        data.constraints_yes, data.constraints_no, device
    )
    program = ReinforcementProgram(
        graph,
        targets=[fact_check],
        reward_key="reward_function",
        decoder=beliefbank_decoder,
        num_samples=args.num_samples,
        estimator="importance_weighted",
        poi=[
            subject,
            facts["name"],
            facts["sentence"],
            facts["label"],
            facts["token_ids"],
            facts["Mask"],
            facts[fact_check],
            implication,
            nimplication,
        ],
        device=device,
    )
    return BuiltProgram(graph, subject, facts, fact_check, implication, nimplication, program)


def beliefbank_decoder(samples, targets, datanode, data_item):
    del datanode
    facts = list(data_item.get("facts", [[]])[0])
    predictions: dict[str, str] = {}
    for concept in targets:
        idx = samples.get(concept)
        if idx is None:
            continue
        for fact, value in zip(facts, idx.reshape(-1).tolist()):
            predictions[fact] = "yes" if int(value) == 1 else "no"
    return predictions


def _fact_datanodes(datanode, fact_check: Concept) -> list[Any]:
    base = datanode.findRootConceptOrRelation(fact_check)
    return datanode.findDatanodes(select=base)


def _logits_from_datanode(datanode, fact_check: Concept) -> torch.Tensor:
    logits: list[torch.Tensor] = []
    for node in _fact_datanodes(datanode, fact_check):
        value = node.getAttribute(fact_check)
        if torch.is_tensor(value) and value.numel() == 2:
            logits.append(value.detach().float().cpu().reshape(-1))
    if not logits:
        raise RuntimeError("No fact_check logits were populated for this item.")
    return torch.stack(logits, dim=0)


def _constraint_satisfaction(datanode) -> float:
    try:
        verification = datanode.verifyResultsLC(key="/local/argmax")
    except Exception:
        return 0.0
    values: list[float] = []
    for result in verification.values():
        score = result.get("ifSatisfied", result.get("satisfied"))
        if score is None:
            continue
        try:
            score_float = float(score)
        except (TypeError, ValueError):
            continue
        if np.isnan(score_float):
            continue
        if score_float > 1.0:
            score_float /= 100.0
        values.append(score_float)
    return float(np.mean(values)) if values else 1.0


def evaluate(
    built: BuiltProgram,
    dataset: list[dict[str, Any]],
    device: str,
    seed: int,
    num_samples: int,
) -> EvalStats:
    set_seed(seed)
    reward_total = 0.0
    correct = 0
    total = 0
    predicted_yes_total = 0
    gold_yes_total = 0
    constraint_scores: list[float] = []

    for item, datanode in zip(dataset, built.program.populate(dataset, device=device)):
        logits = _logits_from_datanode(datanode, built.fact_check)
        probs = torch.softmax(logits, dim=-1)
        draws = torch.multinomial(probs, num_samples=num_samples, replacement=True)
        reward_fn = item["reward_function"]
        sample_rewards = []
        for sample_idx in range(num_samples):
            predictions = {
                fact: "yes" if int(value) == 1 else "no"
                for fact, value in zip(item["facts"][0], draws[:, sample_idx].tolist())
            }
            sample_rewards.append(float(reward_fn(predictions).mean().item()))
        reward_total += float(np.mean(sample_rewards)) if sample_rewards else 0.0

        pred = probs.argmax(dim=-1)
        gold = torch.tensor([1 if label == "yes" else 0 for label in item["labels"][0]])
        correct += int((pred.cpu() == gold).sum().item())
        total += int(gold.numel())
        predicted_yes_total += int((pred.cpu() == 1).sum().item())
        gold_yes_total += int((gold == 1).sum().item())
        constraint_scores.append(_constraint_satisfaction(datanode))

    items = max(1, len(dataset))
    return EvalStats(
        reward=reward_total / items,
        accuracy=(correct / total) if total else 0.0,
        constraint_satisfaction=float(np.mean(constraint_scores)) if constraint_scores else 0.0,
        predicted_yes=predicted_yes_total / items,
        gold_yes=gold_yes_total / items,
        items=len(dataset),
    )


def _zero_grads(program: Any) -> None:
    for module_name in ("model", "cmodel"):
        module = getattr(program, module_name, None)
        if module is not None:
            module.zero_grad(set_to_none=True)
    opt = getattr(program, "opt", None)
    if opt is not None:
        opt.zero_grad(set_to_none=True)


def _grad_summary(program: Any, top_k: int = 5) -> dict[str, Any]:
    rows: list[tuple[str, float, float]] = []
    total_sq = 0.0
    max_abs = 0.0
    for name, param in program.model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        norm = float(grad.norm(2).item())
        max_grad = float(grad.abs().max().item()) if grad.numel() else 0.0
        rows.append((name, norm, max_grad))
        total_sq += norm * norm
        max_abs = max(max_abs, max_grad)
    rows.sort(key=lambda row: row[1], reverse=True)
    return {
        "l2": total_sq**0.5,
        "max_abs": max_abs,
        "param_count": len(rows),
        "top": rows[:top_k],
    }


def _print_grad_report(prefix: str, value_label: str, value: float, summary: dict[str, Any]) -> None:
    print(f"{prefix} gradient report")
    print(f"  {value_label}: {value:.6f}")
    print(f"  model grad L2 norm: {summary['l2']:.6f}")
    print(f"  max abs grad: {summary['max_abs']:.6f}")
    print(f"  trainable params with gradients: {summary['param_count']}")
    if summary["top"]:
        print("  top parameter gradient norms:")
        for name, norm, max_grad in summary["top"]:
            print(f"    {name}: norm={norm:.6f}, max={max_grad:.6f}")


def _loss_value(loss: Any) -> float:
    """Convert tensor and scalar diagnostic losses to a printable float."""
    if torch.is_tensor(loss):
        return float(loss.detach().item())
    if isinstance(loss, (int, float, np.number)):
        return float(loss)
    return float("nan")


def sample_loss_gradient_report(
    built: BuiltProgram,
    item: dict[str, Any],
    device: str,
    label: str,
    program_name: str = "SampleLossProgram",
) -> None:
    _zero_grads(built.program)
    built.program.to(device)
    built.program.model.mode(Mode.TRAIN)
    built.program.model.train()
    built.program.cmodel.train()
    _mloss, _metric, _datanode, builder = built.program.model(item)
    if isinstance(built.program, GumbelSampleLossProgram):
        built.program._update_temperature_for_epoch()
        loss, *_ = built.program._call_cmodel_with_gumbel(builder)
    else:
        loss, *_ = built.program.cmodel(builder)
    if torch.is_tensor(loss) and loss.requires_grad:
        loss.backward()
    summary = _grad_summary(built.program)
    value = _loss_value(loss)
    _print_grad_report(f"{program_name} {label}", "sampled constraint loss", value, summary)
    _zero_grads(built.program)


def reinforcement_gradient_report(
    built: BuiltProgram,
    item: dict[str, Any],
    device: str,
    label: str,
) -> None:
    _zero_grads(built.program)
    built.program.to(device)
    built.program.model.mode(Mode.TRAIN)
    built.program.model.train()
    _mloss, _metric, datanode, _builder = built.program.model(item)
    loss, mean_reward = built.program.reinforcement_loss(
        datanode, item["reward_function"], item
    )
    if torch.is_tensor(loss) and loss.requires_grad:
        loss.backward()
    summary = _grad_summary(built.program)
    loss_value = _loss_value(loss)
    _print_grad_report(
        f"ReinforcementProgram {label}",
        f"reinforcement loss (mean reward {mean_reward:.6f})",
        loss_value,
        summary,
    )
    _zero_grads(built.program)


def train_sample_loss(
    built: BuiltProgram,
    dataset: list[dict[str, Any]],
    args: argparse.Namespace,
    device: str,
) -> None:
    built.program.train(
        dataset,
        train_epoch_num=args.epochs,
        Optim=lambda params: torch.optim.AdamW(
            [param for param in params if param.requires_grad], lr=args.lr
        ),
        c_lr=args.lr,
        c_warmup_iters=0,
        device=device,
    )


def train_reinforcement(
    built: BuiltProgram,
    dataset: list[dict[str, Any]],
    args: argparse.Namespace,
    device: str,
) -> None:
    built.program.train(
        dataset,
        train_epoch_num=args.epochs,
        Optim=lambda params: torch.optim.AdamW(
            [param for param in params if param.requires_grad], lr=args.lr
        ),
        device=device,
    )


def _print_stats(name: str, before: EvalStats, after: EvalStats) -> None:
    print(f"\n{name}")
    print(f"  reward: {before.reward:.4f} -> {after.reward:.4f} ({after.reward - before.reward:+.4f})")
    print(f"  gold-label argmax accuracy: {before.accuracy:.4f} -> {after.accuracy:.4f}")
    print(
        "  graph constraint satisfaction: "
        f"{before.constraint_satisfaction:.4f} -> {after.constraint_satisfaction:.4f}"
    )
    print(
        "  predicted yes count / gold yes count: "
        f"{before.predicted_yes:.2f}/{before.gold_yes:.2f} -> "
        f"{after.predicted_yes:.2f}/{after.gold_yes:.2f}"
    )


def _print_winner(results: list[tuple[str, EvalStats]]) -> None:
    ordered = sorted(results, key=lambda row: row[1].reward, reverse=True)
    top_reward = ordered[0][1].reward
    ties = [name for name, stats in ordered if abs(stats.reward - top_reward) <= 1e-6]
    if len(ties) > 1:
        print("\nWinner: tie on generated BeliefBank reward: " + ", ".join(ties))
    else:
        print(f"\nWinner: {ordered[0][0]} by generated BeliefBank reward.")
    for name, stats in ordered:
        print(f"  final reward, {name}: {stats.reward:.4f}")
    print(
        "  Sample-loss variants use graph constraint-loss gradients; the Gumbel "
        "variant applies a differentiable Gumbel-Softmax perturbation before "
        "that loss. ReinforcementProgram optimizes the dense generated-output "
        "reward directly, but with sampled policy-gradient variance."
    )


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)

    data = load_beliefbank_data(args.batch_size, args.train_items, args.eval_items)
    print("BeliefBank SampleLossProgram vs GumbelSampleLossProgram vs ReinforcementProgram")
    print(f"  device: {device}")
    print(f"  train/eval items: {len(data.train)}/{len(data.eval)}")
    print(f"  batch size: {args.batch_size}")
    print("  ReinforcementProgram reward: explicit dense generated-output belief reward")

    print("\nSampleLossProgram diagnostics")
    sample = build_sample_loss_program(args, data, device)
    sample_loss_gradient_report(sample, data.train[0], device, "before training")
    sample_before = evaluate(sample, data.eval, device, args.seed + 11, args.num_samples)
    print("\nTraining SampleLossProgram")
    train_sample_loss(sample, data.train, args, device)
    sample_loss_gradient_report(sample, data.train[0], device, "after training")
    sample_after = evaluate(sample, data.eval, device, args.seed + 29, args.num_samples)
    del sample
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nGumbelSampleLossProgram diagnostics")
    gumbel = build_gumbel_sample_loss_program(args, data, device)
    sample_loss_gradient_report(
        gumbel,
        data.train[0],
        device,
        "before training",
        program_name="GumbelSampleLossProgram",
    )
    gumbel_before = evaluate(gumbel, data.eval, device, args.seed + 11, args.num_samples)
    print("\nTraining GumbelSampleLossProgram")
    train_sample_loss(gumbel, data.train, args, device)
    sample_loss_gradient_report(
        gumbel,
        data.train[0],
        device,
        "after training",
        program_name="GumbelSampleLossProgram",
    )
    gumbel_after = evaluate(gumbel, data.eval, device, args.seed + 29, args.num_samples)
    del gumbel
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nReinforcementProgram diagnostics")
    reinforcement = build_reinforcement_program(args, data, device)
    reinforcement_gradient_report(reinforcement, data.train[0], device, "before training")
    rl_before = evaluate(reinforcement, data.eval, device, args.seed + 11, args.num_samples)

    print("\nTraining ReinforcementProgram")
    train_reinforcement(reinforcement, data.train, args, device)
    reinforcement_gradient_report(reinforcement, data.train[0], device, "after training")
    rl_after = evaluate(reinforcement, data.eval, device, args.seed + 29, args.num_samples)

    _print_stats("SampleLossProgram", sample_before, sample_after)
    _print_stats("GumbelSampleLossProgram", gumbel_before, gumbel_after)
    _print_stats("ReinforcementProgram", rl_before, rl_after)
    _print_winner([
        ("SampleLossProgram", sample_after),
        ("GumbelSampleLossProgram", gumbel_after),
        ("ReinforcementProgram", rl_after),
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
