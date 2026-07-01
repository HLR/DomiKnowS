from __future__ import annotations

import argparse
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

from domiknows.graph import Concept, EnumConcept, Graph, Relation
from domiknows.graph.logicalConstrain import exactL
from domiknows.program import ReinforcementProgram
from domiknows.program.lossprogram import SampleLossProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor import Sensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor
from domiknows.sensor.pytorch.sensors import ReaderSensor

from reward import make_count_reward_function


class ElementClassifier(nn.Module):
    """Small per-`b` classifier used by both programs."""

    def __init__(self, input_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Tanh(),
            nn.Linear(input_size, 2),
        )

    def forward(self, relation_tensor: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        del relation_tensor
        return self.layers(features.float())


@dataclass
class BuiltExample:
    graph: Graph
    a: Concept
    b: Concept
    b_answer: Concept
    program: Any


@dataclass
class Stats:
    sampled_reward: float
    expected_zeros: float
    argmax_zeros: int
    zero_probs: list[float]
    program_reward: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare sampling-based primal-dual constraint learning with ReinforcementProgram."
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--features", type=int, default=6)
    parser.add_argument("--num-b", type=int, default=8)
    parser.add_argument("--expected-zeros", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda", "cuda:0", "cuda:1"])
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--eval-samples", type=int, default=300)
    parser.add_argument(
        "--rl-estimator",
        choices=["importance_weighted", "reinforce"],
        default="importance_weighted",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return "cuda:0"
    return device


def create_dataset(seed: int, features: int, num_b: int, expected_zeros: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    logic_str = f"exactL(answer_b.zero, limit={expected_zeros})"
    logic_label = {
        "expected_value": "zero",
        "expected_count": expected_zeros,
        "mode": "exact",
    }
    return [{
        "a": [0],
        "b": rng.normal(size=(num_b, features)).astype(np.float32).tolist(),
        "logic_str": logic_str,
        "logic_label": logic_label,
        "reward_function": make_count_reward_function(logic_str, logic_label),
    }]


def build_graph(expected_zeros: int) -> tuple[Graph, Concept, Concept, Any, Concept]:
    Graph.clear()
    Concept.clear()
    Relation.clear()
    Sensor.clear()

    with Graph("sample_loss_vs_reinforcement") as graph:
        a = Concept(name="a")
        b = Concept(name="b")
        (a_contains_b,) = a.contains(b)
        b_answer = b(name="answer_b", ConceptClass=EnumConcept, values=["zero", "one"])
        exactL(b_answer.__getattr__("zero"), limit=expected_zeros)

    return graph, a, b, a_contains_b, b_answer


def connect_sensors(
    a: Concept,
    b: Concept,
    a_contains_b: Any,
    b_answer: Concept,
    features: int,
    device: str,
) -> ElementClassifier:
    a["index"] = ReaderSensor(keyword="a")
    b["features"] = ReaderSensor(keyword="b")
    b[a_contains_b] = EdgeSensor(
        b["features"],
        a["index"],
        relation=a_contains_b,
        forward=lambda b_features, _a_index: torch.ones(len(b_features)).unsqueeze(-1),
    )

    module = ElementClassifier(features).to(device)
    b[b_answer] = ModuleLearner(a_contains_b, "features", module=module, device=device)
    return module


def build_sample_loss_example(args: argparse.Namespace, device: str) -> BuiltExample:
    torch.manual_seed(args.seed)
    graph, a, b, a_contains_b, b_answer = build_graph(args.expected_zeros)
    connect_sensors(a, b, a_contains_b, b_answer, args.features, device)
    program = SampleLossProgram(
        graph,
        SolverModel,
        poi=[a, b, b_answer],
        inferTypes=["local/softmax"],
        loss=None,
        sample=True,
        sampleSize=args.sample_size,
        sampleGlobalLoss=True,
        beta=args.beta,
        device=device,
        tnorm="L",
    )
    return BuiltExample(graph, a, b, b_answer, program)


def build_reinforcement_example(args: argparse.Namespace, device: str) -> BuiltExample:
    torch.manual_seed(args.seed)
    graph, a, b, a_contains_b, b_answer = build_graph(args.expected_zeros)
    connect_sensors(a, b, a_contains_b, b_answer, args.features, device)
    program = ReinforcementProgram(
        graph,
        targets=[b_answer],
        reward_key="reward_function",
        decoder=counting_decoder,
        num_samples=args.num_samples,
        estimator=args.rl_estimator,
        poi=[a, b, b_answer],
        device=device,
    )
    return BuiltExample(graph, a, b, b_answer, program)


def counting_decoder(samples, targets, datanode, data_item):
    """Convert sampled class indices into generated labels for the reward function."""
    del datanode, data_item
    generated: list[str | int] = []
    for concept in targets:
        idx = samples.get(concept)
        if idx is None:
            continue
        names = getattr(concept, "enum", None)
        for value in idx.reshape(-1).tolist():
            value = int(value)
            if isinstance(names, (list, tuple)) and value < len(names):
                generated.append(str(names[value]))
            else:
                generated.append(value)
    return generated


def _logits_for_answer(program: Any, dataset: list[dict[str, Any]], b_answer: Concept, device: str) -> torch.Tensor:
    logits: list[torch.Tensor] = []
    for datanode in program.populate(dataset=dataset, device=device):
        for child in datanode.getChildDataNodes():
            value = child.getAttribute(b_answer)
            if torch.is_tensor(value) and value.numel() == 2:
                logits.append(value.detach().float().cpu().reshape(-1))
    if not logits:
        raise RuntimeError("No answer logits were populated; check the graph POI and sensors.")
    return torch.stack(logits, dim=0)


def evaluate_exact_count(
    program: Any,
    dataset: list[dict[str, Any]],
    b_answer: Concept,
    expected_zeros: int,
    eval_samples: int,
    device: str,
    seed: int,
) -> Stats:
    logits = _logits_for_answer(program, dataset, b_answer, device)
    probs = torch.softmax(logits, dim=-1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    draws = torch.multinomial(probs, eval_samples, replacement=True, generator=generator)
    zero_counts = (draws == 0).sum(dim=0)
    sampled_reward = float((zero_counts == expected_zeros).float().mean().item())
    zero_probs = probs[:, 0]
    return Stats(
        sampled_reward=sampled_reward,
        expected_zeros=float(zero_probs.sum().item()),
        argmax_zeros=int((probs.argmax(dim=-1) == 0).sum().item()),
        zero_probs=[float(x) for x in zero_probs.tolist()],
    )


def evaluate_program_reward(
    program: Any,
    dataset: list[dict[str, Any]],
    eval_samples: int,
    device: str,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    return float(program.evaluate_reward(dataset, num_samples=eval_samples, device=device))


def train_program(program: Any, dataset: list[dict[str, Any]], args: argparse.Namespace, device: str) -> None:
    program.train(
        dataset,
        train_epoch_num=args.epochs,
        Optim=lambda params: torch.optim.Adam(params, lr=args.lr),
        c_lr=args.lr,
        c_warmup_iters=0,
        device=device,
    )


def _zero_program_grads(program: Any) -> None:
    for module in (program.model, getattr(program, "cmodel", None)):
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.grad = None


def _gradient_summary(program: Any) -> tuple[float, float, int, list[tuple[float, float, str]]]:
    grad_rows = []
    squared_norm = 0.0
    max_abs = 0.0
    params_with_grad = 0
    for name, parameter in program.model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        norm = float(grad.norm().item())
        local_max = float(grad.abs().max().item()) if grad.numel() else 0.0
        squared_norm += norm * norm
        max_abs = max(max_abs, local_max)
        params_with_grad += 1
        grad_rows.append((norm, local_max, name))

    grad_rows.sort(reverse=True, key=lambda row: row[0])
    return squared_norm ** 0.5, max_abs, params_with_grad, grad_rows


def _print_gradient_summary(
    header: str,
    primary_name: str,
    primary_value: float,
    grad_l2: float,
    grad_max_abs: float,
    params_with_grad: int,
    grad_rows: list[tuple[float, float, str]],
    extra_lines: list[tuple[str, float]] | None = None,
    top_k: int = 4,
) -> None:
    print(f"\n{header}")
    print("-" * len(header))
    print(f"{primary_name:<29}: {primary_value:.6f}")
    for label, value in extra_lines or []:
        print(f"{label:<29}: {value:.6f}")
    print(f"{'model grad L2 norm':<29}: {grad_l2:.6e}")
    print(f"{'model grad max abs':<29}: {grad_max_abs:.6e}")
    print(f"{'parameters with gradients':<29}: {params_with_grad}")
    if grad_rows:
        print("top parameter grad norms     :")
        for norm, local_max, name in grad_rows[:top_k]:
            print(f"  {name}: norm={norm:.6e}, max={local_max:.6e}")
    else:
        print("top parameter grad norms     : none")


def sample_loss_gradient_report(
    program: Any,
    dataset: list[dict[str, Any]],
    device: str,
    title: str,
    top_k: int = 4,
) -> None:
    """Print model-parameter gradients from one sample-loss constraint step."""
    program.to(device)
    program.model.train()
    program.model.reset()
    program.cmodel.train()
    program.cmodel.reset()
    _zero_program_grads(program)

    mloss, _metric, *output = program.model(dataset[0])
    del mloss
    closs, *_ = program.cmodel(output[1])

    if torch.is_tensor(closs) and closs.requires_grad:
        closs.backward()

    grad_l2, grad_max_abs, params_with_grad, grad_rows = _gradient_summary(program)
    closs_value = float(closs.detach().item()) if torch.is_tensor(closs) else 0.0

    _print_gradient_summary(
        f"SampleLossProgram gradient report ({title})",
        "constraint loss",
        closs_value,
        grad_l2,
        grad_max_abs,
        params_with_grad,
        grad_rows,
        top_k=top_k,
    )

    _zero_program_grads(program)


def reinforcement_gradient_report(
    program: Any,
    dataset: list[dict[str, Any]],
    device: str,
    title: str,
    top_k: int = 4,
) -> None:
    """Print model-parameter gradients from one reinforcement reward step."""
    program.to(device)
    program.model.train()
    program.model.reset()
    _zero_program_grads(program)

    data_item = dataset[0]
    _mloss, _metric, datanode, _builder = program.model(data_item)
    reward_fn = program._resolve_reward_fn(data_item)
    loss, mean_reward = program.reinforcement_loss(datanode, reward_fn, data_item)

    if torch.is_tensor(loss) and loss.requires_grad:
        loss.backward()

    grad_l2, grad_max_abs, params_with_grad, grad_rows = _gradient_summary(program)
    loss_value = float(loss.detach().item()) if torch.is_tensor(loss) else 0.0
    reward_value = float(mean_reward) if mean_reward is not None else 0.0

    _print_gradient_summary(
        f"ReinforcementProgram gradient report ({title})",
        "reinforcement loss",
        loss_value,
        grad_l2,
        grad_max_abs,
        params_with_grad,
        grad_rows,
        extra_lines=[("sampled mean reward", reward_value)],
        top_k=top_k,
    )

    _zero_program_grads(program)


def print_result(name: str, before: Stats, after: Stats) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"sampled exact-count reward : {before.sampled_reward:.3f} -> {after.sampled_reward:.3f}")
    if before.program_reward is not None and after.program_reward is not None:
        print(f"program reward             : {before.program_reward:.3f} -> {after.program_reward:.3f}")
    print(f"expected # zero           : {before.expected_zeros:.2f} -> {after.expected_zeros:.2f}")
    print(f"argmax # zero             : {before.argmax_zeros} -> {after.argmax_zeros}")
    print("after P(zero)             : " + ", ".join(f"{p:.2f}" for p in after.zero_probs))


def reward_gain(before: Stats, after: Stats) -> float:
    return after.sampled_reward - before.sampled_reward


def print_comparison(sample_before: Stats, sample_after: Stats, rl_before: Stats, rl_after: Stats) -> None:
    sample_gain = reward_gain(sample_before, sample_after)
    rl_gain = reward_gain(rl_before, rl_after)
    final_gap = sample_after.sampled_reward - rl_after.sampled_reward

    print("\nWhich program was better?")
    print("-------------------------")
    if abs(final_gap) < 1e-6:
        print("Result: tie on final sampled exact-count reward.")
    elif final_gap > 0:
        print("Result: SampleLossProgram was better on this run.")
    else:
        print("Result: ReinforcementProgram was better on this run.")

    print(f"final reward gap          : {final_gap:+.3f} (sample loss - reinforcement)")
    print(f"reward gain, sample loss  : {sample_gain:+.3f}")
    print(f"reward gain, reinforcement: {rl_gain:+.3f}")

    if abs(final_gap) < 1e-6:
        print("Why: neither program has a measured advantage on this run. Both")
        print("improved the sampled exact-count reward, but the small evaluation")
        print("sample and stochastic training make this result effectively equal.")
        print("Methodologically, sample loss uses a direct graph constraint-loss")
        print("signal, while reinforcement uses generated labels scored by a")
        print("sampled 0/1 count reward.")
    elif final_gap > 0:
        print("Why: the sample-loss path receives a direct constraint-loss signal from")
        print("the graph, so it can move probabilities toward the exact-count target")
        print("with lower variance on this small toy task.")
        print("ReinforcementProgram learns from generated labels scored by a sampled")
        print("0/1 reward; that is more general, but the signal is sparser and noisier here.")
    else:
        print("Why: the reinforcement objective directly optimizes sampled satisfying")
        print("decodings, and in this run its reward estimator found better probability")
        print("moves than the sample-loss constraint approximation.")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dataset = create_dataset(args.seed, args.features, args.num_b, args.expected_zeros)

    print("=" * 72)
    print("Same task for both learners")
    print(f"constraint     : exactly {args.expected_zeros} of {args.num_b} b instances are zero")
    print(f"device         : {device}")
    print(f"epochs         : {args.epochs}")
    print("=" * 72)

    sample_example = build_sample_loss_example(args, device)
    before_sample = evaluate_exact_count(
        sample_example.program, dataset, sample_example.b_answer,
        args.expected_zeros, args.eval_samples, device, args.seed + 10,
    )
    torch.manual_seed(args.seed + 200)
    sample_loss_gradient_report(sample_example.program, dataset, device, "before training")
    torch.manual_seed(args.seed + 100)
    train_program(sample_example.program, dataset, args, device)
    torch.manual_seed(args.seed + 201)
    sample_loss_gradient_report(sample_example.program, dataset, device, "after training")
    after_sample = evaluate_exact_count(
        sample_example.program, dataset, sample_example.b_answer,
        args.expected_zeros, args.eval_samples, device, args.seed + 11,
    )

    rl_example = build_reinforcement_example(args, device)
    before_rl = evaluate_exact_count(
        rl_example.program, dataset, rl_example.b_answer,
        args.expected_zeros, args.eval_samples, device, args.seed + 10,
    )
    before_rl.program_reward = evaluate_program_reward(
        rl_example.program, dataset, args.eval_samples, device, args.seed + 20,
    )
    torch.manual_seed(args.seed + 300)
    reinforcement_gradient_report(rl_example.program, dataset, device, "before training")
    torch.manual_seed(args.seed + 100)
    train_program(rl_example.program, dataset, args, device)
    torch.manual_seed(args.seed + 301)
    reinforcement_gradient_report(rl_example.program, dataset, device, "after training")
    after_rl = evaluate_exact_count(
        rl_example.program, dataset, rl_example.b_answer,
        args.expected_zeros, args.eval_samples, device, args.seed + 11,
    )
    after_rl.program_reward = evaluate_program_reward(
        rl_example.program, dataset, args.eval_samples, device, args.seed + 21,
    )

    print_result("SampleLossProgram (primal-dual sample loss)", before_sample, after_sample)
    print_result("ReinforcementProgram (generated-output reward)", before_rl, after_rl)
    print_comparison(before_sample, after_sample, before_rl, after_rl)

    print("\nNote: exact-count tasks are judged by sampled reward and expected count.")
    print("Argmax count is reported as a useful diagnostic, but it is not the training objective.")


if __name__ == "__main__":
    main()
