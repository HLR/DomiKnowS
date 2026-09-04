import argparse
from pathlib import Path
import sys

import numpy as np
import torch

RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[2]
for path in (RUN_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.program import ReinforcementProgram

from graph import get_graph
from utils import create_dataset, TestTrainLearner, return_contain


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Machine Learning Experiment')
    parser.add_argument('--device', default='cpu', choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1'], help='Device to use')
    parser.add_argument('--atLeastL', default=False, type=bool, help='Use at least L constraint')
    parser.add_argument('--atMostL', default=False, type=bool, help='Use at most L constraint')
    parser.add_argument('--epoch', default=200, type=int, help='Number of training epochs')
    parser.add_argument('--expected_atLeastL', default=3, type=int, help='Expected value for at least L')
    parser.add_argument('--expected_atMostL', default=3, type=int, help='Expected value for at most L')
    parser.add_argument('--expected_value', default=0, type=int, help='Expected value')
    parser.add_argument('--N', default=10, type=int, help='N parameter (feature size)')
    parser.add_argument('--M', default=8, type=int, help='M parameter (number of b elements)')
    parser.add_argument('--model', default='sampling', type=str,
                        help='[sampling -> importance_weighted estimator | reinforce -> REINFORCE estimator]')
    parser.add_argument('--num_samples', default=16, type=int, help='Decodings sampled per step')
    parser.add_argument('--lr', default=5e-3, type=float, help='Learning rate')
    parser.add_argument('--visualize', action='store_true',
                        help='Launch the Flask step-by-step visualizer (training pauses each step)')
    parser.add_argument('--port', default=5000, type=int, help='Visualizer port')
    return parser.parse_args()


def _estimator_from_model(model: str) -> str:
    return 'reinforce' if str(model).lower() in ('reinforce', 'rl') else 'importance_weighted'


def setup_graph(args, a, b, a_contain_b, b_answer, device: str = 'cpu'):
    """Wire the sensors/learner so each ``b`` gets predicted answer logits."""
    a['index'] = ReaderSensor(keyword='a')
    b['index'] = ReaderSensor(keyword='b')
    b[a_contain_b] = EdgeSensor(b['index'], a['index'], relation=a_contain_b, forward=return_contain)

    model = TestTrainLearner(args.N)
    if hasattr(model, 'to'):
        model = model.to(device)
    b[b_answer] = ModuleLearner(a_contain_b, 'index', module=model, device=device)
    return model


def build_program(args, device: str = 'cpu'):
    """Build (but do not train) the reward-driven program and its dataset."""
    graph, a, b, a_contain_b, b_answer, reward_function = get_graph(args)
    dataset = create_dataset(args.N, args.M)

    answer_module = setup_graph(args, a, b, a_contain_b, b_answer, device=device)
    # Break symmetry so the initial predictions are not all identical.
    with torch.no_grad():
        for param in answer_module.parameters():
            param.add_(torch.randn_like(param) * 0.2)

    program = ReinforcementProgram(
        graph,
        targets=[b_answer],
        reward_function=reward_function,
        num_samples=args.num_samples,
        estimator=_estimator_from_model(args.model),
        poi=[a, b, b_answer],
        device=device,
        visualize=getattr(args, 'visualize', False),
        visualize_port=getattr(args, 'port', 5000),
    )
    return program, dataset, b_answer


def _target_text(args, class_name, m):
    if args.atLeastL and args.atMostL:
        bound = f"between {args.expected_atLeastL} and {args.expected_atMostL}"
    elif args.atMostL:
        bound = f"at most {args.expected_atMostL}"
    elif args.atLeastL:
        bound = f"at least {args.expected_atLeastL}"
    else:
        bound = f"exactly {args.expected_atLeastL}"
    return f"{bound} of the {m} b's should be '{class_name}'"


def _print_easy_summary(args, program, dataset, b_answer, baseline, trained, device):
    """Report what the model learned in terms meaningful for this reward."""
    counted = args.expected_value                 # class index the reward counts
    class_name = b_answer.enum[counted]
    probs, argmax_counts = [], {}
    for datanode in program.populate(dataset=dataset, device=device):
        for child in datanode.getChildDataNodes():
            logits = child.getAttribute(b_answer)
            if logits is None:
                continue
            p = torch.softmax(logits.float().reshape(-1), dim=-1)
            probs.append(float(p[counted]))
            pred = int(logits.argmax().item())
            argmax_counts[pred] = argmax_counts.get(pred, 0) + 1
    m = len(probs)
    expected_count = sum(probs)

    print("\n" + "=" * 64)
    print("Reinforcement training summary (easy example)")
    print("-" * 64)
    print(f"Target: {_target_text(args, class_name, m)}")
    print("Reward scores SAMPLED decodings (how many '%s' appear in a random" % class_name)
    print("draw), not the per-instance argmax.")
    print("-" * 64)
    print(f"  estimator             : {program.estimator}")
    print(f"  mean reward before    : {baseline:.4f}   (random/initial model)")
    print(f"  mean reward after     : {trained:.4f}   (after training)")
    print(f"  improvement           : {trained - baseline:+.4f}")
    print(f"  per-instance P('{class_name}')  : "
          + ", ".join(f"{x:.2f}" for x in probs))
    print(f"  expected # of '{class_name}'    : {expected_count:.2f}   "
          f"(want {args.expected_atLeastL} for the default exact target)")
    print(f"  argmax counts per class: {argmax_counts}")
    if (not args.atLeastL and not args.atMostL):
        k = args.expected_atLeastL
        print(f"  note: for an EXACT-{k} target the best independent policy sets each")
        print(f"        P('{class_name}') ~ {k}/{m} = {k / m:.2f} < 0.5, so every argmax is the")
        print(f"        other class ({argmax_counts}). That is expected — judge success by")
        print(f"        'expected # of {class_name}' ~ {k} and the rising mean reward, not argmax.")
    print("=" * 64)


def main(args: argparse.Namespace):
    np.random.seed(0)
    torch.manual_seed(0)

    device = 'cpu' if args.device in ('auto', 'cpu') else args.device

    program, dataset, b_answer = build_program(args, device=device)

    baseline = program.evaluate_reward(dataset, num_samples=200, device=device)
    program.train(
        dataset,
        train_epoch_num=args.epoch,
        Optim=lambda params: torch.optim.Adam(params, lr=args.lr),
        device=device,
    )
    trained = program.evaluate_reward(dataset, num_samples=200, device=device)

    _print_easy_summary(args, program, dataset, b_answer, baseline, trained, device)

    return program.graph, dataset


if __name__ == '__main__':
    main(parse_arguments())
