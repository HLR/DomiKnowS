import sys
import argparse
from typing import Any
import numpy as np
import torch
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, FunctionalSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor import Sensor
from domiknows.program.metric import MacroAverageTracker
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import PrimalDualProgram
from domiknows.program.model.pytorch import SolverModel

from utils import TestTrainLearner,return_contain,create_dataset,evaluate_model,train_model
sys.path.append('../../../../domiknows/')
from graph import get_graph

Sensor.clear()

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Machine Learning Experiment")
    parser.add_argument("--counting_tnorm", choices=["G", "P", "L", "SP"], default="G", help="The tnorm method to use for the counting constraints")
    parser.add_argument("--atLeastL", default=False, type=bool, help="Use at least L constraint")
    parser.add_argument("--atMostL", default=False, type=bool, help="Use at most L constraint")
    parser.add_argument("--epoch", default=500, type=int, help="Number of training epochs")
    parser.add_argument("--expected_atLeastL", default=3, type=int, help="Expected value for at least L")
    parser.add_argument("--expected_atMostL", default=3, type=int, help="Expected value for at most L")
    parser.add_argument("--expected_value", default=0, type=int, help="Expected value")
    parser.add_argument("--N", default=10, type=int, help="N parameter")
    parser.add_argument("--M", default=8, type=int, help="M parameter")
    return parser.parse_args()



def setup_graph(args: argparse.Namespace, a: Any, b: Any, a_contain_b: Any, b_answer: Any) -> None:
    a["index"] = ReaderSensor(keyword="a")
    b["index"] = ReaderSensor(keyword="b")
    b["temp_answer"] = ReaderSensor(keyword="label")
    b[a_contain_b] = EdgeSensor(b["index"], a["index"], relation=a_contain_b, forward=return_contain)
    b[b_answer] = ModuleLearner(a_contain_b, "index", module=TestTrainLearner(args.N), device="cpu")
    b[b_answer] = FunctionalSensor(a_contain_b, "temp_answer", forward=lambda _, label: label, label=True)

def main(args: argparse.Namespace):
    np.random.seed(0)
    torch.manual_seed(0)

    graph, a, b, a_contain_b, b_answer = get_graph(args)
    dataset = create_dataset(args.N, args.M)
    setup_graph(args, a, b, a_contain_b, b_answer)
    program = PrimalDualProgram(
        graph, SolverModel, poi=[a, b, b_answer],
        inferTypes=['local/argmax'],
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        beta=10, device='cpu', tnorm="L", counting_tnorm=args.counting_tnorm
    )
    expected_value = args.expected_value

    before_count = evaluate_model(program, dataset, b_answer).get(expected_value, 0)
    train_model(program, dataset, args.epoch)

    pass_test_case = True
    actual_count = evaluate_model(program, dataset, b_answer).get(expected_value, 0)


    if args.atLeastL:
        pass_test_case &= (actual_count >= args.expected_atLeastL)
    if args.atMostL:
        pass_test_case &= (actual_count <= args.expected_atMostL)
    if not args.atLeastL and not args.atMostL:
        pass_test_case &= (actual_count == args.expected_atLeastL)

    print(f"Test case {'PASSED' if pass_test_case else 'FAILED'}")
    print(f"expected_value, before_count, actual_count,pass_test_case): {expected_value, before_count, actual_count,pass_test_case}")
    return pass_test_case, before_count, actual_count

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
