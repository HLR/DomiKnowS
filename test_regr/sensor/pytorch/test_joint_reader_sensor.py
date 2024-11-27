import sys
import argparse
from typing import Any, List, Tuple
import numpy as np
import torch
import random
from domiknows.sensor.pytorch.sensors import ReaderSensor, JointReaderSensor
from domiknows.program import SolverPOIProgram
from domiknows.sensor import Sensor
from domiknows.program.metric import MacroAverageTracker
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.model.pytorch import SolverModel

# from utils import TestTrainLearner, return_contain, create_dataset, evaluate_model, train_model

sys.path.append('../../../../domiknows/')
# from graph import get_graph

Sensor.clear()


def get_graph():
    from domiknows.graph import Graph, Concept, Relation
    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph('subject') as graph:
        subject = Concept(name='name')

    return graph, subject


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Machine Learning Experiment")
    parser.add_argument("--n", default=100, type=int, help="N parameter")
    return parser.parse_args()


def setup_graph(subject: Any, input_keys: List[str], output_key: List[str]) -> None:
    # This is a normal way to read the value from dataset into the graph
    for i, key in enumerate(output_key):
        input_key = input_keys[i]
        subject[input_key] = ReaderSensor(keyword=key)

    # We want to not use the loop and use the following function
    # Note that the *input_key is the same as [key1, key2, key3] if input_key = [obj1, obj2, obj3]
    # subject[*input_keys] = JointReaderSensor(keyword=tuple(output_key))


def evaluate(program: SolverPOIProgram, dataset: Any, key_check: Any):
    pass_test = True
    idx = 0
    for datanode in program.populate(dataset=dataset):
        for key in key_check:
            read_val = datanode.getAttribute(key)
            if isinstance(read_val, list):
                read_val = read_val[0]
            pass_test = pass_test and (read_val == dataset[idx][key])

    return pass_test


def create_dataset(keys: List[str]):
    return [{key: random.randint(0, 10**6) for key in keys}]


def main(args: argparse.Namespace):
    np.random.seed(0)
    torch.manual_seed(0)

    graph, subject = get_graph()
    # dataset = create_dataset(args.N, args.M)
    key_check = [f"test{i}" for i in range(args.n)]
    setup_graph(subject, input_keys=key_check, output_key=key_check)
    dataset = create_dataset(key_check)

    program = SolverPOIProgram(
        graph, poi=[subject],
        inferTypes=['local/argmax'],
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        device='cpu'
    )

    pass_test = evaluate(program, dataset=dataset, key_check=key_check)

    if pass_test:
        print("PASS READING MULTIPLE VALUES BY JOINT READER SENSOR")
    else:
        print("FAIL READING MULTIPLE VALUE BY JOINT READER SENSOR")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
