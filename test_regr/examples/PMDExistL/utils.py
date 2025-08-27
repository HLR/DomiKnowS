import json
import random
from typing import List, Dict, Any
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from domiknows.sensor.pytorch.learners import TorchLearner
from domiknows.program.model.base import Mode
from domiknows.program.lossprogram import PrimalDualProgram
from sklearn.model_selection import train_test_split


class DummyLearner(TorchLearner):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack((torch.ones(len(x)) * 4, torch.ones(len(x)) * 6), dim=-1)


class TestTrainLearner(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Linear(input_size, 2)
        )

    def forward(self, _, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def return_contain(b: torch.Tensor, _: Any) -> torch.Tensor:
    return torch.ones(len(b)).unsqueeze(-1)


def create_dataset(N: int, M: int) -> List[Dict[str, Any]]:
    return [{
        "a": [0],
        "b": [((np.random.rand(N) - np.random.rand(N))).tolist() for _ in range(M)],
        "label": [1] * M
    }]


def create_dataset_relation(args, N: int, M: int, K: int, load_data=False) -> List[Dict[str, Any]]:
    def create_scene(num_all_objs, total_numbers_each_obj):
        generate_objects = [(np.random.rand(total_numbers_each_obj) - np.random.rand(total_numbers_each_obj)).tolist() for _ in range(num_all_objs)]

        def select_number_condition():
            select_condition = random.randint(0, 1)
            if select_condition == 0:
                return lambda a: np.sum(a) > 0, "is_cond1"
            if select_condition == 1:
                return lambda a: np.sum(np.abs(a)) > 0.2, "is_cond2"

        def select_relation_condition():
            select_condition = random.randint(0, 1)
            if select_condition == 0:
                return lambda a, b: a[0] * b[0] >= 0, "is_relation1"
            if select_condition == 1:
                return lambda a, b: a[0] * b[0] < 0, "is_relation2"

        condition_label = False
        cond1_condition, cond_x = select_number_condition()
        cond2_condition, cond_y = select_number_condition()
        relation_condition, rel_text = select_relation_condition()
        for i in range(num_all_objs):
            for j in range(i + 1, num_all_objs):
                obj1 = generate_objects[i]
                obj2 = generate_objects[j]
                cond1 = bool(cond1_condition(obj1))
                cond2 = bool(cond2_condition(obj2))
                rel = bool(relation_condition(obj1, obj2))
                # print(cond1, cond2, rel, j)
                condition_label |= (cond1 & cond2 & rel)

        if not args.constraint_2_existL:
            logic_str = f"existsL({cond_x}('x'), {rel_text}('rel1', path=('x', obj1.reversed)), {cond_y}('y', path=('rel1', obj2)))"
        else:
            logic_str = f"existsL({cond_x}('x'), {rel_text}('rel1', path=('x', obj1.reversed)), existsL({cond_y}('y', path=('rel1', obj2))))"

        logic_order = [cond_x, rel_text, cond_y]

        return {
            "all_obj": [0],
            "obj_index": generate_objects,
            "obj_emb": generate_objects,
            "condition_label": [condition_label],
            "logic_str": logic_str,
            "logic_order": logic_order
        }

    if load_data:
        with open("dataset/train.json", 'rb') as f:
            train = json.load(f)
        with open("dataset/test.json", 'rb') as f:
            test = json.load(f)
    else:
        dataset = [create_scene(M, K) for _ in range(N)]
        train, test = train_test_split(dataset, test_size=0.2)

    return train, test, [data["condition_label"][0] for data in train], [data["condition_label"][0] for data in test]


def train_model(program: PrimalDualProgram, dataset: List[Dict[str, Any]],
                num_epochs: int, constr_loss_only: bool = False) -> None:
    program.model.train()
    program.model.reset()
    program.cmodel.train()
    program.cmodel.reset()
    program.model.mode(Mode.TRAIN)

    opt = torch.optim.Adam(program.model.parameters(), lr=1e-2)
    copt = torch.optim.Adam(program.cmodel.parameters(), lr=1e-3)

    for _ in tqdm(range(num_epochs), desc="Training with PMD"):
        for data in dataset:
            opt.zero_grad()
            copt.zero_grad()
            mloss, _, *output = program.model(data)
            closs, *_ = program.cmodel(output[1])

            if constr_loss_only:
                loss = mloss * 0 + (closs if torch.is_tensor(closs) else 0)
            else:
                loss = mloss

            if loss.item() < 0:
                print("Negative loss", loss.item())
                break
            if loss:
                loss.backward()
                opt.step()
                copt.step()


def evaluate_model(program: PrimalDualProgram, dataset: List[Dict[str, Any]], b_answer: Any) -> Dict[int, int]:
    program.model.eval()
    program.model.reset()
    program.cmodel.eval()
    program.cmodel.reset()
    program.model.mode(Mode.TEST)

    final_result_count = {}
    for datanode in program.populate(dataset=dataset):
        for child in datanode.getChildDataNodes():
            pred = child.getAttribute(b_answer, 'local/argmax').argmax().item()
            final_result_count[pred] = final_result_count.get(pred, 0) + 1
    return final_result_count
