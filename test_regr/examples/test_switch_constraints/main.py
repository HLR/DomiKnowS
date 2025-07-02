from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import orL, existsL, ifL, notL, andL, atMostL, atLeastL, exactL
from domiknows.graph import EnumConcept
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import ModuleSensor, FunctionalSensor
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows import setProductionLogMode

setProductionLogMode(True)

from typing import Any
import torch
from torch import nn


Graph.clear()
Concept.clear()
Relation.clear()

with Graph('main') as graph:
    root = Concept(name='root')

    x = root(name='x')
    y = root(name='y')

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    
    def forward(self, x):
        out = self.linear(x)

        return out

root['input'] = ReaderSensor(keyword='input_vec')

root[x] = ModuleSensor(root['input'], module=DummyModel())
root[y] = ModuleSensor(root['input'], module=DummyModel())

def _get_dataset(size: int = 5000) -> list[dict[str, Any]]:
    rng = torch.Generator()
    rng.manual_seed(0)

    random_constraints = [
        "andL(x, y)",
        "orL(x, y)",
        "ifL(x, y)",
        "andL(orL(x, y), y)"
    ]

    dataset = [
        {
            'input_vec': torch.randn((1, 2), generator=rng),
            'logic_str': random_constraints[
                torch.randint(len(random_constraints), size=(1,), generator=rng)
            ],
            'logic_label': torch.tensor([1.0]),
        }
        for _ in range(size)
    ]

    return dataset

# original_dataset format:
# [
#     {
#         'input_vec': torch.tensor([[1.0, 1.0]]),
#         'logic_str': 'andL(x, y)',
#         'logic_label': torch.tensor([1.0]),
#     },
#     ...
# ]

original_dataset = _get_dataset(size=10_000)

transformed_dataset = graph.compile_logic(
    original_dataset,
    logic_keyword='logic_str',
    logic_label_keyword='logic_label',
)

program = InferenceProgram(
    graph,
    SolverModel,
    poi=[root, x, y, graph.constraint],
    device="cpu",
    tnorm='G'
)

program.train(
    transformed_dataset,
    epochs=10,
    lr=1e-4,
    c_warmup_iters=0,
    device="cpu"
)
