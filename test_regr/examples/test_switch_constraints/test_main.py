import pytest
import torch
from torch import nn
from typing import Any

from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import orL, existsL, ifL, notL, andL, atMostL, atLeastL, exactL
from domiknows.graph import EnumConcept
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import ModuleSensor, FunctionalSensor
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows import setProductionLogMode


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    
    def forward(self, x):
        out = self.linear(x)
        return out


@pytest.fixture
def setup_graph():
    """Setup graph and concepts for testing"""
    Graph.clear()
    Concept.clear()
    Relation.clear()
    
    setProductionLogMode(True)
    
    graph = Graph('main')
    with graph:
        root = Concept(name='root')
        x = root(name='x')
        y = root(name='y')
        
        root['input'] = ReaderSensor(keyword='input_vec')
        root[x] = ModuleSensor(root['input'], module=DummyModel())
        root[y] = ModuleSensor(root['input'], module=DummyModel())
    
    # Return the concepts in a dict so they can be used in extra_namespace_values
    yield graph, root, x, y


@pytest.fixture
def dataset():
    """Generate test dataset"""
    rng = torch.Generator()
    rng.manual_seed(0)

    random_constraints = [
        "andL(x, y)",
        "orL(x, y)",
        "ifL(x, y)",
        "andL(orL(x, y), y)"
    ]

    data = [
        {
            'input_vec': torch.randn((1, 2), generator=rng),
            'logic_str': random_constraints[
                torch.randint(len(random_constraints), size=(1,), generator=rng)
            ],
            'logic_label': torch.tensor([1.0]),
        }
        for _ in range(10)
    ]
    
    return data


def test_graph_compilation_and_training(setup_graph, dataset):
    """Test graph compilation and model training"""
    graph, root, x, y = setup_graph
    
    # Pass x and y in extra_namespace_values since they're not in the graph's varContext
    transformed_dataset = graph.compile_logic(
        dataset,
        logic_keyword='logic_str',
        logic_label_keyword='logic_label',
        extra_namespace_values={'x': x, 'y': y}
    )
    
    assert transformed_dataset is not None
    assert len(transformed_dataset) == len(dataset)
    
    program = InferenceProgram(
        graph,
        SolverModel,
        poi=[root, x, y, graph.constraint],
        device="cpu",
        tnorm='G'
    )
    
    program.train(
        transformed_dataset,
        epochs=2,
        lr=1e-4,
        c_warmup_iters=0,
        device="cpu"
    )
    
    assert program is not None