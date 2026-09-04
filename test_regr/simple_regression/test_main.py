import pytest
import torch
from domiknows.graph import Graph, Concept
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.program import SolverPOIProgram
from domiknows.program.loss import NBMSELoss
from domiknows.program.metric import MacroAverageTracker
from domiknows import setProductionLogMode


class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        pred = self.linear(x)
        return pred


@pytest.fixture
def graph():
    setProductionLogMode(True)
    
    with Graph(name='global') as g:
        number = Concept(name='number')
        x = number(name='x')
        y = number(name='y')
    
    return g, number, x, y


@pytest.fixture
def program(graph):
    g, number, x, y = graph
    
    number['x'] = ReaderSensor(keyword='x')
    number[y] = ModuleLearner('x', module=LinearRegression())
    number[y] = ReaderSensor(keyword='y', label=True)
    
    prog = SolverPOIProgram(
        g,
        poi=(number, x, y),
        inferTypes=['local/argmax'],
        loss=MacroAverageTracker(NBMSELoss())
    )
    
    return prog


def linear_regression_generator(n, k=1, noise_std=0.1):
    for _ in range(n):
        x = torch.randn(1, 1)
        noise = noise_std * torch.randn(1, 1)
        y = k * x + noise
        yield {'x': x.float(), 'y': y.float()}


def test_training(program):
    training_data = list(linear_regression_generator(100))
    
    program.train(
        training_set=training_data,
        device='cpu',
        train_epoch_num=5,
        test_every_epoch=True,
        Optim=torch.optim.Adam,
    )
    
    assert program is not None


def test_inference(program):
    training_data = list(linear_regression_generator(100))
    program.train(
        training_set=training_data,
        device='cpu',
        train_epoch_num=5,
        test_every_epoch=False,
        Optim=torch.optim.Adam,
    )
    
    valid_data = list(linear_regression_generator(10))
    
    for sample in valid_data:
        node = program.populate_one(sample, grad=True)
        attributes = node.getAttributes()
        
        assert attributes is not None
        assert 'x' in attributes or 'y' in attributes


def test_data_generator():
    data = list(linear_regression_generator(10, k=2, noise_std=0.05))
    
    assert len(data) == 10
    
    for sample in data:
        assert 'x' in sample
        assert 'y' in sample
        assert sample['x'].shape == (1, 1)
        assert sample['y'].shape == (1, 1)


def test_linear_regression_module():
    model = LinearRegression()
    x = torch.randn(1, 1)
    output = model(x)
    
    assert output.shape == (1, 1)