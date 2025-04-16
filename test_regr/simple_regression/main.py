from domiknows.graph import Graph, Concept
from domiknows.sensor.pytorch.sensors import FunctionalSensor, ReaderSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.program import SolverPOIProgram
from domiknows.program.loss import NBCrossEntropyLoss, MSELoss
from domiknows.program.metric import MacroAverageTracker
from domiknows import setProductionLogMode
from tqdm import tqdm

setProductionLogMode(True)

import torch

# pytorch lin reg. model
class LinearRegression(torch.nn.Module):
	def __init__(self):
		super(LinearRegression, self).__init__()
		self.linear = torch.nn.Linear(1, 1)

	def forward(self, x):
		pred = self.linear(x)
		return pred.squeeze(0)

# graph
with Graph(name='global') as graph:
	number = Concept(name='number')
	
	x = number(name='x')
	y = number(name='y')


# program
number['x'] = ReaderSensor(keyword='x')

number[y] = ModuleLearner('x', module=LinearRegression())
number[y] = ReaderSensor(keyword='y', label=True)

program = SolverPOIProgram(
	graph,
	poi=(number, x, y),
	inferTypes=['local/argmax'],
	loss=MacroAverageTracker(MSELoss())
)

# data generation
def linear_regression_generator(n, k=1, noise_std=0.1):
    for _ in range(n):
        x = torch.randn(1, 1)

        noise = noise_std * torch.randn(1, 1)
        y = k * x + noise

        yield {'x': x.float(), 'y': y.float()}

# training
program.train(
	training_set=list(linear_regression_generator(1000)),
	device='cpu',
	train_epoch_num=10,
	test_every_epoch=True,
	Optim=torch.optim.Adam,
)

# inference
valid_data = linear_regression_generator(1000)
for sample in valid_data:
	node = program.populate_one(sample, grad=True)

	print(node.getAttributes())

	# model prediction => tensor([[1.0231]])
	# node.getAttributes() =>
	# {'x': tensor([[-1.4816]]), '<y>': [-0.023057222366333008, 1.023057222366333], '<y>/label': tensor([-1.4350])}

# warning: UserWarning: Using a target size (torch.Size([1, 1])) that is different to the input size (torch.Size([1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
