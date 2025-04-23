from domiknows.graph import Graph, Concept
from domiknows.sensor.pytorch.sensors import FunctionalSensor, ReaderSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.program import SolverPOIProgram
from domiknows.program.loss import MSELoss
from domiknows.program.metric import MacroAverageTracker
from domiknows import setProductionLogMode
import torch
from torch import nn
from torch.nn import functional as F
import argparse

setProductionLogMode(True)

from model import Net
from data import make_sum_graph, get_readers
from utils import batch_iterator, call_once


# parse arguments
argparser = argparse.ArgumentParser()

argparser.add_argument(
    '--op',
    type=str,
    default='summation',
    choices=['summation', 'multiplication', 'subtraction', 'division'],
)

argparser.add_argument(
    '--device',
    type=str,
    default='cpu',
    choices=['cpu', 'cuda'],
)

argparser.add_argument(
    '--epochs',
    type=int,
    default=10,
)

args = argparser.parse_args()

print('run parameters:')
print(args.__dict__)

# util for evaluations
@torch.no_grad()
def get_accuracy_iter(
        dataloader: torch.utils.data.DataLoader,
        net: torch.nn.Module,
        device: str
    ) -> tuple[float, int]:
    correctness_all = []

    for batch_data in batch_iterator(dataloader, batch_size=32):
        # collate batch data
        digits_batch = torch.stack([x['pixels_1'] for x in batch_data], dim=0).to(device)
        digit_labels_batch = torch.stack([x['digit'] for x in batch_data], dim=0).to(device) # (batch_size, 2)
        digit_labels_batch = digit_labels_batch[:, 0]

        # forward pass for all digit predictions
        digits_batch_flat = digits_batch.view(-1, 28*28)
        logits_batch_flat = net(digits_batch_flat)

        logits_batch = logits_batch_flat.view(-1, 10)
        
        # get accuracies
        digit_preds = logits_batch.argmax(dim=-1) # (batch_size, 2)
        
        batch_correctness = (digit_preds == digit_labels_batch).flatten() # (batch_size * 2,)

        correctness_all.extend(batch_correctness.cpu().tolist())
    
    return (sum(correctness_all) / len(correctness_all), len(correctness_all))


OPERATIONS = {
    'summation': lambda x, y: x + y,
    'multiplication': lambda x, y: x * y,
    'subtraction': lambda x, y: x - y,
    'division': lambda x, y: x / (y + 1e-4)
}


# module for converting continuous to discrete
class Discretized(nn.Module):
    """
    Wrapper for `target_module` that discretizes categorical outputs into an integer
    using Gumbel-Softmax and straight-through estimation.
    """

    def __init__(
			self,
			target_module: nn.Module,
            tau: float = 2.0,
            device: str = args.device
        ):
        """
        Args:
            target_module (nn.Module): The module to discretize; expects logits with shape (batch_size, num_classes).
            tau (float): Temperature parameter for Gumbel-Softmax.
            device (str): Device to use.
        """
        super().__init__()

        self.tau = tau
        self.target_module = target_module
        self.device = device

    def forward(self, *args) -> torch.Tensor:
        """
        Forward pass that discretizes the output of the target module.
        Produces a tensor of size (batch_size,) with integer values in range [0, num_classes - 1].

        Args:
            *args: Arguments to pass to the target module.
        """

        logits = self.target_module(*args)

        onehot = F.gumbel_softmax(
            logits,
            tau=self.tau,
            hard=True
        )

        integer_range = torch.arange(onehot.shape[1], device=self.device)

        return torch.sum(onehot * integer_range, dim=-1)

    def get_undiscretized_module(self) -> nn.Module:
        """
        Gets the original, unwrapped module.
        """

        return self.target_module


# build graph
with Graph(name='global') as graph:
	image = Concept(name='image')

	digit_1 = image(name='digit_1')
	digit_2 = image(name='digit_2')

	arith_result = Concept(name='arith_result')


# build program
net = Discretized(Net()).to(args.device)

image['pixels_1'] = ReaderSensor(keyword='pixels_1')
image['pixels_2'] = ReaderSensor(keyword='pixels_2')

image[digit_1] = ModuleLearner('pixels_1', module=net)
image[digit_2] = ModuleLearner('pixels_2', module=net)

image[arith_result] = FunctionalSensor(
    image[digit_1],
    image[digit_2],
    forward=OPERATIONS[args.op]
)

image[arith_result] = ReaderSensor(keyword=args.op, label=True)

program = SolverPOIProgram(
	graph,
	poi=(image, digit_1, digit_2, arith_result),
	inferTypes=['local/argmax'],
	loss=MacroAverageTracker(MSELoss())
)


# load data
trainloader, _, validloader, _ = get_readers(
    num_train=10_000,
    sample_maker=make_sum_graph
)

# for division, filter for divide-by-zero cases
# TODO: make wrapper around DataLoader since this doesn't retain e.g., shuffling
if args.op == 'division':
    trainloader = list(filter(
        lambda x: x['division'] is not None,
        trainloader
    ))

    validloader = list(filter(
        lambda x: x['division'] is not None,
        validloader
    ))


# run training
@call_once
def _optimizer(params):
    return torch.optim.Adam(params, lr=1e-3)

for epoch_idx in range(args.epochs):
    program.train(
        trainloader,
        train_epoch_num=1,
        Optim=_optimizer,
        device=args.device,
        c_warmup_iters=0
    )

    print('train accuracy:', get_accuracy_iter(
        trainloader,
        net.get_undiscretized_module(),
        args.device,
    ))

    print('validation accuracy:', get_accuracy_iter(
        validloader,
        net.get_undiscretized_module(),
        args.device,
    ))
