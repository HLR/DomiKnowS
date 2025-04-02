import torch
from torch.nn import functional as F
from functools import partial
import pickle
import os
import argparse

from training import TrainingConfig, loop
from data import get_readers


def weighted_sum(logits: torch.Tensor, device: str, temp: float = 1.0) -> torch.Tensor:
    # logits: (batch_size, 10)

    probs = F.softmax(logits / temp, dim=-1) # (batch_size, 10)
    
    zero_to_nine = torch.arange(10).to(device).unsqueeze(0) # (1, 10)
    return torch.sum(probs * zero_to_nine, dim=-1)


class Cat2IntSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.argmax(input, dim=1).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.unsqueeze(1).repeat(1, 10)

def ste_full(logits: torch.Tensor, device: str, temp: float = 1.0) -> torch.Tensor:
    # logits: (batch_size, 10)
    
    probs = F.softmax(logits / temp, dim=-1) # (batch_size, 10)
    
    return Cat2IntSTE.apply(probs)


class OneHotSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.nn.functional.one_hot( # (batch_size, 10)
            torch.argmax(input, dim=1), # (batch_size,)
            num_classes=input.shape[1]
        ).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def ste_separate(
        logits: torch.Tensor,
        device: str,
        add_gumbel_noise: bool = False,
        temp: float = 1.0
    ) -> torch.Tensor:

    # setting add_gumbel_noise = True corresponds to https://arxiv.org/pdf/2109.08512

    # logits: (batch_size, 10)

    if not add_gumbel_noise:
        probs = F.softmax(logits / temp, dim=-1) # (batch_size, 10)

        # using custom autograd function for STE
        # onehot = OneHotSTE.apply(probs) # (batch_size, 10)

        # using pytorch implementation of STE (see gumbel_softmax in functional.py)
        argmax_idx = probs.max(-1, keepdim=True)[1] # (batch_size,)
        onehot_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(-1, argmax_idx, 1.0) # (batch_size, 10)
        onehot = onehot_hard - probs.detach() + probs # (batch_size, 10)
    
    else:
        onehot = F.gumbel_softmax(
            logits,
            tau=temp,
            hard=True
        )

    # one-hot to integer
    zero_to_nine = torch.arange(10).to(device).unsqueeze(0) # (1, 10)

    selected_digit = torch.sum(onehot * zero_to_nine, dim=-1) # (batch_size,)

    return selected_digit


approximations = {
    'ste_separate_gumbel': partial(ste_separate, add_gumbel_noise=True),
    'ste_separate': partial(ste_separate, add_gumbel_noise=False),
    'ste_full': ste_full,
    'weighted_sum': weighted_sum
}

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--device', type=str, required=True)
    args.add_argument('--output_dir', type=str, default='results')
    args = args.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device

    print(f'training using device {device}')

    trainloader, trainloader_mini, validloader, testloader = get_readers(40_000)

    for approx_name, approx_func in approximations.items():
        print(approx_name)
        config = TrainingConfig(
            approx_select=approx_func,
            device=device,
            batch_size=64,
            num_epochs=20,
            operations=['summation', 'subtraction', 'multiplication', 'division'],
            lr=1e-3,
        )

        results = loop(config, trainloader, validloader)

        with open(f'{args.output_dir}/{approx_name}.pkl', 'wb') as f:
            pickle.dump(results, f)
