import random
from typing import List, Dict, Any
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from domiknows.sensor.pytorch.learners import TorchLearner
from domiknows.program.model.base import Mode
from domiknows.program.lossprogram import PrimalDualProgram


class DummyLearner(TorchLearner):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack((torch.ones(len(x)) * 4, torch.ones(len(x)) * 6), dim=-1)


class TestTrainLearner(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
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


def train_model(program: PrimalDualProgram, dataset: List[Dict[str, Any]],
                num_epochs: int, constr_loss_only: bool = False) -> None:
    program.model.train()
    program.model.reset()
    program.cmodel.train()
    program.cmodel.reset()
    program.model.mode(Mode.TRAIN)

    # Higher learning rates for constraint-only phase
    lr = 1e-3 if constr_loss_only else 1e-4
    opt = torch.optim.AdamW(program.model.parameters(), lr=lr, weight_decay=1e-5)
    copt = torch.optim.AdamW(program.cmodel.parameters(), lr=lr, weight_decay=1e-5)

    msched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
    csched = torch.optim.lr_scheduler.CosineAnnealingLR(copt, T_max=num_epochs)

    ACCUM_STEPS = 4

    for _ in tqdm(range(num_epochs), desc="Training with PMD"):
        opt.zero_grad(set_to_none=True)
        copt.zero_grad(set_to_none=True)

        for step, data in enumerate(dataset, start=1):
            mloss, _, *output = program.model(data)
            closs, *_ = program.cmodel(output[1])

            if constr_loss_only:
                # Use constraint loss to update model parameters
                loss = closs if torch.is_tensor(closs) else None
                if loss is None or not torch.isfinite(loss):
                    continue
                
                # Scale up constraint loss for stronger signal
                loss = loss * 10.0
            else:
                loss = mloss
                if not torch.isfinite(loss):
                    continue

            (loss / ACCUM_STEPS).backward()

            if step % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(program.model.parameters(), max_norm=2.0)
                torch.nn.utils.clip_grad_norm_(program.cmodel.parameters(), max_norm=2.0)

                if constr_loss_only:
                    # Always step both optimizers in constraint phase
                    opt.step()
                    copt.step()
                else:
                    opt.step()
                    has_cmodel_grads = any(
                        p.grad is not None and p.grad.abs().sum() > 0 
                        for p in program.cmodel.parameters() if p.requires_grad
                    )
                    if has_cmodel_grads:
                        copt.step()
                
                opt.zero_grad(set_to_none=True)
                copt.zero_grad(set_to_none=True)

        msched.step()
        csched.step()


def evaluate_model(program: PrimalDualProgram, dataset: List[Dict[str, Any]], b_answer: Any) -> Dict[int, int]:
    program.model.eval()
    program.model.reset()
    program.cmodel.eval()
    program.cmodel.reset()
    program.model.mode(Mode.TEST)

    final_result_count: Dict[int, int] = {}
    with torch.no_grad():
        for datanode in program.populate(dataset=dataset):
            for child in datanode.getChildDataNodes():
                pred = child.getAttribute(b_answer, 'local/argmax').argmax().item()
                final_result_count[pred] = final_result_count.get(pred, 0) + 1
    return final_result_count