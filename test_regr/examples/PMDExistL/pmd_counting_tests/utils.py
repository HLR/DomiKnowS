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

    # Much higher learning rates for constraint-only phase
    lr = 5e-3 if constr_loss_only else 1e-4
    opt = torch.optim.SGD(program.model.parameters(), lr=lr, momentum=0.9)
    copt = torch.optim.SGD(program.cmodel.parameters(), lr=lr, momentum=0.9)

    constraint_loss_zero_count = 0
    no_gradient_count = 0

    for epoch in tqdm(range(num_epochs), desc="Training with PMD"):
        epoch_loss = 0.0
        num_steps = 0
        
        for data in dataset:
            opt.zero_grad(set_to_none=True)
            copt.zero_grad(set_to_none=True)
            
            mloss, _, *output = program.model(data)
            closs, *_ = program.cmodel(output[1])

            if constr_loss_only:
                # Use constraint loss to update model parameters
                loss = closs if torch.is_tensor(closs) else None
                
                # Check if constraint loss exists and is valid
                if loss is None:
                    if epoch == 0:
                        print("[WARN] Constraint loss is None - constraints may not be active")
                    continue
                
                if not torch.isfinite(loss):
                    if epoch == 0:
                        print(f"[WARN] Non-finite constraint loss: {loss.item()}")
                    continue
                
                # Check if constraint loss is non-zero
                if abs(loss.item()) < 1e-8:
                    constraint_loss_zero_count += 1
                    if constraint_loss_zero_count == 1:
                        print(f"[WARN] Constraint loss is near zero: {loss.item()} - constraints may already be satisfied or not properly configured")
                    continue
                
                # Very strong scaling for constraint loss
                loss = loss * 50.0
            else:
                loss = mloss
                if not torch.isfinite(loss):
                    continue

            loss.backward()
            
            # Check if gradients were actually computed
            if constr_loss_only:
                has_model_grads = any(
                    p.grad is not None and p.grad.abs().sum() > 1e-8
                    for p in program.model.parameters() if p.requires_grad
                )
                
                if not has_model_grads:
                    no_gradient_count += 1
                    if no_gradient_count == 1:
                        print("[WARN] No gradients flowing to model parameters from constraint loss")
                        print("      This means constraints are not connected to model outputs")
                    continue
            
            # Aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(program.model.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(program.cmodel.parameters(), max_norm=5.0)

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
            
            epoch_loss += loss.item()
            num_steps += 1
    
    # Summary at end of training
    if constr_loss_only:
        if constraint_loss_zero_count > 0:
            print(f"[INFO] Constraint loss was zero for {constraint_loss_zero_count}/{num_epochs * len(dataset)} steps")
        if no_gradient_count > 0:
            print(f"[INFO] No gradients to model for {no_gradient_count}/{num_epochs * len(dataset)} steps")


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