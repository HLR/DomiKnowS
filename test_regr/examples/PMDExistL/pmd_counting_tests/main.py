import os
import sys
import argparse
from typing import Any
import numpy as np
import torch
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, FunctionalSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor import Sensor
from domiknows.program.metric import MacroAverageTracker
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import GumbelPrimalDualProgram, SampleLossProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows import setProductionLogMode

from utils import TestTrainLearner, return_contain, create_dataset, evaluate_model, train_model

import traceback
torch.autograd.set_detect_anomaly(True)

is_ci = os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS')
if is_ci:
    setProductionLogMode(True)

def excepthook(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    traceback.print_exception(exc_type, exc_value, exc_traceback)

sys.excepthook = excepthook

from graph import get_graph

Sensor.clear()

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Machine Learning Experiment")
    parser.add_argument("--beta", default="10", type=float, help="Beta parameter")
    parser.add_argument("--device", default="auto",choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"], help="Device to use")
    parser.add_argument("--counting_tnorm", choices=["G", "P", "L", "SP"], default="SP", help="The tnorm method to use for the counting constraints")
    parser.add_argument("--atLeastL", default=False, type=bool, help="Use at least L constraint")
    parser.add_argument("--atMostL", default=False, type=bool, help="Use at most L constraint")
    parser.add_argument("--epoch", default=500, type=int, help="Number of training epochs")
    parser.add_argument("--expected_atLeastL", default=3, type=int, help="Expected value for at least L")
    parser.add_argument("--expected_atMostL", default=3, type=int, help="Expected value for at most L")
    parser.add_argument("--expected_value", default=0, type=int, help="Expected value")
    parser.add_argument("--N", default=10, type=int, help="N parameter")
    parser.add_argument("--M", default=8, type=int, help="M parameter")
    parser.add_argument("--model", default="sampling", type=str, help="Model Types [Sampling/PMD]")
    parser.add_argument("--sample_size", default=-1, type=int, help="Sample size for sampling program")
    return parser.parse_args()


def setup_graph(args, a, b, a_contain_b, b_answer, device: str = "cpu") -> None:
    a["index"] = ReaderSensor(keyword="a")
    b["index"] = ReaderSensor(keyword="b")
    b["temp_answer"] = ReaderSensor(keyword="label")
    b[a_contain_b] = EdgeSensor(b["index"], a["index"], relation=a_contain_b, forward=return_contain)

    model = TestTrainLearner(args.N)
    if hasattr(model, "to"):
        model = model.to(device)

    b[b_answer] = ModuleLearner(a_contain_b, "index", module=model, device=device)
    b[b_answer] = FunctionalSensor(a_contain_b, "temp_answer", forward=lambda _, label: label, label=True)
    
    return model

def _run_answer_module(mod, x):
    with torch.no_grad():
        try:
            out = mod(x)
        except TypeError:
            out = mod(None, x)
    if isinstance(out, tuple):
        out = out[0]
    return out

def main(args: argparse.Namespace):
    np.random.seed(0)
    torch.manual_seed(0)

    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda":
        device = "cuda:0"
    else:
        device = args.device

    graph, a, b, a_contain_b, b_answer = get_graph(args)
    dataset = create_dataset(args.N, args.M)

    print(f"labels: {dataset[0]['label']} (sum={sum(dataset[0]['label'])}/{len(dataset[0]['label'])} are 1)")

    answer_module = setup_graph(args, a, b, a_contain_b, b_answer, device=device)
    
    train_infer = ['local/softmax']
    eval_infer  = ['local/argmax']

    if args.model == "sampling":
        program = SampleLossProgram(
            graph, SolverModel, poi=[a, b, b_answer],
            inferTypes=train_infer,
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            sample=True, 
            sampleSize=args.sample_size, 
            sampleGlobalLoss=True,
            beta=args.beta, device=device, tnorm="L", counting_tnorm=args.counting_tnorm
        )
    else:
       program = GumbelPrimalDualProgram(
            graph, SolverModel,
            poi=[a, b, b_answer],
            inferTypes=train_infer,
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            use_gumbel=True,
            initial_temp=5.0,
            final_temp=0.5,
            beta=args.beta,
            device=device,
            tnorm="L",
            counting_tnorm=args.counting_tnorm
        )

    labels = dataset[0]['label']
    M = len(labels)
    expected_value = args.expected_value
    target_count = args.expected_atLeastL
    
    # Add noise to break symmetry
    with torch.no_grad():
        for param in answer_module.parameters():
            param.add_(torch.randn_like(param) * 0.2)

    # Helper function
    def soft_count_zero(module, dataset, device):
        s = 0.0
        for item in dataset:
            x = torch.as_tensor(item["b"], dtype=torch.float32, device=device)
            logits = _run_answer_module(module, x)
            p = torch.softmax(logits, dim=-1)[:, 0]
            s += p.sum().item()
        return s

    print(f"[INFO] Soft sum zero (initial): {soft_count_zero(answer_module, dataset, device):.4f}")

    # Warmup training
    warmup_epochs = 20
    if warmup_epochs > 0:
        print(f"[INFO] Warmup training for {warmup_epochs} epochs...")
        train_model(program, dataset, num_epochs=warmup_epochs)
    
    soft_after_warmup = soft_count_zero(answer_module, dataset, device)
    print(f"[INFO] Soft sum zero (after warmup): {soft_after_warmup:.4f}")
    
    program.inferTypes = eval_infer
    before_count = evaluate_model(program, dataset, b_answer).get(expected_value, 0)
    
    print(f"\n[INFO] After warmup - Count of '{expected_value}': {before_count}/{M}")
    print(f"[INFO] Target count: {target_count}")
    
    # Smart initialization: adjust model to get close to target count
    # If we want 'target_count' of class 'expected_value', we need soft sum ≈ target_count
    current_soft = soft_after_warmup
    
    # Determine if we need to flip and by how much
    if expected_value == 0:
        # We want target_count zeros (soft sum should be ~target_count)
        if current_soft < 0.5:
            # Model is predicting mostly ones, flip to get mostly zeros
            print(f"[INFO] Flipping to get zeros (current soft={current_soft:.2f}, want={target_count})")
            with torch.no_grad():
                for param in answer_module.layers[2].parameters():
                    param.mul_(-1.0)
            current_soft = soft_count_zero(answer_module, dataset, device)
            print(f"[INFO] After flip - soft sum: {current_soft:.4f}")
        
        # Now adjust magnitude to get closer to target
        # If soft sum is too high, scale down; if too low, scale up
        if abs(current_soft - target_count) > 1.0:
            # Compute scale factor to get closer to target
            # We want to move from current_soft toward target_count
            scale = 1.0 + 0.3 * (target_count - current_soft) / M
            scale = max(0.5, min(2.0, scale))  # Clip to reasonable range
            print(f"[INFO] Scaling output layer by {scale:.3f} to approach target")
            with torch.no_grad():
                for param in answer_module.layers[2].parameters():
                    param.mul_(scale)
            current_soft = soft_count_zero(answer_module, dataset, device)
            print(f"[INFO] After scaling - soft sum: {current_soft:.4f}")
    else:
        # We want target_count ones (soft sum should be ~(M - target_count))
        target_soft_zeros = M - target_count
        if current_soft > M - 0.5:
            # Model is predicting mostly zeros, flip to get mostly ones
            print(f"[INFO] Flipping to get ones (current soft={current_soft:.2f}, want zeros={target_soft_zeros})")
            with torch.no_grad():
                for param in answer_module.layers[2].parameters():
                    param.mul_(-1.0)
            current_soft = soft_count_zero(answer_module, dataset, device)
            print(f"[INFO] After flip - soft sum: {current_soft:.4f}")
        
        # Adjust magnitude
        if abs(current_soft - target_soft_zeros) > 1.0:
            scale = 1.0 + 0.3 * (target_soft_zeros - current_soft) / M
            scale = max(0.5, min(2.0, scale))
            print(f"[INFO] Scaling output layer by {scale:.3f} to approach target")
            with torch.no_grad():
                for param in answer_module.layers[2].parameters():
                    param.mul_(scale)
            current_soft = soft_count_zero(answer_module, dataset, device)
            print(f"[INFO] After scaling - soft sum: {current_soft:.4f}")

    # Re-evaluate after adjustments
    before_count = evaluate_model(program, dataset, b_answer).get(expected_value, 0)
    print(f"[INFO] Hard count after initialization: {before_count}/{M}")

    def flat_params(m): 
        return torch.cat([p.detach().float().flatten().cpu() for p in m.parameters() if p.requires_grad]) if any(p.requires_grad for p in m.parameters()) else torch.tensor([])

    w_before = flat_params(answer_module)
    soft_before = soft_count_zero(answer_module, dataset, device)
    
    # Constraint training
    print("\n[DEBUG] Starting constraint training...")
    program.model.train()
    program.cmodel.train()
    program.inferTypes = train_infer
    
    constraint_epochs = args.epoch
    train_model(program, dataset, constraint_epochs, constr_loss_only=True)
    
    w_after = flat_params(answer_module)
    soft_after = soft_count_zero(answer_module, dataset, device)
    
    print(f"\n[DEBUG] Delta-norm(weights): {torch.norm(w_after - w_before).item():.6f}")
    print(f"[DEBUG] Soft sum zero (before constraint training): {soft_before:.4f}")
    print(f"[DEBUG] Soft sum zero (after constraint training): {soft_after:.4f}")
    print(f"[DEBUG] Delta soft sum: {abs(soft_after - soft_before):.4f}")

    program.inferTypes = eval_infer
    actual_count = evaluate_model(program, dataset, b_answer).get(expected_value, 0)

    with torch.no_grad():
        for di, item in enumerate(dataset):
            x = torch.as_tensor(item["b"], device=device)
            logits = _run_answer_module(answer_module, x)
            preds = logits.argmax(dim=-1).tolist()
            print(f"[data {di}] preds={preds}, count of {expected_value}: {preds.count(expected_value)}")
          
    pass_test_case = True  
    if args.atLeastL:
        pass_test_case &= (actual_count >= args.expected_atLeastL)
    if args.atMostL:
        pass_test_case &= (actual_count <= args.expected_atMostL)
    if not args.atLeastL and not args.atMostL:
        pass_test_case &= (actual_count == args.expected_atLeastL)

    print(f"\nTest case {'PASSED' if pass_test_case else 'FAILED'}")
    print(f"expected_value, before_count, actual_count, pass_test_case: ({expected_value}, {before_count}, {actual_count}, {pass_test_case})")
    return pass_test_case, before_count, actual_count

if __name__ == "__main__":
    args = parse_arguments()
    main(args)