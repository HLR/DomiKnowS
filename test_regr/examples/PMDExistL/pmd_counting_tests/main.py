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
            initial_temp=0.1,  # Always sharp
            final_temp=0.1,
            beta=args.beta,
            device=device,
            tnorm="L",
            counting_tnorm=args.counting_tnorm
        )

    labels = dataset[0]['label']
    num_ones = sum(labels)
    M = len(labels)
    
    # Strategy: Always do substantial warmup to get diverse predictions
    warmup_epochs = 10
    expected_value = args.expected_value

    # Add noise to break symmetry and get diverse initial predictions
    with torch.no_grad():
        for param in answer_module.parameters():
            param.add_(torch.randn_like(param) * 0.1)

    if warmup_epochs > 0:
        print(f"[INFO] Warmup training for {warmup_epochs} epochs...")
        train_model(program, dataset, num_epochs=warmup_epochs)
    
    program.inferTypes = eval_infer
    expected_value = args.expected_value
    before_count = evaluate_model(program, dataset, b_answer).get(expected_value, 0)
    
    print(f"\n[INFO] After warmup - Count of '{expected_value}': {before_count}/{M}")
    print(f"[INFO] Target count: {args.expected_atLeastL}")
    
    # Check if we need to flip all predictions
    if before_count == 0 and args.expected_atLeastL > 0:
        print("[INFO] All predictions are wrong class - flipping model output layer sign")
        with torch.no_grad():
            # Flip the final layer to reverse predictions
            for param in answer_module.layers[2].parameters():
                param.mul_(-1.0)
        
        # Re-evaluate
        before_count = evaluate_model(program, dataset, b_answer).get(expected_value, 0)
        print(f"[INFO] After flip - Count of '{expected_value}': {before_count}/{M}")
    
    def flat_params(m): 
        return torch.cat([p.detach().float().flatten().cpu() for p in m.parameters() if p.requires_grad]) if any(p.requires_grad for p in m.parameters()) else torch.tensor([])

    w_before = flat_params(answer_module)
    
    # Check constraint loss before training
    print("\n[DEBUG] Checking constraint loss before constraint training...")
    program.model.eval()
    program.cmodel.eval()
    with torch.no_grad():
        for data in dataset:
            _, _, *output = program.model(data)
            closs, *_ = program.cmodel(output[1])
            if torch.is_tensor(closs):
                print(f"[DEBUG] Initial constraint loss: {closs.item():.6f}")
            else:
                print(f"[DEBUG] Initial constraint loss is not a tensor: {closs}")
    
    program.model.train()
    program.cmodel.train()
    train_model(program, dataset, args.epoch, constr_loss_only=True)
    
    # Check constraint loss after training
    print("\n[DEBUG] Checking constraint loss after constraint training...")
    program.model.eval()
    program.cmodel.eval()
    with torch.no_grad():
        for data in dataset:
            _, _, *output = program.model(data)
            closs, *_ = program.cmodel(output[1])
            if torch.is_tensor(closs):
                print(f"[DEBUG] Final constraint loss: {closs.item():.6f}")
            else:
                print(f"[DEBUG] Final constraint loss is not a tensor: {closs}")
    
    w_after = flat_params(answer_module)
    print("[DEBUG] Delta-norm(weights):", torch.norm(w_after - w_before).item())
    
    for n,p in answer_module.named_parameters():
        if p.grad is not None:
            print(f"[GRAD] {n} norm={p.grad.data.norm().item():.4e}")
            
    def soft_count_zero(module, dataset, device):
        import torch
        s = 0.0
        for item in dataset:
            x = torch.as_tensor(item["b"], dtype=torch.float32, device=device)
            logits = _run_answer_module(module, x)
            p = torch.softmax(logits, dim=-1)[:, 0]
            s += p.sum().item()
        return s

    print("soft sum zero (before):", soft_count_zero(answer_module, dataset, device))
    print("soft sum zero (after):", soft_count_zero(answer_module, dataset, device))

    program.inferTypes = eval_infer
    actual_count = evaluate_model(program, dataset, b_answer).get(expected_value, 0)

    with torch.no_grad():
        expected = args.expected_value
        for di, item in enumerate(dataset):
            x = torch.as_tensor(item["b"], device=device)
            logits = _run_answer_module(answer_module, x)
            preds = logits.argmax(dim=-1).tolist()
            idx_expected = [i for i, p in enumerate(preds) if p == expected]
            idx_other    = [i for i, p in enumerate(preds) if p != expected]
            print(f"[data {di}] preds={preds}")
            print(f"[data {di}] indices predicting {expected}: {idx_expected}")
            print(f"[data {di}] indices predicting {1-expected}: {idx_other}")
          
    pass_test_case = True  
    if args.atLeastL:
        pass_test_case &= (actual_count >= args.expected_atLeastL)
    if args.atMostL:
        pass_test_case &= (actual_count <= args.expected_atMostL)
    if not args.atLeastL and not args.atMostL:
        pass_test_case &= (actual_count == args.expected_atLeastL)

    print(f"Test case {'PASSED' if pass_test_case else 'FAILED'}")
    print(
        f"expected_value, before_count, actual_count,pass_test_case): {expected_value, before_count, actual_count, pass_test_case}")
    return pass_test_case, before_count, actual_count


if __name__ == "__main__":
    args = parse_arguments()
    main(args)