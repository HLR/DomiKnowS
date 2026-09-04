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
from domiknows.program.lossprogram import GumbelPrimalDualProgram
from domiknows.program.lossprogram import GumbelSampleLossProgram

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
    parser.add_argument("--model", default="sampling", type=str, help="Model Types [Sampling/PMD/gumbel_pmd/gumbel_sampling]")
    parser.add_argument("--sample_size", default=-1, type=int, help="Sample size for sampling program")
    parser.add_argument("--use_gumbel", default=True, type=bool, help="Enable Gumbel-Softmax")
    parser.add_argument("--initial_temp", default=2.0, type=float, help="Initial temperature for Gumbel-Softmax")
    parser.add_argument("--final_temp", default=0.1, type=float, help="Final temperature for Gumbel-Softmax")
    parser.add_argument("--hard_gumbel", default=False, type=bool, help="Use hard Gumbel-Softmax")
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

    print(f"Dataset created with M={args.M} samples")

    answer_module = setup_graph(args, a, b, a_contain_b, b_answer, device=device)
    
    train_infer = ['local/softmax']
    eval_infer  = ['local/argmax']

    if args.model in ["sampling", "gumbel_sampling"]:
        program = GumbelSampleLossProgram(
            graph, SolverModel, 
            poi=[a, b, b_answer],
            inferTypes=train_infer,
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            sample=True, 
            sampleSize=args.sample_size,
            sampleGlobalLoss=True,
            use_gumbel=args.use_gumbel,
            initial_temp=args.initial_temp,
            final_temp=args.final_temp,
            hard_gumbel=args.hard_gumbel,
            anneal_start_epoch=20,
            beta=args.beta, 
            device=device, 
            tnorm="L", 
            counting_tnorm=args.counting_tnorm
        )
    else:  # PMD or gumbel_pmd
        program = GumbelPrimalDualProgram(
            graph, SolverModel,
            poi=[a, b, b_answer],
            inferTypes=train_infer,
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            use_gumbel=args.use_gumbel,
            initial_temp=args.initial_temp,
            final_temp=args.final_temp,
            beta=args.beta,
            device=device,
            tnorm="L",
            counting_tnorm=args.counting_tnorm
        )

    # Add noise to break symmetry
    with torch.no_grad():
        for param in answer_module.parameters():
            param.add_(torch.randn_like(param) * 0.2)

    # Setup optimizer for model training
    optimizer = torch.optim.Adam(answer_module.parameters(), lr=5e-3)
    program.opt = optimizer
    
    # Use NEW generalized training with phases
    warmup_epochs = 20
    constraint_epochs = args.epoch
    
    print(f"[INFO] Starting phased training: warmup={warmup_epochs}, constraint={constraint_epochs}")
    
    program.train(
        training_set=dataset,
        valid_set=None,
        test_set=None,
        warmup_epochs=warmup_epochs,
        constraint_epochs=constraint_epochs,
        constraint_only=True,  # Use constraint-only mode for phase 2
        constraint_loss_scale=200.0,  # Strong scaling for constraint loss
        c_lr=5e-3,
        batch_size=1,
        dataset_size=len(dataset)
    )
    
    # Final evaluation
    program.inferTypes = eval_infer
    actual_count = evaluate_model(program, dataset, b_answer).get(args.expected_value, 0)

    # Debug output: show predictions
    with torch.no_grad():
        for di, item in enumerate(dataset):
            x = torch.as_tensor(item["b"], device=device)
            logits = _run_answer_module(answer_module, x)
            preds = logits.argmax(dim=-1).tolist()
            print(f"[data {di}] preds={preds}, count of {args.expected_value}: {preds.count(args.expected_value)}")
    
    # Note: before_count not available with program.train() approach
    before_count = 0
          
    # Check if constraints are satisfied
    pass_test_case = True  
    if args.atLeastL and args.atMostL:
        pass_test_case = (actual_count >= args.expected_atLeastL) and (actual_count <= args.expected_atMostL)
    elif args.atLeastL:
        pass_test_case = (actual_count >= args.expected_atLeastL)
    elif args.atMostL:
        pass_test_case = (actual_count <= args.expected_atMostL)
    else:
        pass_test_case = (actual_count == args.expected_atLeastL)

    print(f"\nTest case {'PASSED' if pass_test_case else 'FAILED'}")
    print(f"expected_value, before_count, actual_count, pass_test_case: ({args.expected_value}, {before_count}, {actual_count}, {pass_test_case})")
    return pass_test_case, before_count, actual_count