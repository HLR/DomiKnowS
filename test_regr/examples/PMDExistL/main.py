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
from domiknows.program.lossprogram import PrimalDualProgram, SampleLossProgram
from domiknows.program.model.pytorch import SolverModel

from utils import TestTrainLearner, return_contain, create_dataset, evaluate_model, train_model

import traceback
torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

def excepthook(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    traceback.print_exception(exc_type, exc_value, exc_traceback)

sys.excepthook = excepthook

sys.path.append('../../../../domiknows/')
from graph import get_graph

Sensor.clear()

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Machine Learning Experiment")
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
    a["index"] = ReaderSensor(keyword="a")  # if these accept device=..., pass device=device
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
    # x: [M, N] float tensor on the right device
    with torch.no_grad():
        try:
            out = mod(x)                 # works if forward(self, x)
        except TypeError:
            out = mod(None, x)           # works if forward(self, _, x)
    if isinstance(out, tuple):            # some modules return (logits, extra)
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
    
    train_infer = ['local/softmax']   # differentiable
    eval_infer  = ['local/argmax']    # discrete for evaluation
    beta = 20

    if args.model == "sampling":
        program = SampleLossProgram(
            graph, SolverModel, poi=[a, b, b_answer],
            inferTypes=train_infer,
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            sample=True, sampleSize=args.sample_size, sampleGlobalLoss=False,
            beta=beta, device=device, tnorm="L", counting_tnorm=args.counting_tnorm
        )
    else:
        program = PrimalDualProgram(
            graph, SolverModel, poi=[a, b, b_answer],
            inferTypes=train_infer,
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            beta=beta, device=device, tnorm="L", counting_tnorm=args.counting_tnorm
        )

    expected_value = args.expected_value

    # ---------------- Warmup train (soft) ----------------
    train_model(program, dataset, num_epochs=2)

    # ---- Switch to eval for baseline count ----
    _prev = program.inferTypes
    program.inferTypes = eval_infer
    before_count = evaluate_model(program, dataset, b_answer).get(expected_value, 0)

    # ---- Restore training infer for constraint phase ----
    program.inferTypes = _prev
    train_model(program, dataset, args.epoch, constr_loss_only=True)

    # ---- Final eval (discrete) ----
    program.inferTypes = eval_infer
    actual_count = evaluate_model(program, dataset, b_answer).get(expected_value, 0)

    with torch.no_grad():
        expected = args.expected_value
        for di, item in enumerate(dataset):
            x = torch.as_tensor(item["b"], device=device)           # shape [M, N]
            logits = _run_answer_module(answer_module, x)           # shape [M, C]
            preds = logits.argmax(dim=-1).tolist()                  # [M]
            idx_expected = [i for i, p in enumerate(preds) if p == expected]
            idx_other    = [i for i, p in enumerate(preds) if p != expected]
            print(f"[data {di}] preds={preds}")
            print(f"[data {di}] indices predicting {expected}: {idx_expected}")
            print(f"[data {di}] indices predicting {1-expected}: {idx_other}")
            
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