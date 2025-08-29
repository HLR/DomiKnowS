import sys
sys.path.append('../../..')
import os
#os.environ["GRB_LICENSE_FILE"] = "/Users/tanawanpremsri/Downloads/gurobi-5.lic"
from pathlib import Path
import numpy as np
from utils import create_dataset_relation
from collections import Counter
import argparse
from graph_rel import get_graph

from domiknows.graph import Graph, Concept, Relation, andL, orL
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import InferenceProgram, PrimalDualProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import ModuleSensor, FunctionalSensor, ModuleLearner
from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor
from domiknows.program.metric import MacroAverageTracker
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, FunctionalSensor, CompositionCandidateSensor
import torch
import random

def set_seed_everything(seed=380):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        if torch.cuda.device_count() > 1:
            print(f"Multiple GPUs detected: {torch.cuda.device_count()} GPUs available")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument("--constraint_2_existL", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--use_andL", action="store_true")
    args = parser.parse_args()
    
    # Get device
    device = get_device()
    
    # Print setup parameters to console
    print("\n=== Setup Parameters ===")
    print(f"Device: {device}")
    print(f"Number of samples (N): {args.N}")
    print(f"Learning rate: {args.lr}")
    print(f"Number of epochs: {args.epoch}")
    if args.constraint_2_existL:
        print(f"Using Constraint 2 ExistL")
    if args.use_andL:
        print(f"Using andL")
    if args.evaluate:
        print(f"Evaluate mode: {args.evaluate}")
    print("=====================\n")

    set_seed_everything()

    np.random.seed(0)
    # N scene, each has M objects, each object has length of K emb
    # Condition if
    N = args.N
    M = 2
    K = 8
    train, test, all_label_test = create_dataset_relation(args, N=N, M=M, K=K, read_data=True)

    dataset = test if args.evaluate else train

    count_all_labels = Counter(all_label_test)
    majority_vote = max([val for val in count_all_labels.values()])

    (graph, scene, objects, scene_contain_obj, relation, obj1, obj2,
     is_cond1, is_cond2,
     is_relation1, is_relation2) = get_graph(args)
    #
    #
    # Read Constraint Label
    scene["all_obj"] = ReaderSensor(keyword="all_obj")
    objects["obj_index"] = ReaderSensor(keyword="obj_index")
    objects["obj_emb"] = ReaderSensor(keyword="obj_emb")
    objects[scene_contain_obj] = EdgeSensor(objects["obj_index"], scene["all_obj"],
                                            relation=scene_contain_obj,
                                            forward=lambda b, _: torch.ones(len(b)).unsqueeze(-1).to(device))

    class Regular2LayerMultiGPU(torch.nn.Module):
        def __init__(self, size, device):
            super().__init__()
            self.size = size
            self.device = device
            self.layer = torch.nn.Sequential(
                torch.nn.Linear(self.size, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 2)
            ).to(device)
            self.softmax = torch.nn.Softmax(dim=1)
            
            # Enable multi-GPU if available
            if torch.cuda.device_count() > 1:
                self.layer = torch.nn.DataParallel(self.layer)

        def forward(self, p):
            p = p.to(self.device)
            output = self.layer(p)
            return self.softmax(output)

    objects[is_cond1] = ModuleLearner("obj_emb", module=Regular2LayerMultiGPU(size=K, device=device))
    objects[is_cond2] = ModuleLearner("obj_emb", module=Regular2LayerMultiGPU(size=K, device=device))

    # Relation Layer
    class RelationLayersMultiGPU(torch.nn.Module):
        def __init__(self, size, device):
            super().__init__()
            self.size = size
            self.device = device
            self.layer = torch.nn.Sequential(
                torch.nn.Linear(self.size * 2, 512),
                torch.nn.Sigmoid(),
                torch.nn.Linear(512, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 2)
            ).to(device)
            self.softmax = torch.nn.Softmax(dim=1)
            
            # Enable multi-GPU if available
            if torch.cuda.device_count() > 1:
                self.layer = torch.nn.DataParallel(self.layer)

        def forward(self, p):
            emb = p.to(self.device)
            N, K = emb.shape

            left = emb.unsqueeze(1).expand(N, N, K)
            right = emb.unsqueeze(0).expand(N, N, K)

            mask = ~torch.eye(N, dtype=torch.bool, device=self.device)
            left = left[mask]
            right = right[mask]

            pairs = torch.cat((left, right), dim=-1)
            output = self.layer(pairs)
            return self.softmax(output)
    
    def filter_relation(_, arg1, arg2):
        return arg1.getAttribute("obj_index") != arg2.getAttribute("obj_index")

    relation[obj1.reversed, obj2.reversed] = CompositionCandidateSensor(
        objects['obj_index'],
        relations=(obj1.reversed, obj2.reversed),
        forward=filter_relation)

    relation[is_relation1] = ModuleLearner(objects["obj_emb"], module=RelationLayersMultiGPU(size=K, device=device))
    relation[is_relation2] = ModuleLearner(objects["obj_emb"], module=RelationLayersMultiGPU(size=K, device=device))

    # Move dataset tensors to device
    for i in range(len(dataset)):
        dataset[i]["logic_label"] = torch.LongTensor([bool(dataset[i]['condition_label'][0])]).to(device)
        # Move other tensors in dataset to device as needed
        for key, value in dataset[i].items():
            if isinstance(value, torch.Tensor):
                dataset[i][key] = value.to(device)

    dataset = graph.compile_logic(dataset, logic_keyword='logic_str', logic_label_keyword='logic_label')
    program = InferenceProgram(graph, SolverModel,
                               poi=[scene, objects, is_cond1, is_cond2, relation, is_relation1, is_relation2, graph.constraint],
                               tnorm="G")

    # Move program to device if possible
    if hasattr(program, 'to'):
        program = program.to(device)

    program.train(dataset, Optim=torch.optim.Adam, train_epoch_num=args.epoch, c_lr=args.lr, c_warmup_iters=-1,
                  batch_size=1, print_loss=False)
    acc_train_after = program.evaluate_condition(dataset)

    results_files = open(f"results_N_{args.N}.text", "a")

    from datetime import datetime

    # Function for dual printing
    def dual_print(message):
        print(message)  # Print to console
        print(message, file=results_files)  # Print to file

    # Get current date and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    dual_print(f"=== Run at: {current_time} ===")
    dual_print(f"Device: {device}")
    dual_print(f"N = {args.N}\nLearning Rate = {args.lr}\nNum Epoch = {args.epoch}")
    dual_print(f"Logic Used: {'ExistL' if args.constraint_2_existL else 'AndL' if args.use_andL else 'None'}")
    dual_print(f"Acc on training set after training: {acc_train_after}")
    dual_print(f"Acc Majority Vote: {majority_vote * 100 / len(test):.2f}")
    dual_print("#" * 50)

    results_files.close()

    out_dir = Path(__file__).resolve().parent / "models"  
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"1_existL_diverse_relation_{args.N}_lr_{args.lr}.pth"
    program.save(out_path)