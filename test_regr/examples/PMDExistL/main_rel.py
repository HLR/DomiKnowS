import sys

sys.path.append('../../..')
import os
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

from domiknows import setProductionLogMode

setProductionLogMode(True)


def set_seed_everything(seed=380):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=4)
    parser.add_argument("--constraint_2_existL", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--use_andL", action="store_true")
    parser.add_argument("--max_property", type=int, default=3)
    parser.add_argument("--max_relation", type=int, default=3)
    parser.add_argument("--load_save", type=str, default="")
    parser.add_argument("--save_file", type=str, default="model.pth")
    args = parser.parse_args()

    # Print setup parameters to console
    print("\n=== Setup Parameters ===")
    print(f"Number of samples (N): {args.N}")
    print(f"Learning rate: {args.lr}")
    print(f"Number of epochs: {args.epoch}")
    print("Number of relations: ", args.max_relation)
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
    train, test, all_label_test = create_dataset_relation(args, N=N, M=M, K=K, read_data=False)

    dataset = test if args.evaluate else train

    count_all_labels = Counter(all_label_test)
    majority_vote = max([val for val in count_all_labels.values()])
    print(f"{count_all_labels}")

    (graph, scene, objects, scene_contain_obj, relation_obj1_obj2, obj1, obj2,
     is_cond1, is_cond2, is_cond3, is_cond4,
     is_relation1, is_relation2, is_relation3, is_relation4) = get_graph(args)
    #
    #
    # Read Constraint Label
    scene["all_obj"] = ReaderSensor(keyword="all_obj")
    objects["obj_index"] = ReaderSensor(keyword="obj_index")
    objects["obj_emb"] = ReaderSensor(keyword="obj_emb")
    objects[scene_contain_obj] = EdgeSensor(objects["obj_index"], scene["all_obj"],
                                            relation=scene_contain_obj,
                                            forward=lambda b, _: torch.ones(len(b)).unsqueeze(-1))


    # Set label of inference condition
    # graph.constraint[learning_condition] = ReaderSensor(keyword="condition_label", label=True)

    class ProbabilityExtractor(torch.nn.Module):
        def __init__(self, module_learner):
            super().__init__()
            self.module_learner = module_learner

        def forward(self, *args, **kwargs):
            output = self.module_learner(*args, **kwargs)  # Shape: (batch_size, 2)
            return output[:, 1]  # Return only index 1 probabilities


    class Regular2Layer(torch.nn.Module):
        def __init__(self, size):
            super().__init__()
            self.size = size
            self.layer = torch.nn.Sequential(torch.nn.Linear(self.size, 256),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(256, 2))
            self.softmax = torch.nn.Softmax(dim=1)

        def forward(self, p):
            # print(self.layer.weight)
            output = self.layer(p)
            return self.softmax(output)


    objects[is_cond1] = ModuleLearner("obj_emb", module=Regular2Layer(size=K))
    objects[is_cond2] = ModuleLearner("obj_emb", module=Regular2Layer(size=K))
    objects[is_cond3] = ModuleLearner("obj_emb", module=Regular2Layer(size=K))
    objects[is_cond4] = ModuleLearner("obj_emb", module=Regular2Layer(size=K))


    # Relation Layer
    class RelationLayers(torch.nn.Module):
        def __init__(self, size):
            super().__init__()
            self.size = size
            self.layer = torch.nn.Sequential(
                torch.nn.Linear(self.size * 2, 512),
                torch.nn.Sigmoid(),
                torch.nn.Linear(512, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 2))
            self.softmax = torch.nn.Softmax(dim=1)

        def forward(self, p):
            # print(self.layer.weight)
            # Prepare NxN matrix from object emb
            emb = p
            N, K = emb.shape

            # 1. Broadcast each row against every other row
            left = emb.unsqueeze(1).expand(N, N, K)
            right = emb.unsqueeze(0).expand(N, N, K)

            mask = ~torch.eye(N, dtype=torch.bool)  # False on the diagonal
            left = left[mask]  # shape (N*(N-1), K)
            right = right[mask]  # shape (N*(N-1), K)

            pairs = torch.cat((left, right), dim=-1)
            output = self.layer(pairs)
            return self.softmax(output)


    def filter_relation(_, arg1, arg2):
        # print(f"filter_relation called with {arg1.getAttribute('obj_index')} and {arg2.getAttribute('obj_index')}")
        result = arg1.getAttribute("obj_index") != arg2.getAttribute("obj_index")
        return result


    relation_obj1_obj2[obj1.reversed, obj2.reversed] = CompositionCandidateSensor(
        objects['obj_index'],
        relations=(obj1.reversed, obj2.reversed),
        forward=filter_relation)

    relation_obj1_obj2[is_relation1] = ModuleLearner(objects["obj_emb"], module=RelationLayers(size=K))
    relation_obj1_obj2[is_relation2] = ModuleLearner(objects["obj_emb"], module=RelationLayers(size=K))
    relation_obj1_obj2[is_relation3] = ModuleLearner(objects["obj_emb"], module=RelationLayers(size=K))
    relation_obj1_obj2[is_relation4] = ModuleLearner(objects["obj_emb"], module=RelationLayers(size=K))

    for i in range(len(dataset)):
        dataset[i]["logic_label"] = torch.LongTensor([bool(dataset[i]['condition_label'][0])])

    dataset = graph.compile_logic(dataset, logic_keyword='logic_str', logic_label_keyword='logic_label')

    program = InferenceProgram(graph, SolverModel,
                               poi=[scene, objects, is_cond1, is_cond2, relation_obj1_obj2, is_relation1, is_relation2,
                                    graph.constraint],
                               tnorm="G", inferTypes=['local/argmax'])

    # acc_train_before = program.evaluate_condition(dataset)
    # print(55)
    if args.load_save:
        program.load(args.load_save)

    if not args.evaluate:
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
    dual_print(f"Previous Save {args.load_save}")
    dual_print(f"N = {args.N}\nLearning Rate = {args.lr}\nNum Epoch = {args.epoch}")
    dual_print(f"Number of Relation: {args.max_relation}")
    dual_print(f"Logic Used: {'ExistL' if args.constraint_2_existL else 'AndL' if args.use_andL else 'None'}")
    dual_print(f"Acc on training set after training: {acc_train_after}")
    dual_print(f"Acc Majority Vote: {majority_vote * 100 / len(test):.2f}")
    dual_print("#" * 50)

    results_files.close()

    out_dir = Path(__file__).resolve().parent / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.evaluate:
        program.save(args.save_file)
