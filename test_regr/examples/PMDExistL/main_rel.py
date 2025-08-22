from click import Path
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
import numpy


def set_seed_everything(seed=380):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument("--constraint_2_existL", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    set_seed_everything()

    np.random.seed(0)
    # N scene, each has M objects, each object has length of K emb
    # Condition if
    N = args.N
    M = 6
    K = 6
    train, test, all_label_test = create_dataset_relation(args, N=N, M=M, K=K)

    dataset = test if args.evaluate else train

    count_all_labels = Counter(all_label_test)
    majority_vote = max([val for val in count_all_labels.values()])

    (graph, scene, objects, scene_contain_obj, relation, obj1, obj2,
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

    class Regular2Layer(torch.nn.Module):
        def __init__(self, size):
            super().__init__()
            self.size = size
            self.layer = torch.nn.Sequential(torch.nn.Linear(self.size, 256),
                                             torch.nn.Sigmoid(),
                                             torch.nn.Linear(256, 2))

        def forward(self, p):
            # print(self.layer.weight)
            output = self.layer(p)
            return output


    objects[is_cond1] = ModuleLearner("obj_emb", module=Regular2Layer(size=K))
    objects[is_cond2] = ModuleLearner("obj_emb", module=Regular2Layer(size=K))
    objects[is_cond3] = ModuleLearner("obj_emb", module=Regular2Layer(size=K))
    objects[is_cond4] = ModuleLearner("obj_emb", module=Regular2Layer(size=K))


    # Relation Layer
    class RelationLayers(torch.nn.Module):
        def __init__(self, size):
            super().__init__()
            self.size = size
            self.layer = torch.nn.Sequential(torch.nn.Linear(self.size, 256),
                                             torch.nn.Sigmoid(),
                                             torch.nn.Linear(256, 2))

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
            return output


    def filter_relation(_, arg1, arg2):
        return arg1.getAttribute("obj_index") != arg2.getAttribute("obj_index")


    relation[obj1.reversed, obj2.reversed] = CompositionCandidateSensor(
        objects['obj_index'],
        relations=(obj1.reversed, obj2.reversed),
        forward=filter_relation)
    #

    relation[is_relation1] = ModuleLearner(objects["obj_emb"], module=RelationLayers(size=2 * K))
    relation[is_relation2] = ModuleLearner(objects["obj_emb"], module=RelationLayers(size=2 * K))
    relation[is_relation3] = ModuleLearner(objects["obj_emb"], module=RelationLayers(size=2 * K))
    relation[is_relation4] = ModuleLearner(objects["obj_emb"], module=RelationLayers(size=2 * K))

    # TODO: Adding logic_str and logic_label

    # TODO: Fix this Following this
    for i in range(len(dataset)):
        dataset[i]["logic_label"] = torch.LongTensor([bool(dataset[i]['condition_label'][0])])

    dataset = graph.compile_logic(dataset, logic_keyword='logic_str', logic_label_keyword='logic_label')
    program = InferenceProgram(graph, SolverModel,
                               poi=[scene, objects, is_cond1, is_cond2, is_cond3, is_cond4, relation, is_relation1, is_relation2, is_relation3, is_relation4, graph.constraint],
                               tnorm="G")

    acc_train_before = program.evaluate_condition(dataset)
    program.train(dataset, Optim=torch.optim.Adam, train_epoch_num=args.epoch, c_lr=args.lr, c_warmup_iters=-1,
                  batch_size=1, print_loss=False)
    acc_train_after = program.evaluate_condition(dataset)

    results_files = open(f"results_N_{args.N}.text", "a")

    print(f"N = {args.N}\nLearning Rate = {args.lr}\nNum Epoch = {args.epoch}", file=results_files)
    print("Constraint Two ExistL:", args.constraint_2_existL, file=results_files)
    print("Acc on training set before training: ", acc_train_before, file=results_files)
    print("Acc on training set after training: ", acc_train_after, file=results_files)
    # print("Acc on testing set after training: ", acc_test_after, file=results_files)
    print("Acc Majority Vote: {:.2f}".format(majority_vote * 100 / len(test)), file=results_files)
    print("#" * 50, file=results_files)

    out_dir = Path(__file__).resolve().parent / "models"  
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"1_existL_diverse_relation_{args.N}_lr_{args.lr}.pth"
    program.save(out_path)
