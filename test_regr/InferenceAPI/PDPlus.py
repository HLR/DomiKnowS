import json
import logging, torch

import numpy as np

from domiknows.graph import Graph, Concept, Relation, andL, orL
from NER_utils import generate_dataset, reader_format
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import InferenceProgram, PrimalDualProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import ModuleSensor, FunctionalSensor, ModuleLearner
from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor
from domiknows.program.metric import MacroAverageTracker

logging.basicConfig(level=logging.INFO)
Graph.clear()
Concept.clear()
Relation.clear()

with Graph(name='global') as graph:
    person = Concept(name='person')
    # Input 1
    p1 = person(name='p1')  # [a, b] -> Learnable
    p2 = person(name='p2')
    p3 = person(name='p3')

    location = Concept(name='location')

    l1 = location(name='l1')  # [a, b]
    l2 = location(name='l2')
    l3 = location(name='l3')

    pair = Concept(name="pair")
    people_arg, location_arg = pair.has_a(people_arg=person, location_arg=location)

    work_in1 = pair(name="workin1")  # Relation of p1 and l1 -> Learnable
    work_in2 = pair(name="workin2")
    work_in3 = pair(name="workin3")

    # work_in = learnable (people1, location1)
    # cond1_check = (work1 and people1) and (work2 and people2)
    # Supervised from these two conditions
    constarint1 = andL(andL(p1("x"), work_in1("z", path=('x', people_arg.reversed))),
                       andL(p2("y"), work_in2("t", path=('y', people_arg.reversed))))

    # cond2_check = (work2 and people2) or (work3 and people3)
    constarint2 = orL(andL(p2("x"), work_in2("z", path=('x', people_arg.reversed))),
                      andL(p3("y"), work_in3("t", path=('y', people_arg.reversed))))

# torch.manual_seed(4)
N = 1000
data_list = generate_dataset(sample_num=N)
data_list = reader_format(data_list)
# json.dump(data_list, open("dataset_1000.json", "w"), indent=3)
# N = 1000
# training_file = "dataset_10.json" if N==10 else "dataset_1000.json"
# with open(training_file, "r") as jsonf:
#     data_list = json.load(jsonf)
#     for data in data_list:
#         for k, v in data.items():
#             data[k] = [torch.Tensor(vi) if not isinstance(vi, bool) else vi for vi in v]

from sklearn.model_selection import train_test_split
train, test = train_test_split(data_list, test_size=0.2, random_state=0)

person["embedding1"] = ReaderSensor(keyword="person1")
person["embedding2"] = ReaderSensor(keyword="person2")
person["embedding3"] = ReaderSensor(keyword="person3")

graph.constraint[constarint1] = ReaderSensor(keyword="condition_1", label=True)
graph.constraint[constarint2] = ReaderSensor(keyword="condition_2", label=True)


class PeopleModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.layer = torch.nn.Sequential(torch.nn.Linear(self.size, 256),
                                         torch.nn.Sigmoid(),
                                         torch.nn.Linear(256, 2))

    def forward(self, p):
        # print(self.layer.weight)
        return self.layer(p[0]).unsqueeze(0)


is_real_person = PeopleModel(2).to("cpu")
person[p1] = ModuleLearner(person["embedding1"], module=is_real_person, device="cpu")
person[p2] = ModuleLearner(person["embedding2"], module=is_real_person, device="cpu")
person[p3] = ModuleLearner(person["embedding3"], module=is_real_person, device="cpu")

location["embedding1"] = ReaderSensor(keyword="location1")
location["embedding2"] = ReaderSensor(keyword="location2")
location["embedding3"] = ReaderSensor(keyword="location3")


class WorkModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.layer = torch.nn.Sequential(torch.nn.Linear(self.size, 256),
                                         torch.nn.Sigmoid(),
                                         torch.nn.Linear(256, 2))

    def forward(self, p, l):
        # print(self.layer.weight)
        return self.layer(torch.cat([p[0], l[0]])).unsqueeze(0)

work_in_model = WorkModel(4).to("cpu")
pair[people_arg.reversed, location_arg.reversed] = JointSensor(forward=lambda *_: (torch.ones(1, 1), torch.ones(1, 1)))
pair[work_in1] = ModuleLearner(person["embedding1"], location["embedding1"], module=work_in_model, device="cpu")
pair[work_in2] = ModuleLearner(person["embedding2"], location["embedding2"], module=work_in_model, device="cpu")
pair[work_in3] = ModuleLearner(person["embedding3"], location["embedding3"], module=work_in_model, device="cpu")
program = InferenceProgram(graph, SolverModel, poi=[person, p1, p2, p3, location, pair, graph.constraint], device="cpu", tnorm='G')

# print(list(program.model.parameters()))

from tqdm import tqdm


def check_constraints_acc(cur_program, cur_data):
    acc = [0, 0]
    total = 0
    for datanode in tqdm(cur_program.populate(cur_data, device="cpu"), "Manually Testing"):
        verify_constrains = datanode.verifyResultsLC()

        condition_list = []
        if not verify_constrains:
            continue

        for lc in verify_constrains:
            condition_list.append(1 if verify_constrains[lc]["satisfied"] == 100.0 else 0)

        # This might be a better way for this to finding the label, but this work
        constraints_keys = list(verify_constrains.keys())
        find_constraints_label = datanode.myBuilder.findDataNodesInBuilder(select=datanode.graph.constraint)[0]
        constraint_labels_dict = find_constraints_label.getAttributes()
        # Getting label of constraints and convert to 0-1
        constraint_labels = [int(constraint_labels_dict[lc + "/label"].item()) for lc in constraints_keys]


        # This is checking the constraint manually, should match the verify constraints
        work1 = datanode.getAttribute(work_in1, 'local/softmax').argmax().item()
        work2 = datanode.getAttribute(work_in2, 'local/softmax').argmax().item()
        work3 = datanode.getAttribute(work_in3, 'local/softmax').argmax().item()

        datanode_person = datanode.getRelationLinks(people_arg)[0]
        people1 = datanode_person.getAttribute(p1, 'local/softmax').argmax().item()
        # print(datanode_person.getAttribute(p1, 'local/softmax'))
        people2 = datanode_person.getAttribute(p2, 'local/softmax').argmax().item()
        people3 = datanode_person.getAttribute(p3, 'local/softmax').argmax().item()

        # Condition used in ER examples
        cond1_check = (work1 and people1) and (work2 and people2)
        cond2_check = (work2 and people2) or (work3 and people3)
        if condition_list[0] != cond1_check or condition_list[1] != cond2_check:
            raise Exception("verify_constrains cannot be used")

        # print(constraint_labels, condition_list)
        # == check matching the label
        for i in range(2):
            acc[i] += int(constraint_labels[i] == condition_list[i])
        total += 1

    return np.array(acc) / total


output_file = open("results.txt", "a")
# print(list(program.model.parameters()))
print("Training Size :", N*4, file=output_file)

print("Acc train before training:", program.evaluate_condition(train), file=output_file)
print("Acc test before training:", program.evaluate_condition(test), file=output_file)
# print(list(program.model.parameters()))
program.train(train, Optim=lambda param: torch.optim.AdamW(param, 1e-3), train_epoch_num=30, c_lr=1e-4, c_warmup_iters=-1, device="cpu")
print("Acc train after training:", program.evaluate_condition(train), file=output_file)
print("Acc test after training:", program.evaluate_condition(test), file=output_file)
# print(list(program.model.parameters()))
output_file.close()