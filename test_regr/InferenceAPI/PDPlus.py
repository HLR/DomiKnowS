import logging, torch
from domiknows.graph import Graph, Concept, Relation, andL, orL
from NER_utils import generate_dataset, reader_format
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import InferenceProgram, PrimalDualProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import ModuleSensor, FunctionalSensor
from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor
from domiknows.program.metric import MacroAverageTracker

logging.basicConfig(level=logging.INFO)
Graph.clear()
Concept.clear()
Relation.clear()

with Graph(name='global') as graph:

    person = Concept(name='person')

    p1 = person(name='p1')
    p2 = person(name='p2')
    p3 = person(name='p3')

    location = Concept(name='location')

    l1 = location(name='l1')
    l2 = location(name='l2')
    l3 = location(name='l3')

    pair = Concept(name="pair")
    people_arg,location_arg = pair.has_a(people_arg=person, location_arg=location)

    work_in1 = pair(name="workin1")
    work_in2 = pair(name="workin2")
    work_in3 = pair(name="workin3")

    constarint1 = andL(andL(p1("x"),work_in1("z",path=('x', people_arg.reversed))),andL(p2("y"),work_in2("t",path=('y', people_arg.reversed))))
    # constarint1 = andL(p1_is_real("x"), work_in1("z", path=('x', people_arg1.reversed)))
    constarint2 = orL(andL(p2("x"),work_in2("z",path=('x', people_arg.reversed))),andL(p3("y"),work_in3("t",path=('y', people_arg.reversed))))
    # constarint2 = andL(p2_is_real("x"), work_in2("z", path=('x', people_arg2.reversed)))

data_list = generate_dataset(sample_num=1000)
data_list = reader_format(data_list)

person["embedding1"] = ReaderSensor(keyword="person1")
person["embedding2"] = ReaderSensor(keyword="person2")
person["embedding3"] = ReaderSensor(keyword="person3")

graph.constraint[constarint1] = ReaderSensor(keyword="condition_1",label=True)
graph.constraint[constarint2] = ReaderSensor(keyword="condition_2",label=True)

class PeopleModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.layer = torch.nn.Linear(self.size, 2)

    def forward(self, p):
        return self.layer(p[0]).unsqueeze(0)

is_real_person=PeopleModel(2).to("cpu")
person[p1] = ModuleSensor(person["embedding1"],module=is_real_person,device="cpu")
person[p2] = ModuleSensor(person["embedding2"],module=is_real_person,device="cpu")
person[p3] = ModuleSensor(person["embedding3"],module=is_real_person,device="cpu")

location["embedding1"] = ReaderSensor(keyword="location1")
location["embedding2"] = ReaderSensor(keyword="location2")
location["embedding3"] = ReaderSensor(keyword="location3")

class WorkModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.layer = torch.nn.Linear(self.size, 2)

    def forward(self, p, l):
        return self.layer(torch.cat([p[0], l[0]])).unsqueeze(0)

work_in_model=WorkModel(4).to("cpu")
pair[people_arg.reversed,location_arg.reversed] = JointSensor(forward=lambda *_ : (torch.ones(1,1),torch.ones(1,1)))
pair[work_in1] = ModuleSensor(person["embedding1"],location["embedding1"],module=work_in_model,device="cpu")
pair[work_in2] = ModuleSensor(person["embedding2"],location["embedding2"],module=work_in_model,device="cpu")
pair[work_in3] = ModuleSensor(person["embedding3"],location["embedding3"],module=work_in_model,device="cpu")

# program = PrimalDualProgram(graph,SolverModel,poi=[p1,p2,p3,l1,l2,l3,pair1,pair2,pair3,graph.constraint],loss=MacroAverageTracker(NBCrossEntropyLoss()),device="cpu")
program = InferenceProgram(graph,SolverModel,poi=[person,p1,p2,p3,location,pair,graph.constraint],device="cpu",tnorm='G')
program.train(data_list, epochs=1, lr=1e-4, c_warmup_iters=0,device="cpu")



program = InferenceProgram(graph, SolverModel, poi=[person, p1, p2, p3, location, pair, graph.constraint], device="cpu",
                           tnorm='G')

from tqdm import tqdm

def check_constraints_acc(cur_program, cur_data):
    acc = 0
    total = 0
    for datanode in tqdm(cur_program.populate(cur_data, device="cpu"), "Manually Testing"):
        verify_constrains = datanode.verifyResultsLC()

        condition_list = []
        if not verify_constrains:
            continue

        for lc in verify_constrains:
            condition_list.append(1 if verify_constrains[lc]["satisfied"] == 100.0 else 0)
        constraints_keys = list(verify_constrains.keys())

        # This might be a better way for this to finding the label, but this work
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
        people2 = datanode_person.getAttribute(p2, 'local/softmax').argmax().item()
        people3 = datanode_person.getAttribute(p3, 'local/softmax').argmax().item()

        # Condition used in ER examples
        cond1_check = (work1 and people1) and (work2 and people2)
        cond2_check = (work2 and people2) or (work3 and people3)
        if condition_list[0] != cond1_check or condition_list[1] != cond2_check:
            raise Exception("verify_constrains cannot be used")

        # == check matching the label
        acc += int(constraint_labels == condition_list)
        total += 1

    return acc / total


print("Acc before training:", check_constraints_acc(program, data_list))
program.train(data_list, train_epoch_num=10, lr=1e-6, c_lr=1e-6, c_warmup_iters=-1, device="cpu")
print("Acc after training:", check_constraints_acc(program, data_list))