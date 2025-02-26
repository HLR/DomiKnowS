import logging, torch
from domiknows.graph import Graph, Concept, Relation, andL, orL
from NER_utils import generate_dataset, reader_format
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import ModuleSensor, FunctionalSensor
from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor
from domiknows.program.metric import MacroAverageTracker

logging.basicConfig(level=logging.INFO)
Graph.clear()
Concept.clear()
Relation.clear()

with Graph(name='global') as graph:

    p1 = Concept(name='p1')
    p2 = Concept(name='p2')
    p3 = Concept(name='p3')

    p1_is_real = p1(name='p1isreal')
    p2_is_real = p2(name='p2isreal')
    p3_is_real = p3(name='p3isreal')

    l1 = Concept(name='l1')
    l2 = Concept(name='l2')
    l3 = Concept(name='l3')

    pair1 = Concept(name="pair1")
    people_arg1,location_arg1 = pair1.has_a(people_arg1=p1, location_arg1=l1)
    pair2 = Concept(name="pair2")
    people_arg2, location_arg2 = pair2.has_a(people_arg2=p2, location_arg2=l2)
    pair3 = Concept(name="pair3")
    people_arg3, location_arg3 = pair3.has_a(people_arg3=p3, location_arg3=l3)

    work_in1 = pair1(name="workin1")
    work_in2 = pair2(name="workin2")
    work_in3 = pair3(name="workin3")

    constarint1 = andL(andL(p1_is_real("x"),work_in1("z",path=('x', people_arg1.reversed))),andL(p2_is_real("y"),work_in2("t",path=('y', people_arg2.reversed))))
    constarint2 = orL(andL(p2_is_real("x"),work_in2("z",path=('x', people_arg2.reversed))),andL(p3_is_real("y"),work_in3("t",path=('y', people_arg3.reversed))))

data_list = generate_dataset(sample_num=1)
data_list = reader_format(data_list)

p1["embedding"] = ReaderSensor(keyword="person1")
p2["embedding"] = ReaderSensor(keyword="person2")
p3["embedding"] = ReaderSensor(keyword="person3")

graph.constraint[constarint1] = ReaderSensor(keyword="condition_1",label=True)
graph.constraint[constarint2] = ReaderSensor(keyword="condition_2",label=True)

class PeopleModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.layer = torch.nn.Linear(self.size, 2)

    def forward(self, p):
        return self.layer(p[0]).unsqueeze(0)

is_real_person=PeopleModel(2)
p1[p1_is_real] = ModuleSensor(p1["embedding"],module=is_real_person,device="cpu")
p2[p2_is_real] = ModuleSensor(p2["embedding"],module=is_real_person,device="cpu")
p3[p3_is_real] = ModuleSensor(p3["embedding"],module=is_real_person,device="cpu")

l1["embedding"] = ReaderSensor(keyword="location1")
l2["embedding"] = ReaderSensor(keyword="location2")
l3["embedding"] = ReaderSensor(keyword="location3")

class WorkModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.layer = torch.nn.Linear(self.size, 2)

    def forward(self, p, l):
        return self.layer(torch.cat([p[0], l[0]])).unsqueeze(0)

work_in_model=WorkModel(4)
pair1[people_arg1.reversed,location_arg1.reversed] = JointSensor(p1["embedding"],l1["embedding"],forward=lambda *_ : (torch.ones(1,1),torch.ones(1,1)))
pair1[work_in1] = ModuleSensor(p1["embedding"],l1["embedding"],module=work_in_model,device="cpu")
pair2[people_arg2.reversed,location_arg2.reversed] = JointSensor(p2["embedding"],l2["embedding"],forward=lambda *_ : (torch.ones(1,1),torch.ones(1,1)))
pair2[work_in2] = ModuleSensor(p2["embedding"],l2["embedding"],module=work_in_model,device="cpu")
pair3[people_arg3.reversed,location_arg3.reversed] = JointSensor(p3["embedding"],l3["embedding"],forward=lambda *_ : (torch.ones(1,1),torch.ones(1,1)))
pair3[work_in3] = ModuleSensor(p3["embedding"],l3["embedding"],module=work_in_model,device="cpu")

# program = PrimalDualProgram(graph,SolverModel,poi=[p1,p2,p3,l1,l2,l3,pair1,pair2,pair3,graph.constraint],loss=MacroAverageTracker(NBCrossEntropyLoss()),device="cpu")
program = InferenceProgram(
    graph,
    SolverModel,
    poi=[p1,p2,p3,l1,l2,l3,pair1,pair2,pair3,graph.constraint],
    device="cpu"
)
program.train(data_list, epochs=1, lr=0.001, c_warmup_iters=0)
