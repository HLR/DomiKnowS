import logging
from domiknows.graph import Graph, Concept, Relation, andL, orL
from NER_utils import generate_dataset, reader_format
from domiknows.program.lossprogram import PrimalDualProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import ModuleSensor
from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor

import torch

logging.basicConfig(level=logging.INFO)
Graph.clear()
Concept.clear()
Relation.clear()

with Graph(name='global') as graph:

    people = Concept(name='people')
    p1 = people(name='p1')
    p2 = people(name='p2')
    p3 = people(name='p3')

    location = Concept(name='location')
    l1 = location(name='l1')
    l2 = location(name='l2')
    l3 = location(name='l3')

    pair = Concept(name="pair")
    people_arg1,location_arg1 = pair.has_a(people_arg1=people, location_arg1=location)
    work_in = pair(name="work_in")

    andL(andL(p1("x"),work_in("z",path=('x', people_arg1.reversed))),andL(p2("y"),work_in("t",path=('y', people_arg1.reversed))))
    orL(andL(p2("x"),work_in("z",path=('x', people_arg1.reversed))),andL(p3("y"),work_in("t",path=('y', people_arg1.reversed))))

data_list = generate_dataset(sample_num=100)
data_list = reader_format(data_list)

location["embedding1"] = ReaderSensor(keyword="location1")
location["embedding2"] = ReaderSensor(keyword="location2")
location["embedding3"] = ReaderSensor(keyword="location3")
people["embedding1"] = ReaderSensor(keyword="person1")
people["embedding2"] = ReaderSensor(keyword="person2")
people["embedding3"] = ReaderSensor(keyword="person3")

class PeopleModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.layer = torch.nn.Linear(self.size, 1)

    def forward(self, p):
        return self.layer(p[0])

is_real_person=PeopleModel(2)
people[p1] = ModuleSensor(people["embedding1"],module=is_real_person,device="cpu")
people[p2] = ModuleSensor(people["embedding2"],module=is_real_person,device="cpu")
people[p3] = ModuleSensor(people["embedding3"],module=is_real_person,device="cpu")

class WorkModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.layer = torch.nn.Linear(self.size, 1)

    def forward(self, p, l):
        return self.layer(torch.cat([p[0], l[0]]))

work_in_model=WorkModel(4)
pair[people_arg1.reversed,location_arg1.reversed] = JointSensor(forward=lambda : (torch.ones(1,1),torch.ones(1,1)))
#pair[work_in] = ModuleSensor(#,module=work_in_model,device="cpu")


program = PrimalDualProgram(graph,SolverModel,poi=[people,location,pair],device="cpu")
program.train(data_list, epochs=10, lr=0.001)
