import logging
from domiknows.graph import Graph, Concept, Relation, andL, orL
from NER_utils import generate_dataset, reader_format
from domiknows.program.lossprogram import PrimalDualProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import ModuleSensor, FunctionalSensor
from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor

import torch

from test_regr.InferenceAPI.NER_utils import is_real_person

logging.basicConfig(level=logging.INFO)
Graph.clear()
Concept.clear()
Relation.clear()

with Graph(name='global') as graph:

    people = Concept(name='people')
    p1 = people(name='p1')
    p2 = people(name='p2')
    p3 = people(name='p3')
    p1_is_real = p1(name='p1_is_real')
    p2_is_real = p1(name='p2_is_real')
    p3_is_real = p1(name='p3_is_real')

    location = Concept(name='location')
    l1 = location(name='l1')
    l2 = location(name='l2')
    l3 = location(name='l3')

    pair = Concept(name="pair")
    people_arg1,location_arg1 = pair.has_a(people_arg1=people, location_arg1=location)
    work_in = pair(name="work_in")

    #andL(andL(p1_is_real("x"),work_in("z",path=('x', people_arg.reversed))),andL(p2_is_real("y"),work_in("t",path=('y', people_arg.reversed))))
    #orL(andL(p2_is_real("x"),work_in("z",path=('x', people_arg.reversed))),andL(p3_is_real("y"),work_in("t",path=('y', people_arg.reversed))))

data_list = generate_dataset(sample_num=100)
data_list = reader_format(data_list)


people["embedding"] = ReaderSensor(keyword="person1")
p1["embedding"] = ReaderSensor(keyword="person1")
p2["embedding"] = ReaderSensor(keyword="person2")
p3["embedding"] = ReaderSensor(keyword="person3")


class PeopleModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.layer = torch.nn.Linear(self.size, 2)

    def forward(self, p):
        return self.layer(p[0])

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
        return self.layer(torch.cat([p[0], l[0]]))


work_in_model=WorkModel(4)
pair[people_arg1.reversed,location_arg1.reversed] = JointSensor(p1["embedding"],l1["embedding"],forward=lambda p,l: (torch.ones(1,1),torch.ones(1,1)))
pair[work_in] = ModuleSensor(p1["embedding"],l1["embedding"],module=work_in_model,device="cpu")
pair[work_in] = ModuleSensor(p2["embedding"],l2["embedding"],module=work_in_model,device="cpu")
pair[work_in] = ModuleSensor(p3["embedding"],l3["embedding"],module=work_in_model,device="cpu")

program = PrimalDualProgram(graph,SolverModel,poi=[pair,people,location,p1,p2,p3,l1,l2,l3],device="cpu")
program.train(data_list, epochs=10, lr=0.001)
