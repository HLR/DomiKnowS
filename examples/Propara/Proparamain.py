import sys
import torch
# from data.reader import EmailSpamReader

sys.path.append('.')
sys.path.append('../..')

from typing import Any, Dict
from regr.sensor.pytorch.sensors import ReaderSensor, JointSensor
from regr.sensor.pytorch.relation_sensors import EdgeSensor
from regr.sensor.pytorch.sensors import ReaderSensor
from regr.program import LearningBasedProgram
from regr.program.model.pytorch import PoiModel
import torch
from torch import nn

from regr.data.reader import RegrReader
class ProparaReader(RegrReader):
    def getprocedureIDval(self, item):
        return [item['id']]
    def getentitiesval(self, item):
        return item['entities']
    def getstepsval(self, item):
        num_steps = len(item['steps']) + 1
        rel = torch.ones(num_steps,1)
        sentences = ["step 0 information"]
        sentences.extend(item['steps'])
        return rel, sentences
    
    def getnon_existenceval(self, item):
        values = []
        for step in range(len(item['steps']) + 1):
            values.append([1 - item['entity_step'][step][2], item['entity_step'][step][2]])
        return torch.tensor(values)
            
    def getknownval(self, item):
        values = []
        for step in range(len(item['steps']) + 1):
            values.append([1 - item['entity_step'][step][0], item['entity_step'][step][0]])
        return torch.tensor(values)
    
    def getunkownval(self, item):
        values = []
        for step in range(len(item['steps']) + 1):
            values.append([1 - item['entity_step'][step][1], item['entity_step'][step][1]])
        return torch.tensor(values)
    
    def getactionval(self, item):
        action1s = torch.diag(torch.ones(len(item['steps']) + 1) )[:-1]
        action2s = torch.diag(torch.ones(len(item['steps']) + 1) )[1:]
        raw = torch.zeros(len(item['steps']))
        return action1s, action2s, raw
    
    def getcreateval(self, item):
        actions = []
        for sid, step in enumerate(item['steps']):
            o = 0
            c = 0
            d = 0
            if sid == 0:
                prev_state = item['entity_step'][sid]
                continue
            else:
                o += (prev_state[0] * item['entity_step'][sid][0])
                o += (prev_state[0] * item['entity_step'][sid][1])
                o += (prev_state[1] * item['entity_step'][sid][0])
                o += (prev_state[1] * item['entity_step'][sid][1])
                o += (prev_state[2] * item['entity_step'][sid][2])
                d += (prev_state[0] * item['entity_step'][sid][2])
                d += (prev_state[1] * item['entity_step'][sid][2])
                c += (prev_state[2] * item['entity_step'][sid][1])
                c += (prev_state[2] * item['entity_step'][sid][0])
                actions.append([1-c, c])
                prev_state = item['entity_step'][sid]
        return actions
                    
    def getdestroyval(self, item):
        actions = []
        for sid, step in enumerate(item['steps']):
            o = 0
            c = 0
            d = 0
            if sid == 0:
                prev_state = item['entity_step'][sid]
                continue
            else:
                o += (prev_state[0] * item['entity_step'][sid][0])
                o += (prev_state[0] * item['entity_step'][sid][1])
                o += (prev_state[1] * item['entity_step'][sid][0])
                o += (prev_state[1] * item['entity_step'][sid][1])
                o += (prev_state[2] * item['entity_step'][sid][2])
                d += (prev_state[0] * item['entity_step'][sid][2])
                d += (prev_state[1] * item['entity_step'][sid][2])
                c += (prev_state[2] * item['entity_step'][sid][1])
                c += (prev_state[2] * item['entity_step'][sid][0])
                actions.append([1-d, d])
                prev_state = item['entity_step'][sid]
        return actions
    
    def getotherval(self, item):
        actions = []
        for sid, step in enumerate(item['steps']):
            o = 0
            c = 0
            d = 0
            if sid == 0:
                prev_state = item['entity_step'][sid]
                continue
            else:
                o += (prev_state[0] * item['entity_step'][sid][0])
                o += (prev_state[0] * item['entity_step'][sid][1])
                o += (prev_state[1] * item['entity_step'][sid][0])
                o += (prev_state[1] * item['entity_step'][sid][1])
                o += (prev_state[2] * item['entity_step'][sid][2])
                d += (prev_state[0] * item['entity_step'][sid][2])
                d += (prev_state[1] * item['entity_step'][sid][2])
                c += (prev_state[2] * item['entity_step'][sid][1])
                c += (prev_state[2] * item['entity_step'][sid][0])
                actions.append([1-o, o])
                prev_state = item['entity_step'][sid]
        return actions
    
    def getbeforeval(self, item):
        b1s = []
        b2s = []
        for step in range(len(item['steps']) + 1):
            b1 = torch.zeros(len(item['steps']) + 1)
            b1[step] = 1
            for step1 in range(len(item['steps']) + 1):
                b2 = torch.zeros(len(item['steps']) + 1)
                b2[step1] = 1
                b1s.append(b1)
                b2s.append(b2)
        return torch.stack(b1s), torch.stack(b2s)
    
    def getbefore_trueval(self, item):
        num_steps = len(item['steps']) + 1
        values = torch.zeros(num_steps * num_steps)
        for step in range(len(item['steps']) + 1):
            for step1 in range(step + 1, len(item['steps']) + 1):
                values[(step*num_steps)+step1] = 1
        return values
    
    
from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, nandL, eqL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    procedure = Concept("procedure")
    step = Concept("step")
    (procedure_contain_step, ) = procedure.contains(step)
#     entity = Concept("entity")
    non_existence = step("non_existence")
    unknown_loc = step("unknown_location")
    known_loc = step("known_location")
    before = Concept("before")
    (before_arg1, before_arg2) = before.has_a(arg1=step, arg2=step)
    action = Concept("action")
    (action_arg1, action_arg2) = action.has_a(arg1=step, arg2=step)
    create = action("create")
    destroy = action("destroy")
    other = action("other")

    #LC1 : An action can not be create, destroy and other at the same time
    nandL(create, destroy, other)
    
    #LC2 : An action should at least be one of the create, destroy or other
    #Don't know how to write LC2
    
    #LC3 : A step can not be known_loc, unknown_loc and non_existence at the same time
    nandL(known_loc, unknown_loc, non_existence)
    
    #LC4 : A step should at least be one of known_loc, unknown_loc or non_existence
    #Don't know how to write LC4
    
    #LC5 : If action is create then the first step should be non_existence and the second step can be either known_loc or unknown_loc
    ifL(create, ("x", "y", ), andL(non_existence, ("x", ), orL(known_loc, ("y", ), unknown_loc, ("y", ))))
    
    #LC 6 : If action is destroy, then first step should be either known_loc,or unknown_loc and the next step should be non_existence 
    ifL(destroy, ("x", "y", ), andL(orL(known_loc, ("x", ), unknown_loc, ("x", )), non_existence, ("y", )))
    
    #LC7 : There should be at most 1 create
    atMostL(1, ("x", ), create, ("x", ))
    
    #LC8 : There should be at most one destroy
    atMostL(1, ("x", ), destroy, ("x", ))
    
    #LC9 : If (x1,x2) is create and (y1, y2) is destroy, then the pair(x2,y2) if before should have the property "check" equal to 1.
    ifL(andL(create, ("x1", "x2", ), destroy, ("y1", "y2", )), eqL(before, "check", 1), ("x2", "y2", ))
    
    # No entity_step


from regr.sensor.pytorch.sensors import ReaderSensor, JointSensor
from regr.sensor.pytorch.relation_sensors import EdgeSensor

class EdgeReaderSensor(EdgeSensor, ReaderSensor):
    def __init__(self, *pres, relation, mode="forward", keyword=None, **kwargs):
        super().__init__(*pres, relation=relation, mode=mode, **kwargs)
        self.keyword = keyword
        self.data = None
        
class JoinReaderSensor(JointSensor, ReaderSensor):
    pass
            
class JoinEdgeReaderSensor(JointSensor, EdgeReaderSensor):
    pass


from regr.sensor.pytorch.sensors import ReaderSensor
from regr.program import LearningBasedProgram
from regr.program.model.pytorch import PoiModel
import torch
from torch import nn

def model_declaration():

    graph.detach()

    # --- City
    procedure['id'] = ReaderSensor(keyword='procedureID')
    step[procedure_contain_step.forward, 'text'] = JoinEdgeReaderSensor(procedure['id'], keyword='steps', relation=procedure_contain_step, mode="forward")
    # word[step_contains_word, 'raw'] = ReaderSensor(keyword='words')
#     entity['raw'] = ReaderSensor(keyword='entities')

    step[non_existence] = ReaderSensor(procedure_contain_step.forward, 'text', keyword='non_existence')
    step[unknown_loc] = ReaderSensor(procedure_contain_step.forward, 'text', keyword='known')
    step[known_loc] = ReaderSensor(procedure_contain_step.forward, 'text', keyword='unkown')
    
    step[non_existence] = ReaderSensor(procedure_contain_step.forward, 'text', keyword='non_existence')
    step[unknown_loc] = ReaderSensor(procedure_contain_step.forward, 'text', keyword='known')
    step[known_loc] = ReaderSensor(procedure_contain_step.forward, 'text', keyword='unkown')
    
    action[action_arg1.forward, action_arg2.forward, 'raw'] = JoinReaderSensor(step['text'], keyword='action')
    
    action[create] = ReaderSensor(action_arg1.forward, action_arg2.forward, 'raw', keyword='create')
    action[destroy] = ReaderSensor(action_arg1.forward, action_arg2.forward, 'raw', keyword='destroy')
    action[other] = ReaderSensor(action_arg1.forward, action_arg2.forward, 'raw', keyword='other')
    
    action[create] = ReaderSensor(keyword='create')
    action[destroy] = ReaderSensor(keyword='destroy')
    action[other] = ReaderSensor(keyword='other')
    
    before[before_arg1.forward, before_arg2.forward] = JoinReaderSensor(step['text'], keyword="before")
    
    before["check"] = ReaderSensor(before_arg1.forward, before_arg2.forward, keyword="before_true")
    before["check"] = ReaderSensor(before_arg1.forward, before_arg2.forward, keyword="before_true")
    
    program = LearningBasedProgram(graph, **{
        'Model': PoiModel,
#         'poi': (known_loc, unknown_loc, non_existence, other, destroy, create),
        'loss': None,
        'metric': None,
    })
    return program

def main():
    # set logger level to see training and testing logs
    import logging
    logging.basicConfig(level=logging.INFO)

    lbp = model_declaration()

    dataset = ProparaReader(file='updated_test_data.json')  # Adding the info on the reader

#     lbp.test(dataset, device='auto')

    for datanode in lbp.populate(dataset, device="cpu"):
        print('datanode:', datanode)
        data1 = datanode.findDatanodes(select = step)[0].getAttribute("text")
        print(data1)
        data1 = datanode.findDatanodes(select = step)[0].getAttribute(non_existence)
        print(data1)
        data1 = datanode.findDatanodes(select = action)[0].getAttribute("check")
        print(data1)
#         datanode.inferILPConstrains(create, destroy, other, non_existence, known_loc, unknown_loc, fun=None)
#         print('datanode:', datanode)

main()

