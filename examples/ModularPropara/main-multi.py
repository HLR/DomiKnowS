import sys
sys.path.append(".")
sys.path.append("../..")
sys.path.append("../Popara")

import torch
from torch import nn
from reader import ProparaReader
from regr.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ReaderSensor, FunctionalReaderSensor, JointReaderSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from regr.sensor.pytorch.query_sensor import DataNodeReaderSensor
from regr.program import POIProgram, IMLProgram, SolverPOIProgram
from regr.program.primaldualprogram import PrimalDualProgram
from regr.program.model.pytorch import SolverModel
from regr.program.metric import MacroAverageTracker, PRF1Tracker, PRF1Tracker, DatanodeCMMetric
from regr.program.loss import NBCrossEntropyLoss, NBCrossEntropyIMLoss

def model_declaration():
    from graph_multi import (
        graph,
        procedure,
        context,
        step,
        entities,
        before,
        action,
        action_label,
        entity
    )
    from graph_multi import (
        procedure_context,
        procedure_entities,
        entity_rel,
        context_step,
        before_arg1,
        before_arg2,
        action_step,
        action_entity,
    )

    

    class JointFunctionalReaderSensor(JointSensor, FunctionalReaderSensor):
        pass

    procedure['id'] = ReaderSensor(keyword='ProcedureID')
    context['text'] = ReaderSensor(keyword='Context')
    entities['text'] = ReaderSensor(keyword='Entities')

    def make_procedure(arg1m, arg2m, data):
        total_procedures = len(arg1m) * len(arg2m)
        rel_links1 = torch.zeros(total_procedures, len(arg1m))
        rel_links2 = torch.zeros(total_procedures, len(arg2m))
        past1 = 0
        past2 = 0
        for i in range(total_procedures):
            rel_links1[i, past2: past2 + len(arg2m)] = 1
            past2 = past2 + len(arg2m)
            rel_links2[i, past1: past1 + len(arg1m)] = 1
            past1 = past1 + len(arg1m)

        return rel_links1, rel_links2


    procedure[procedure_context.reversed, procedure_entities.reversed] = JointFunctionalReaderSensor(context['text'], entities['text'], keyword="ProcedureID", forward=make_procedure)

    def read_initials(*prev, data):
        number = len(data)
        rel_links = torch.ones(number, 1)
        indexes = [i for i in range(number)]
#         print(rel_links, data, indexes)
        return rel_links, data, indexes

    entity[entity_rel, 'text', 'index'] = JointFunctionalReaderSensor(entities['text'], keyword='Entity', forward=read_initials)

    step[context_step, 'text', 'index'] = JointFunctionalReaderSensor(context['text'], keyword='Step', forward=read_initials)

    def make_actions(r1, r2, entities, steps):
#         print(r1, r2)
        all_actions = len(steps) * len(entities)
        link1 = torch.zeros(all_actions, len(steps))
        link2 = torch.zeros(all_actions, len(entities))
        for i in range(len(steps)):
            link1[i*len(entities):(i+1)*len(entities),i] = 1

        for j in range(all_actions):
            link2[j, j%len(entities)] = 1

    #     print(link1, link2)
    #     print("steps: ", len(steps))
    #     print("entities: ", len(entities))
        return link1, link2

    action[action_step.reversed, action_entity.reversed] = JointSensor(entity[entity_rel], step[context_step], entity['index'], step['index'], forward=make_actions)

    def read_labels(*prevs, data):
#         print(prevs[0].shape, prevs[1].shape)
        c = data.view(1, -1, *(data.size()[2:]))
#         print(c.squeeze(0).shape)
        return c.squeeze(0)
    #     pass

    action[action_label] = FunctionalReaderSensor(action_step.reversed, action_entity.reversed, keyword="Action", forward=read_labels)



    before[before_arg1.reversed, before_arg2.reversed] = JointReaderSensor(step["text"], keyword="before")

#     before["check"] = ReaderSensor(before_arg1.reversed, before_arg2.reversed, keyword="before_true")
#     before["check"] = ReaderSensor(keyword="before_true", label=True)

    program = SolverPOIProgram(graph, poi=(procedure, before, action, action_label), 
                               inferTypes=['ILP', 'local/argmax'], 
                               loss=MacroAverageTracker(NBCrossEntropyLoss()), 
                               metric={'ILP':PRF1Tracker(DatanodeCMMetric()),'argmax':PRF1Tracker(DatanodeCMMetric('local/argmax'))})
    return program


def main():
    from graph_multi import (
        graph,
        procedure,
        context,
        step,
        entities,
        before,
        action,
        action_label,
        entity,
    )
    
    from graph_multi import (
        procedure_context,
        procedure_entities,
        entity_rel,
        context_step,
        before_arg1,
        before_arg2,
        action_step,
        action_entity,
    )

    import logging

    logging.basicConfig(level=logging.INFO)

    lbp = model_declaration()

    dataset = ProparaReader(file="data/train.json")  # Adding the info on the reader

    #     lbp.test(dataset, device='auto')
    all_updates = []
    for datanode in lbp.populate(dataset, device="cpu"):
        datanode.inferILPResults(*action_label.enum, fun=None)
        
        final_output = {
            "id": datanode.getAttribute("id"),
            "steps": [],
            "actions": [],
            "steps_before": [],
            "actions_before": [],
        }
        
        for action_info in datanode.findDatanodes(select=action):
            c = action_info.getAttribute(action_label, "ILP").tolist()
            final_output["actions"].append(c)
            c = action_info.getAttribute(action_label).tolist()
            final_output["actions_before"].append(c)
            
        all_updates.append(final_output)
        
    #         print('datanode:', datanode)
    #         print('inference spam:', datanode.getAttribute(Spam, 'ILP'))
    #         print('inference regular:', datanode.getAttribute(Regular, 'ILP'))
    
    return all_updates


updated_data = main()
import json
with open("data/updated_info.json", "w") as f:
    json.dump(updated_data, f)
