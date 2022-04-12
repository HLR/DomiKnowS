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
from regr.utils import setProductionLogMode

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
        entity,
        exact_before,
        locations,
        location,
        entity_location,
        entity_location_label
    )
    from graph_multi import (
        procedure_context,
        procedure_entities,
        procedure_locations,
        entity_rel,
        loc_rel,
        context_step,
        before_arg1,
        before_arg2,
        ebefore_arg1,
        ebefore_arg2,
        action_step,
        action_entity,
        lentity,
        lstep,
        llocation,
    )


    from regr.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ReaderSensor, FunctionalReaderSensor
    from regr.sensor.pytorch.learners import ModuleLearner
    from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor
    from regr.sensor.pytorch.query_sensor import DataNodeReaderSensor

    class JointFunctionalReaderSensor(JointSensor, FunctionalReaderSensor):
        pass


    procedure['id'] = ReaderSensor(keyword='ProcedureID')
    context['text'] = ReaderSensor(keyword='Context')
    entities['text'] = ReaderSensor(keyword='Entities')
    locations['text'] = ReaderSensor(keyword='Locations')

    def make_procedure(arg1m, arg2m, arg3m, data):
        total_procedures = len(arg1m) * len(arg2m) * len(arg3m)
        rel_links1 = torch.ones(total_procedures, len(arg1m))
        rel_links2 = torch.ones(total_procedures, len(arg2m))
        rel_links3 = torch.ones(total_procedures, len(arg3m))
    #     past1 = 0
    #     past2 = 0
    #     for i in range(total_procedures):
    #         rel_links1[i, past2: past2 + len(arg2m)] = 1
    #         past2 = past2 + len(arg2m)
    #         rel_links2[i, past1: past1 + len(arg1m)] = 1
    #         past1 = past1 + len(arg1m)

    #     for i in range(len(arg1m)):
    #         rel_links1[0, i: (i+1)*(len(arg2m) * len(arg3m))] = 1



        return rel_links1, rel_links2, rel_links3


    procedure[procedure_context.reversed, procedure_entities.reversed, procedure_locations.reversed] = JointFunctionalReaderSensor(context['text'], entities['text'], locations['text'], keyword="ProcedureID", forward=make_procedure)

    def read_initials(*prev, data):
        number = len(data)
        rel_links = torch.ones(number, 1)
        indexes = [i for i in range(number)]
    #     print(rel_links, data, indexes)
        return rel_links, data, indexes

    entity[entity_rel, 'text', 'index'] = JointFunctionalReaderSensor(entities['text'], keyword='Entity', forward=read_initials)

    step[context_step, 'text', 'index'] = JointFunctionalReaderSensor(context['text'], keyword='Step', forward=read_initials)

    location[loc_rel, 'text', 'index'] = JointFunctionalReaderSensor(locations['text'], keyword='Location', forward=read_initials)



    def make_before_connection(*prev, data):
        return data[0], data[1]

    before[before_arg1.reversed, before_arg2.reversed] = JointFunctionalReaderSensor(step['text'], keyword='before', forward=make_before_connection)

    exact_before[ebefore_arg1.reversed, ebefore_arg2.reversed] = JointFunctionalReaderSensor(step['text'], keyword='exact_before', forward=make_before_connection)


    def make_actions(r1, r2, entities, steps):
    #     print(r1, r2)
        all_actions = len(steps) * len(entities)
        link1 = torch.zeros(all_actions, len(steps))
        link2 = torch.zeros(all_actions, len(entities))
        for i in range(len(entities)):
            link2[i*len(steps):(i+1)*len(steps),i] = 1

        for j in range(all_actions):
            link1[j, j%len(steps)] = 1

    #     print(link1, link2)
    #     print(link1.shape, link2.shape)
    #     print("steps: ", len(steps))
    #     print("entities: ", len(entities))
        return link1, link2

    action[action_step.reversed, action_entity.reversed] = JointSensor(entity[entity_rel], step[context_step], entity['index'], step['index'], forward=make_actions)


    def make_entity_locations(r1, r2, r3, entities, steps, locations):
    #     print(r1, r2)
        all_actions = len(steps) * len(entities) * len(locations)
        link2 = torch.zeros(all_actions, len(steps))
        link1 = torch.zeros(all_actions, len(entities))
        link3 = torch.zeros(all_actions, len(locations))

        for i in range(len(entities)):
            link1[i*len(steps)*len(locations):(i+1)*len(steps)*len(locations), i] = 1

        for i in range(len(entities)):
            for j in range(len(steps)):
                start = i*len(steps)*len(locations)
                link2[start+(len(locations)*j): start +((j+1)*len(locations)), j] = 1

        for i in range(len(entities)):
            for j in range(len(steps)):
                for k in range(len(locations)):
                    start = i*len(steps)*len(locations) + (j*len(locations))
                    link3[start+k, k] = 1

        return link1, link2, link3


    entity_location[lentity.reversed, lstep.reversed, llocation.reversed] = JointSensor(entity[entity_rel], step[context_step], location[loc_rel], entity['index'], step['index'], location['index'], forward=make_entity_locations)

    def read_location_labels(*prevs, data):
    #     print(prevs[0].shape, prevs[1].shape)
    #     print(data.shape)
        c = torch.softmax(data, dim=-1)
        d = c.repeat(1, 1, 2)
        d = d.view(c.shape[0], c.shape[1], c.shape[2], 2)
        for k in range(d.shape[2]):
            d[:, :, k, 1] = 1 - d[:, :, k, 0]
    #     print(c.shape)

        d = d.view(c.shape[0] * c.shape[1] * c.shape[2], 2)
        return d

    entity_location[entity_location_label] = FunctionalReaderSensor(lentity.reversed, lstep.reversed, llocation.reversed, keyword="LocationLabel", forward=read_location_labels)

    def read_labels(*prevs, data):
    #     print(prevs[0].shape, prevs[1].shape)
    #     print(data.shape)
        c = data.view(data.shape[0] * data.shape[1], data.shape[2])
    #     print(c.shape)
        return c
    #     pass

    action[action_label] = FunctionalReaderSensor(action_step.reversed, action_entity.reversed, keyword="Action", forward=read_labels)

    program = SolverPOIProgram(graph, poi=(procedure, before, action, action_label, exact_before, entity_location, entity_location_label), 
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
    setProductionLogMode()
    dataset = ProparaReader(file="data/train.json")  # Adding the info on the reader

    # dataset = list(dataset)[16]

    # dataset = iter([dataset])

    #     lbp.test(dataset, device='auto')
    all_updates = []
    for datanode in lbp.populate(dataset, device="cpu"):
    #     tdatanode = datanode.findDatanodes(select = context)[0]
    #     print(len(datanode.findDatanodes(select = context)))
    #     print(tdatanode.getChildDataNodes(conceptName=step))
        datanode.inferILPResults(action, fun=None)
        final_output = {
            "id": datanode.getAttribute("id"),
            "steps": [],
            "actions": [],
            "steps_before": [],
            "actions_before": [],
        }

        entities_instances = datanode.findDatanodes(select=entity)
    #     print(len(entities))
        steps_instances = datanode.findDatanodes(select=step)
        actions = datanode.findDatanodes(select=action)
    #         print('a')

        for step_instance in steps_instances:
            a = step_instance.getAttribute('index')
            # rel = step.getRelationLinks(relationName=before)
            # print(rel)
        # print(len(steps_instances), "\n")
        for action_info in datanode.findDatanodes(select=action):
            c = action_info.getAttribute(action_label, "ILP")
            final_output["actions"].append(c)
            c1 = action_info.getAttribute(action_label)
            final_output["actions_before"].append(c1)

        final_output['actions'] = torch.stack(final_output['actions'])
        final_output['actions'] = final_output['actions'].view(len(entities_instances), len(steps_instances), 4)

        final_output['actions_before'] = torch.stack(final_output['actions_before'])
        final_output['actions_before'] = final_output['actions_before'].view(len(entities_instances), len(steps_instances), 4)

        all_updates.append(final_output)
    return all_updates


updated_data = main()

print(updated_data[0]['actions'].argmax(dim=-1))
print(updated_data[0]['actions_before'].argmax(dim=-1))
# import json
# with open("data/updated_info.json", "w") as f:
#     json.dump(updated_data, f)
