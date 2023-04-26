import sys
sys.path.append(".")
sys.path.append("../..")

import torch
from torch import nn
from reader import ProparaReader
from domiknows.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ReaderSensor, FunctionalReaderSensor, JointReaderSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.query_sensor import DataNodeReaderSensor
from domiknows.program import POIProgram, IMLProgram, SolverPOIProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, PRF1Tracker, DatanodeCMMetric
from domiknows.program.loss import NBCrossEntropyLoss, NBCrossEntropyIMLoss
from domiknows.utils import setProductionLogMode


def model_declaration():
    from graph_multi import (
        graph,
        procedure,
        context,
        step,
        entities,
        entity,
        input_entity,
        input_entity_alt,
        output_entity,
        output_entity_alt,
        before,
        exact_before,
        action,
        action_label,
        action_create,
        action_destroy,
        action_move,
        location_change,
        when_create,
        when_destroy,
        before_existence,
        after_existence,
        locations,
        location,
        entity_location,
        entity_location_label,
        entity_location_before_label,
        same_mention
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
        same_entity,
        same_location
    )


    from domiknows.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ReaderSensor, FunctionalReaderSensor
    from domiknows.sensor.pytorch.learners import ModuleLearner
    from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
    from domiknows.sensor.pytorch.query_sensor import DataNodeReaderSensor

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

        return rel_links1, rel_links2, rel_links3


    procedure[procedure_context.reversed, procedure_entities.reversed, procedure_locations.reversed] = JointFunctionalReaderSensor(context['text'], entities['text'], locations['text'], keyword="ProcedureID", forward=make_procedure)

    # def read_initials(*prev, data):
    #     number = len(data)
    #     rel_links = torch.ones(number, 1)
    #     indexes = [i for i in range(number)]
    # #     print(rel_links, data, indexes)
    #     return rel_links, data, indexes
    
    def read_initials(entities):
        data = entities[0]
        number = len(data)
        rel_links = torch.ones(number, 1)
        indexes = torch.tensor([i for i in range(number)])
        return rel_links, data, indexes

    entity[entity_rel, 'text', 'index'] = JointSensor(entities['text'], forward=read_initials)

    step[context_step, 'text', 'index'] = JointSensor(context['text'], forward=read_initials)

    location[loc_rel, 'text', 'index'] = JointSensor(locations['text'], forward=read_initials)

    entity[input_entity] = ReaderSensor(keyword='InputProb')
    entity[input_entity_alt] = ReaderSensor(keyword='AltInputProb')
    entity[output_entity] = ReaderSensor(keyword='OutputProb')
    entity[output_entity_alt] = ReaderSensor(keyword='AltOutputProb')

    def make_before_connection(*prev, data):
        return data[0], data[1]

    before[before_arg1.reversed, before_arg2.reversed] = JointFunctionalReaderSensor(step['text'], keyword='before', forward=make_before_connection)

    exact_before[ebefore_arg1.reversed, ebefore_arg2.reversed] = JointFunctionalReaderSensor(step['text'], keyword='exact_before', forward=make_before_connection)


    def make_actions(r1, r2, entities, steps):
        all_actions = len(steps) * len(entities)
        link1 = torch.zeros(all_actions, len(steps))
        link2 = torch.zeros(all_actions, len(entities))
        for i in range(len(entities)):
            link2[i*len(steps):(i+1)*len(steps),i] = 1

        for j in range(all_actions):
            link1[j, j%len(steps)] = 1

        return link1, link2

    action[action_step.reversed, action_entity.reversed] = JointSensor(entity[entity_rel], step[context_step], entity['index'], step['index'], forward=make_actions)
    
    def make_same_mentions(r1, r2, entities, locations):
        matches = []
        for eid, ent in enumerate(entities):
            for lid, loc in enumerate(locations):
                if ent == loc:
                    matches.append(eid, lid)
        link1 = torch.zeros(len(matches), len(entities))
        link2 = torch.zeros(len(matches), len(locations))

        for mid, match in enumerate(matches):
            link1[mid][match[0]] = 1
            link2[mid][match[1]] = 1

        return link1, link2
                

    # same_mention[same_entity.reversed, same_location.reversed] = JointSensor(entity[entity_rel], location[loc_rel], entity['text'], location['text'], forward=make_same_mentions)



    def make_entity_locations(r1, r2, r3, entities, steps, locations):
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
        c = data
        d = c.unsqueeze(-1)
        d = d.repeat(1, 1, 1, 2)
        d[:, :, :, 0] = 1 - d[:, :, :, 1]
        d = d.view(c.shape[0] * c.shape[1] * c.shape[2], 2)        
        return d

    entity_location[entity_location_label] = FunctionalReaderSensor(lentity.reversed, lstep.reversed, llocation.reversed, keyword="AfterLocationProb", forward=read_location_labels)

    entity_location[entity_location_before_label] = FunctionalReaderSensor(lentity.reversed, lstep.reversed, llocation.reversed, keyword="BeforeLocationProb", forward=read_location_labels)

    def read_labels(*prevs, data):
        c = data.view(data.shape[0] * data.shape[1], data.shape[2])
        return c
    
    def read_labels_to_bool(*prevs, data):
        ### add another shape to the last dimension of the data in c, (5, 6) --> (5, 6, 2)
        ### consider the never option as negative for all choices, plus other choices (5, 6, 2) --> (5, 5, 2)
        c_positive = data[:, 1:].unsqueeze(-1)
        c_negative = 1 - c_positive
        c = torch.cat((c_negative, c_positive), dim=-1)
        c = c.reshape(c.shape[0] * c.shape[1], c.shape[2])
        return c

    action[action_label] = FunctionalReaderSensor(action_step.reversed, action_entity.reversed, keyword="MultiActionProb", forward=read_labels)
    action[action_create] = FunctionalReaderSensor(action_step.reversed, action_entity.reversed, keyword="CreateProb", forward=read_labels)
    action[action_destroy] = FunctionalReaderSensor(action_step.reversed, action_entity.reversed, keyword="DestroyProb", forward=read_labels)
    action[action_move] = FunctionalReaderSensor(action_step.reversed, action_entity.reversed, keyword="MoveProb", forward=read_labels)
    action[location_change] = FunctionalReaderSensor(action_step.reversed, action_entity.reversed, keyword="ChangeProb", forward=read_labels)
    action[when_create] = FunctionalReaderSensor(action_step.reversed, action_entity.reversed, keyword="WhenCreateProb", forward=read_labels_to_bool)
    action[when_destroy] = FunctionalReaderSensor(action_step.reversed, action_entity.reversed, keyword="WhenDestroyProb", forward=read_labels_to_bool)
    action[before_existence] = FunctionalReaderSensor(action_step.reversed, action_entity.reversed, keyword="BeforeExistenceProb", forward=read_labels)
    action[after_existence] = FunctionalReaderSensor(action_step.reversed, action_entity.reversed, keyword="AfterExistenceProb", forward=read_labels)

    program = SolverPOIProgram(graph, 
                               poi=(
                                        procedure, before, action, entity, location,
                                        action_label, exact_before, 
                                        entity_location, entity_location_label, 
                                        entity_location_before_label, 
                                        when_create, when_destroy,
                                        before_existence, after_existence,
                                        action_create, action_destroy, action_move, location_change,
                                        input_entity, input_entity_alt, output_entity, output_entity_alt
                                    ), 
                               inferTypes=['ILP', 'local/argmax'],
                            #    inference_with=[action_label],
                            #    loss=MacroAverageTracker(NBCrossEntropyLoss()), 
                            #    metric={'ILP':PRF1Tracker(DatanodeCMMetric()),'argmax':PRF1Tracker(DatanodeCMMetric('local/argmax'))}
                               )
    return program


def main():
    from graph_multi import (
        graph,
        procedure,
        context,
        step,
        entities,
        entity,
        input_entity,
        input_entity_alt,
        output_entity,
        output_entity_alt,
        before,
        exact_before,
        action,
        action_label,
        action_create,
        action_destroy,
        action_move,
        location_change,
        when_create,
        when_destroy,
        before_existence,
        after_existence,
        locations,
        location,
        entity_location,
        entity_location_label,
        entity_location_before_label,
        same_mention
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
        same_entity,
        same_location
    )

    import logging

    logging.basicConfig(level=logging.INFO)

    lbp = model_declaration()
#     setProductionLogMode()
    dataset = ProparaReader(file="Tasks/T5Procedural/data/prepared_results.pt", type="_pt")  # Adding the info on the reader

    dataset = list(dataset)

    d2 = iter(dataset)
    dataset = iter(dataset)
    

    #     lbp.test(dataset, device='auto')
    all_updates = []
    for item_set, datanode in zip(d2, lbp.populate(dataset, device="cpu")):
    #     tdatanode = datanode.findDatanodes(select = context)[0]
    #     print(len(datanode.findDatanodes(select = context)))
    #     print(tdatanode.getChildDataNodes(conceptName=step))
        # datanode.inferILPResults(action_label, fun=None)
        final_output = {
            "id": datanode.findDatanodes(select=procedure)[0].getAttribute("id"),
            "steps": [],
            "entities": []
        }
        for _concept in [
            action_create, action_destroy, action_move, location_change,
            when_create, when_destroy,
            action_label,
            input_entity, input_entity_alt, output_entity, output_entity_alt,
            entity_location_before_label, entity_location_label,
            after_existence, before_existence
            ]:
            final_output[f"{_concept.name}"] = []
            final_output[f"{_concept.name}_before"] = []

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
        def assing_labels(node, concept, final_output):
            ### adding action_label information after ILP
            c = node.getAttribute(concept, "ILP")
            _map_list = ["create", "exists", "move", "destroy", "outside"]
            if not concept in {entity_location_label, entity_location_before_label}:
                if c.shape[-1] > 2:
                    c = torch.argmax(c, dim=-1).item()
                    c = _map_list[c]
                elif c.shape[-1] == 1:
                    c = int(c.item())
                    if c == 1:
                        c = "yes"
                    else:
                        c = "no"
            final_output[f"{concept.name}"].append(c)
            ### adding action_label information before ILP
            c1 = node.getAttribute(concept)
            if not concept in {entity_location_label, entity_location_before_label}:
                if c1.shape[-1] > 2:
                    c1 = torch.argmax(c1, dim=-1).item()
                    c1 = _map_list[c1]
                elif c1.shape[-1] == 1:
                    c1 = c1.item()
                    if c1 == 1:
                        c1 = "yes"
                    else:
                        c1 = "no"
                elif c1.shape[-1] == 2:
                    c1 = c1.argmax(-1).item()
                    if c1 == 1:
                        c1 = "yes"
                    else:
                        c1 = "no"
            final_output[f"{concept.name}_before"].append(c1)
            return final_output
        
        for action_info in datanode.findDatanodes(select=action):
            ### adding action_label information
            final_output = assing_labels(action_info, action_label, final_output)
            
            ### adding action_create information
            final_output = assing_labels(action_info, action_create, final_output)
            ### adding action_destroy information
            final_output = assing_labels(action_info, action_destroy, final_output)
            ### adding action_move information
            final_output = assing_labels(action_info, action_move, final_output)
            ### adding location_change information
            final_output = assing_labels(action_info, location_change, final_output)

            ### adding when_create information
            final_output = assing_labels(action_info, when_create, final_output)
            ### adding when_destroy information
            final_output = assing_labels(action_info, when_destroy, final_output)

            ### adding before_existence information
            final_output = assing_labels(action_info, before_existence, final_output)
            ### adding after_existence information
            final_output = assing_labels(action_info, after_existence, final_output)

        def fix_action_format(final_output, key, entities_instances, steps_instances):
            ### fix the format to be a list of list instead of a flat list, use the lenght of entities and steps
            ### the input is not a tensor but a list
            temp = []
            for i in range(len(entities_instances)):
                temp.append(final_output[key][i*len(steps_instances):(i+1)*len(steps_instances)])
            final_output[key] = temp


        for _concept in [
            action_label, action_create, action_destroy, action_move, location_change,
            when_create, when_destroy,
            before_existence, after_existence
        ]:
            fix_action_format(final_output, f"{_concept.name}", entities_instances, steps_instances)
            fix_action_format(final_output, f"{_concept.name}_before", entities_instances, steps_instances)
            

        # for _concept in [action_label, action_create, action_destroy, action_move, location_change, when_create, when_destroy, before_existence, after_existence]:
        #     final_output[f"{_concept.name}"] = torch.stack(final_output[f"{_concept.name}"])
        #     final_output[f"{_concept.name}_before"] = torch.stack(final_output[f"{_concept.name}_before"])

        #     if _concept == action_label:
        #         final_output[f"{_concept.name}"] = final_output[f"{_concept.name}"].reshape(len(entities_instances), len(steps_instances), 5)
        #         final_output[f"{_concept.name}_before"] = final_output[f"{_concept.name}_before"].reshape(len(entities_instances), len(steps_instances), 5)
        #     else:
        #         final_output[f"{_concept.name}"] = final_output[f"{_concept.name}"].reshape(len(entities_instances), len(steps_instances), 1)
        #         final_output[f"{_concept.name}_before"] = final_output[f"{_concept.name}_before"].reshape(len(entities_instances), len(steps_instances), 2)

        for entity_info in datanode.findDatanodes(select=entity):
            ### adding input_entity information
            final_output = assing_labels(entity_info, input_entity, final_output)
            ### adding input_entity_alt information
            final_output = assing_labels(entity_info, input_entity_alt, final_output)
            ### adding output_entity information
            final_output = assing_labels(entity_info, output_entity, final_output)
            ### adding output_entity_alt information
            final_output = assing_labels(entity_info, output_entity_alt, final_output)
        
        for entity_location_info in datanode.findDatanodes(select=entity_location):
            ### adding entity_location information
            final_output = assing_labels(entity_location_info, entity_location_label, final_output)
            ### adding entity_location_before information
            final_output = assing_labels(entity_location_info, entity_location_before_label, final_output)


        for _concept in [entity_location_before_label, entity_location_label]:
            final_output[f"{_concept.name}"] = torch.stack(final_output[f"{_concept.name}"])
            final_output[f"{_concept.name}"] = final_output[f"{_concept.name}"].reshape(len(entities_instances), len(steps_instances), -1).argmax(-1)
            temp = []
            for i in range(len(entities_instances)):
                temp.append([])
                for j in range(len(steps_instances)):
                    temp[-1].append(item_set['Location'][final_output[f"{_concept.name}"][i][j].item()])
            final_output[f"{_concept.name}"] = temp
            final_output[f"{_concept.name}_before"] = torch.stack(final_output[f"{_concept.name}_before"])
            final_output[f"{_concept.name}_before"] = final_output[f"{_concept.name}_before"].argmax(-1)
            final_output[f"{_concept.name}_before"] = final_output[f"{_concept.name}_before"].reshape(len(entities_instances), len(steps_instances), -1).argmax(-1)
            temp = []
            for i in range(len(entities_instances)):
                temp.append([])
                for j in range(len(steps_instances)):
                    temp[-1].append(item_set['Location'][final_output[f"{_concept.name}_before"][i][j].item()])
            final_output[f"{_concept.name}_before"] = temp

        final_output['entities'] = item_set['Entities'][0]
        final_output['steps'] = item_set['Step']
        final_output['locations'] = item_set['Location']

        def check_diff(final_output, key):
            ### check whether there are differences between key and key_before
            ### if there are differences, then return the index of the differences
            ### else return None
            diff = []
            if type(final_output[key][0]) != list:
                for i in range(len(final_output[key])):
                    if final_output[key][i] != final_output[key+"_before"][i]:
                        diff.append(i)
            else:
                for i in range(len(final_output[key])):
                    for j in range(len(final_output[key][i])):
                        if final_output[key][i][j] != final_output[key+"_before"][i][j]:
                            diff.append((i, j))
                            break
            if len(diff) == 0:
                return None
            else:
                return diff
            
        for _concept in [
            entity_location_before_label, entity_location_label,
            input_entity, input_entity_alt, output_entity, output_entity_alt,
            action_label, action_create, action_destroy, action_move, location_change,
            after_existence, before_existence,
        ]:
            print(f"\n{_concept.name} diff:")
            diff = check_diff(final_output, _concept.name)
            if diff:
                print(diff)
                

        all_updates.append(final_output)
        
        print("\nVerify Learned Results:")
        verifyResult = datanode.verifyResultsLC()
        if verifyResult:
            for lc in verifyResult:
                print("lc %s is %i%% satisfied by learned results"%(lc, verifyResult[lc]['satisfied']))

        print("\nVerify ILP Results:")
        verifyResultILP = datanode.verifyResultsLC(key = "/ILP")
        if verifyResultILP:
            for lc in verifyResultILP:
                print("lc %s is %i%% satisfied by ilp results"%(lc, verifyResultILP[lc]['satisfied']))
    
    return all_updates


updated_data = main()

print("\nResults from model before ILP")
print(updated_data[0]['actions_before'].argmax(dim=-1))

print("\nResults after ILP")
print(updated_data[0]['actions'].argmax(dim=-1))
# import json
# with open("data/updated_info.json", "w") as f:
#     json.dump(updated_data, f)
