import sys
sys.path.append(".")
sys.path.append("../..")

import torch
from torch import nn
import math


from reader import ProparaReader
from domiknows.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ReaderSensor, FunctionalReaderSensor, JointReaderSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.query_sensor import DataNodeReaderSensor
from domiknows.program import POIProgram, IMLProgram, SolverPOIProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, PRF1Tracker, DatanodeCMMetric
from domiknows.program.loss import NBCrossEntropyLoss, NBCrossEntropyIMLoss
from domiknows.utils import setProductionLogMode, setDnSkeletonMode


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
    
    # def make_same_mentions(r1, r2, entities, locations):
    #     matches = []
    #     for eid, ent in enumerate(entities):
    #         for lid, loc in enumerate(locations):
    #             if ent == loc:
    #                 matches.append(eid, lid)
    #     link1 = torch.zeros(len(matches), len(entities))
    #     link2 = torch.zeros(len(matches), len(locations))

    #     for mid, match in enumerate(matches):
    #         link1[mid][match[0]] = 1
    #         link2[mid][match[1]] = 1

    #     return link1, link2    

    same_mention[same_entity.reversed, same_location.reversed] = JointReaderSensor(entity['text'], location['text'], keyword="SameMentions")

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
        sums = c.sum(-1).unsqueeze(-1).repeat(1, 1, c.shape[-1]).unsqueeze(-1)
        negatives = sums - d
        d = torch.cat((negatives, d), dim=-1)
        # d[:, :, :, 0] = 1 - d[:, :, :, 1]
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
        sums = data.sum(-1).unsqueeze(-1).repeat(1, data.shape[-1])[: , 1:].unsqueeze(-1)
        c_negative = sums - data[:, 1:].unsqueeze(-1)
        c_positive = data[:, 1:].unsqueeze(-1)
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
                                        same_mention,
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
    setProductionLogMode(no_UseTimeLog=True)
    setDnSkeletonMode(True)

    # dataset = ProparaReader(file="Tasks/T5Procedural/data/prepared_results.pt", type="_pt")  # Adding the info on the reader
    dataset = ProparaReader(file="Tasks/T5Procedural/data/normal_large/prepared_results.pt", type="_pt")  # Adding the info on the reader

    dataset = list(dataset)

    d2 = iter(dataset)
    dataset = iter(dataset)
    

    #     lbp.test(dataset, device='auto')
    all_updates = []
    changes, correct, correct_before, total = {}, {}, {}, {}
    correct_inferred, total_inferred = {}, {},
    actual_total, actual_correct = {}, {}
    actual_correct_before = {}
    actual_inferred_total, actual_inferred_correct = {}, {}
    final_loc_results = {}
    actions_matrix = {}
    actions_matrix_before = {}
    actions_matrix_inferred = {}
    for action_key in ["create", "exists", "destroy", "move", "outside"]:
        actions_matrix[action_key] = {}
        actions_matrix_before[action_key] = {}
        actions_matrix_inferred[action_key] = {}
        for action_key2 in ["create", "exists", "destroy", "move", "outside"]:
            actions_matrix[action_key][action_key2] = 0
            actions_matrix_before[action_key][action_key2] = 0
            actions_matrix_inferred[action_key][action_key2] = 0

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
        final_loc_out = {
            "id": datanode.findDatanodes(select=procedure)[0].getAttribute("id"),
            "steps": [],
            "entities": []
        }
        # print("stop here")
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
            # temp = []
            # for i in range(len(entities_instances)):
            #     temp.append([])
            #     for j in range(len(steps_instances)):
            #         temp[-1].append(item_set['Location'][final_output[f"{_concept.name}"][i][j].item()])
            # final_output[f"{_concept.name}"] = temp
            final_output[f"{_concept.name}_before"] = torch.stack(final_output[f"{_concept.name}_before"])
            final_output[f"{_concept.name}_before"] = final_output[f"{_concept.name}_before"][:, 1]
            final_output[f"{_concept.name}_before"] = final_output[f"{_concept.name}_before"].reshape(len(entities_instances), len(steps_instances), -1).argmax(-1)
            # temp = []
            # for i in range(len(entities_instances)):
            #     temp.append([])
            #     for j in range(len(steps_instances)):
            #         temp[-1].append(item_set['Location'][final_output[f"{_concept.name}_before"][i][j].item()])
            # final_output[f"{_concept.name}_before"] = temp

        final_output['entities'] = item_set['Entities'][0]
        final_output['steps'] = item_set['Step']
        final_output['locations'] = item_set['Location']
        final_loc_out['entities'] = item_set['Entities'][0]
        final_loc_out['steps'] = item_set['Step']
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
            if len(diff) == 0:
                return None
            else:
                return diff
            
        # for _concept in [
        #     entity_location_before_label, entity_location_label,
        #     input_entity, input_entity_alt, output_entity, output_entity_alt,
        #     action_label, action_create, action_destroy, action_move, location_change,
        #     after_existence, before_existence,
        # ]:
        #     print(f"\n{_concept.name} diff:")
        #     diff = check_diff(final_output, _concept.name)
        #     if diff:
        #         print(diff)
                
        assert torch.all(final_output['entity_location_before_label'][:, 1:] == final_output['entity_location_label'][:, :-1])
        assert final_output['output_entity'] == final_output['output_entity_alt']
        assert final_output['input_entity'] == final_output['input_entity_alt']
        all_updates.append(final_output)

        ### evaluate the accuracy before and after the changes
        key_data = [
            "MultiActionTrue",
            "AfterExistenceTrue", "BeforeExistenceTrue",
            "AfterLocationTrue", "BeforeLocationTrue",
            "AltInputTrue", "AltOutputTrue",
            "ChangeTrue", "CreateTrue", "DestroyTrue", "MoveTrue",
            "InputTrue", "OutputTrue",
            "WhenCreateTrue", "WhenDestroyTrue",
        ]
        key_decision = [
            action_label,
            after_existence, before_existence,
            entity_location_label, entity_location_before_label,
            input_entity_alt, output_entity_alt,
            location_change, action_create, action_destroy, action_move,
            input_entity, output_entity,
            when_create, when_destroy,
        ]
        def change_format_to_tensor(x):
            ### x is the values changed from function assign_label which we want to revert back
            ### to the original format
            _mapping = {"yes": 1, "no": 0}
            ### maping based on the index for "create", "exists", "move", "destroy", "outside"
            _mapping_action = {"create": 0, "exists": 1, "move": 2, "destroy": 3, "outside": 4}
            if type(x[0]) == list and x[0][0] in {"yes", "no"}:
                final_data = []
                for x1 in x:
                    final_data.append(torch.tensor([_mapping[_it] for _it in x1]))
                final_data = torch.stack(final_data)
                return final_data
            elif type(x[0]) == str and x[0] in {"yes", "no"}:
                final_data = torch.tensor([_mapping[_it] for _it in x])
                return final_data
            elif type(x[0]) == list and x[0][0] in {"create", "exists", "outside", "destroy", "move"}:
                final_data = []
                for x1 in x:
                    final_data.append(torch.tensor([_mapping_action[_it] for _it in x1]))
                final_data = torch.stack(final_data)
                return final_data
            else:
                if torch.is_tensor(x):
                    return x
                else:
                    return torch.tensor(x)

        kept_original_decisions = {}
        check_fixed_decisions = {}
        if "3" in item_set['Location']:
            index3 = item_set['Location'].index("3")
            if index3 < 3:
                assert index3 > 3
        else: index3 = -1

        if "1" in item_set['Location']:
            index1 = item_set['Location'].index("1")
            if index1 < 3:
                assert index1 > 3
        else: index1 = -1

        for key1, key2 in zip(key_data, key_decision):
            if not key2.name in total:
                total[key2.name] = 0
                correct[key2.name] = 0
                correct_before[key2.name] = 0
                changes[key2.name] = 0
                if "location" in key2.name:
                    actual_total[key2.name] = 0
                    actual_correct[key2.name] = 0
                    actual_correct_before[key2.name] = 0
            
            fixed_decision = final_output[key2.name]
            original_decision = final_output[key2.name+"_before"]
            fixed_decision = change_format_to_tensor(fixed_decision)
            original_decision = change_format_to_tensor(original_decision)
            ground_truth = item_set[key1]
            if not key2 in {action_label} and "location" in key2.name:
                # ground_truth[ground_truth==3] = 7752
                fixed_decision[fixed_decision==index3] = 0
                original_decision[original_decision==index3] = 0
            if "when" in key2.name:
                ground_truth = ground_truth - 1
                new_fixed = []
                for i in range(len(fixed_decision)):
                    if 1 in fixed_decision[i]:
                        ### find the first index in fixed_decision[i] that is 1 using tensor operation
                        positive_ind = torch.nonzero(fixed_decision[i] == 1)[0].item()
                        ### and add that to new_fixed
                        new_fixed.append(positive_ind)
                    else:
                        new_fixed.append(-1)
                fixed_decision = torch.tensor(new_fixed)
                new_original = []
                for i in range(len(original_decision)):
                    if 1 in original_decision[i]:
                        ### find the first index in original_decision[i] that is 1 using tensor operation
                        positive_ind = torch.nonzero(original_decision[i] == 1)[0].item()
                        ### and add that to new_original
                        new_original.append(positive_ind)
                    else:
                        new_original.append(-1)
                original_decision = torch.tensor(new_original)

            if key2 in {action_label, entity_location_before_label, entity_location_label, input_entity}:
                kept_original_decisions[key2.name] = original_decision
                check_fixed_decisions[key2.name] = fixed_decision

            changes[key2.name] += (original_decision.flatten() != fixed_decision.flatten()).sum().item()
            correct[key2.name] += (ground_truth.flatten() == fixed_decision.flatten()).sum().item()
            correct_before[key2.name] += (ground_truth.flatten() == original_decision.flatten()).sum().item()
            total[key2.name] += len(ground_truth.flatten())
            if "location" in key2.name:
                actual_correct[key2.name] += ((ground_truth.flatten() == fixed_decision.flatten()) & (ground_truth.flatten() != 0)).sum().item()
                actual_total[key2.name] += (ground_truth.flatten() != 0).sum().item()
                actual_correct_before[key2.name] += ((ground_truth.flatten() == original_decision.flatten()) & (ground_truth.flatten() != 0)).sum().item()
            if "action" in key2.name:
                _mapping = {0: "create", 1: "exists", 2: "move", 3: "destroy", 4: "outside"}
                for eid in range(ground_truth.shape[0]):
                    for sid in range(ground_truth.shape[1]):
                        gaction = ground_truth[eid][sid]
                        gaction = _mapping[gaction.item()]
                        gfixed = fixed_decision[eid][sid]
                        gfixed = _mapping[gfixed.item()]
                        goriginal = original_decision[eid][sid]
                        goriginal = _mapping[goriginal.item()]
                        actions_matrix[gaction][gfixed] += 1
                        actions_matrix_before[gaction][goriginal] += 1
        ### assertion code for location and action dependencies
        location_changes_fixed = check_fixed_decisions[entity_location_before_label.name] - check_fixed_decisions[entity_location_label.name]
        for eid in range(location_changes_fixed.shape[0]):
            for sid in range(location_changes_fixed.shape[1]):
                if location_changes_fixed[eid][sid] != 0:
                    if check_fixed_decisions[entity_location_before_label.name][eid][sid] == 0:
                        assert check_fixed_decisions[action_label.name][eid][sid].item() == 0 ### create
                    elif check_fixed_decisions[entity_location_label.name][eid][sid] == 0:
                        assert check_fixed_decisions[action_label.name][eid][sid].item() == 3 ### destroy
                    else:
                        assert check_fixed_decisions[action_label.name][eid][sid].item() == 2 ### move
                else:
                    if check_fixed_decisions[entity_location_before_label.name][eid][sid] == 0:
                        assert check_fixed_decisions[action_label.name][eid][sid].item() == 4 ### outside
                    else:
                        assert check_fixed_decisions[action_label.name][eid][sid].item() == 1 ### exists

        ### Logically and Sequentially fixing the decisions in action_label, entity_location_label, and entity_location_before_label
        sequentially_fixed_decisions = {
            action_label.name: torch.zeros(kept_original_decisions[action_label.name].shape),
            "entity_locations": torch.zeros(kept_original_decisions[entity_location_label.name].shape[0], kept_original_decisions[entity_location_label.name].shape[1]+1),
            }
        for eid in range(kept_original_decisions[action_label.name].shape[0]):
            entity_actions = kept_original_decisions[action_label.name][eid]
            entity_locations_after = kept_original_decisions[entity_location_label.name][eid]
            entity_locations_start = kept_original_decisions[entity_location_before_label.name][eid][0]
            exists = True
            ### "create", "exists", "move", "destroy", "outside"
            if entity_actions[0].item() in {0, 4}: ### if entity is being created or is outside at first
                sequentially_fixed_decisions["entity_locations"][eid][0] = 0 ### set the location to be none
                sequentially_fixed_decisions[action_label.name][eid][0] = entity_actions[0].item()
                exists = False
            else:
                sequentially_fixed_decisions["entity_locations"][eid][0] = entity_locations_start.item()
                sequentially_fixed_decisions[action_label.name][eid][0] = entity_actions[0].item()
            for sid in range(0, kept_original_decisions[action_label.name].shape[1]):
                if entity_actions[sid].item() == 0: ### if entity is being created
                    if exists:
                        sequentially_fixed_decisions[action_label.name][eid][sid] = 1
                        sequentially_fixed_decisions["entity_locations"][eid][sid+1] = sequentially_fixed_decisions["entity_locations"][eid][sid]
                    else:
                        sequentially_fixed_decisions[action_label.name][eid][sid] = 0
                        if not entity_locations_after[sid].item() in {0, index3, index1}:
                            sequentially_fixed_decisions["entity_locations"][eid][sid+1] = entity_locations_after[sid].item()
                        else:
                            sequentially_fixed_decisions["entity_locations"][eid][sid+1] = 1
                        exists = True
                elif entity_actions[sid].item() == 2: ### if entity is moved
                    if not exists:
                        sequentially_fixed_decisions[action_label.name][eid][sid] = 0
                        if not entity_locations_after[sid].item() in {0, index3, index1}:
                            sequentially_fixed_decisions["entity_locations"][eid][sid+1] = entity_locations_after[sid].item()
                        else:
                            sequentially_fixed_decisions["entity_locations"][eid][sid+1] = 1
                        exists = True
                    else:
                        sequentially_fixed_decisions[action_label.name][eid][sid] = 2
                        if not entity_locations_after[sid].item() in {0, index3, index1}:
                            sequentially_fixed_decisions["entity_locations"][eid][sid+1] = entity_locations_after[sid].item()
                        else:
                            sequentially_fixed_decisions["entity_locations"][eid][sid+1] = 1
                elif entity_actions[sid].item() == 3: ### if entity is destroyed
                    if exists:
                        sequentially_fixed_decisions[action_label.name][eid][sid] = 3
                        sequentially_fixed_decisions["entity_locations"][eid][sid+1] = 0
                        exists = False
                    else:
                        sequentially_fixed_decisions[action_label.name][eid][sid] = 1
                        sequentially_fixed_decisions["entity_locations"][eid][sid+1] = sequentially_fixed_decisions["entity_locations"][eid][sid]
                elif entity_actions[sid].item() == 4: ### if entity is outside
                    if exists:
                        sequentially_fixed_decisions[action_label.name][eid][sid] = 3
                        sequentially_fixed_decisions["entity_locations"][eid][sid+1] = 0
                        exists = False
                    else:
                        sequentially_fixed_decisions[action_label.name][eid][sid] = 4
                        sequentially_fixed_decisions["entity_locations"][eid][sid+1] = 0
                elif entity_actions[sid].item() == 1: ### if entity exists
                    if exists:
                        sequentially_fixed_decisions[action_label.name][eid][sid] = 1
                        sequentially_fixed_decisions["entity_locations"][eid][sid+1] = sequentially_fixed_decisions["entity_locations"][eid][sid]
                    else:
                        sequentially_fixed_decisions[action_label.name][eid][sid] = 0
                        if not entity_locations_after[sid].item() in {0, index3, index1}:
                            sequentially_fixed_decisions["entity_locations"][eid][sid+1] = entity_locations_after[sid].item()
                        else:
                            sequentially_fixed_decisions["entity_locations"][eid][sid+1] = 1
                        exists = True


        key_set = [
            action_label,
            after_existence, before_existence,
            entity_location_label, entity_location_before_label,
            input_entity_alt, output_entity_alt,
            location_change, action_create, action_destroy, action_move,
            input_entity, output_entity,
            when_create, when_destroy,
        ]
        key_grs = [
            "MultiActionTrue",
            "AfterExistenceTrue", "BeforeExistenceTrue",
            "AfterLocationTrue", "BeforeLocationTrue",
            "AltInputTrue", "AltOutputTrue",
            "ChangeTrue", "CreateTrue", "DestroyTrue", "MoveTrue",
            "InputTrue", "OutputTrue",
            "WhenCreateTrue", "WhenDestroyTrue",
        ]
        for key_gr, key in zip(key_grs, key_set):
            ground_truth = item_set[key_gr]
            if not key.name in total_inferred:
                correct_inferred[key.name] = 0
                total_inferred[key.name] = 0
                if "location" in key.name:
                    actual_inferred_correct[key.name] = 0
                    actual_inferred_total[key.name] = 0

            if key == action_label:
                # inferred_answer = kept_original_decisions[key.name]
                inferred_answer = sequentially_fixed_decisions[key.name]
            elif key == entity_location_label:
                # inferred_answer = kept_original_decisions[key.name]
                inferred_answer = sequentially_fixed_decisions["entity_locations"][:, 1:]
            elif key == entity_location_before_label:
                # inferred_answer = torch.zeros(kept_original_decisions[entity_location_before_label.name].shape)
                # for i in range(inferred_answer.shape[0]):
                #     for j in range(inferred_answer.shape[1]):
                #         if j == 0:
                #             inferred_answer[i][j] = kept_original_decisions[entity_location_before_label.name][i][j].item()
                #         else:
                #             inferred_answer[i][j] = kept_original_decisions[entity_location_label.name][i][j-1].item()
                inferred_answer = sequentially_fixed_decisions["entity_locations"][:, :-1]
            elif key == after_existence:
                inferred_answer = torch.zeros(kept_original_decisions[entity_location_label.name].shape)
                # inferred_answer[kept_original_decisions['entity_location_label'] != 5839] = 1
                inferred_answer[sequentially_fixed_decisions["entity_locations"][:, 1:] != 0] = 1
                
            elif key == before_existence:
                inferred_answer = torch.zeros(kept_original_decisions[entity_location_before_label.name].shape)
                # inferred_answer[kept_original_decisions['entity_location_label'] != 5839] = 1
                inferred_answer[sequentially_fixed_decisions["entity_locations"][:, :-1] != 0] = 1
                
                # inferred_answer = torch.zeros(kept_original_decisions[entity_location_before_label.name].shape)
                # for i in range(inferred_answer.shape[0]):
                #     for j in range(inferred_answer.shape[1]):
                #         if j == 0:
                #             if kept_original_decisions[entity_location_before_label.name][i][j] != 5839:
                #                 inferred_answer[i][j] = 1
                #         else:
                #             if kept_original_decisions[entity_location_label.name][i][j-1] != 5839:
                #                 inferred_answer[i][j] = 1
            elif key in {input_entity_alt, input_entity}:
                inferred_answer = torch.zeros(kept_original_decisions[entity_location_before_label.name].shape[0])
                for i in range(inferred_answer.shape[0]):
                    # if kept_original_decisions[entity_location_before_label.name][i][0] != 5839:
                    if sequentially_fixed_decisions["entity_locations"][i][0] != 0:
                        inferred_answer[i] = 1
            elif key in {output_entity_alt, output_entity}:
                inferred_answer = torch.zeros(kept_original_decisions[entity_location_label.name].shape[0])
                for i in range(inferred_answer.shape[0]):
                    # if kept_original_decisions[entity_location_label.name][i][-1] != 5839:
                    if sequentially_fixed_decisions["entity_locations"][i][-1] != 0:  
                        inferred_answer[i] = 1
            elif key == location_change:
                inferred_answer = torch.zeros(kept_original_decisions[entity_location_label.name].shape)
                for i in range(inferred_answer.shape[0]):
                    # cur_loc = kept_original_decisions[entity_location_before_label.name][i][0].item()
                    cur_loc = sequentially_fixed_decisions['entity_locations'][i][0].item()
                    # for j in range(inferred_answer.shape[1]-1):
                    for j in range(inferred_answer.shape[1]):
                        # if cur_loc != kept_original_decisions[entity_location_label.name][i][j].item():
                        if cur_loc != sequentially_fixed_decisions['entity_locations'][i][j].item():
                            inferred_answer[i][j] = 1
                        # cur_loc = kept_original_decisions[entity_location_label.name][i][j].item()
                        cur_loc = sequentially_fixed_decisions['entity_locations'][i][j].item()
                        
            elif key == action_create:
                inferred_answer = torch.zeros(kept_original_decisions[action_label.name].shape)
                ### "create", "exists", "move", "destroy", "outside"
                inferred_answer[sequentially_fixed_decisions[action_label.name] == 0] = 1
            elif key == action_destroy:
                inferred_answer = torch.zeros(kept_original_decisions[action_label.name].shape)
                inferred_answer[sequentially_fixed_decisions[action_label.name] == 3] = 1
            elif key == action_move:
                inferred_answer = torch.zeros(kept_original_decisions[action_label.name].shape)
                inferred_answer[sequentially_fixed_decisions[action_label.name] == 2] = 1
            elif key == when_create:
                inferred_answer = torch.zeros(kept_original_decisions[action_label.name].shape[0])
                for i in range(kept_original_decisions[action_label.name].shape[0]):
                    check = False
                    for j in range(kept_original_decisions[action_label.name].shape[1]):
                        if sequentially_fixed_decisions[action_label.name][i][j] == 0:
                            inferred_answer[i] = j
                            check = True
                            break
                    if not check:
                        inferred_answer[i] = -1
            elif key == when_destroy:
                inferred_answer = torch.zeros(kept_original_decisions[action_label.name].shape[0])
                for i in range(kept_original_decisions[action_label.name].shape[0]):
                    check = False
                    for j in range(kept_original_decisions[action_label.name].shape[1]):
                        if sequentially_fixed_decisions[action_label.name][i][j] == 3:
                            inferred_answer[i] = j
                            check = True
                            break
                    if not check:
                        inferred_answer[i] = -1

            if not key in {action_label} and "location" in key2.name:
                # ground_truth[ground_truth==index3] = 1
                inferred_answer[inferred_answer==index3] = 0
            if "when" in key.name:
                ground_truth = ground_truth - 1

            correct_inferred[key.name] += (ground_truth.flatten() == inferred_answer.flatten()).sum().item()
            total_inferred[key.name] += len(ground_truth.flatten())
            if "location" in key.name:
                actual_inferred_correct[key.name] += ((ground_truth.flatten() == inferred_answer.flatten()) & (ground_truth.flatten() != 0)).sum().item()
                actual_inferred_total[key.name] += (ground_truth.flatten() != 0).sum().item()
            if "action" in key.name:
                _mapping = {0: "create", 1: "exists", 2: "move", 3: "destroy", 4: "outside"}
                for eid in range(ground_truth.shape[0]):
                    for sid in range(ground_truth.shape[1]):
                        gaction = ground_truth[eid][sid]
                        gaction = _mapping[gaction.item()]
                        ginferred = inferred_answer[eid][sid]
                        ginferred = _mapping[ginferred.item()]
                        actions_matrix_inferred[gaction][ginferred] += 1

        # assert correct_before[action_label.name] == correct_inferred[action_label.name]
        assert total[action_label.name] == total_inferred[action_label.name]
        # assert correct_before[entity_location_label.name] == correct_inferred[entity_location_label.name]
        assert total[entity_location_label.name] == total_inferred[entity_location_label.name]
        updated_locations = check_fixed_decisions[entity_location_label.name]
        initial_updated_locations = check_fixed_decisions[entity_location_before_label.name][:,0]
        updated_locations = torch.cat((initial_updated_locations.unsqueeze(-1), updated_locations), dim=-1)
        temp = []
        for i in range(updated_locations.shape[0]):
            temp.append([])
            for j in range(updated_locations.shape[1]):
                an = item_set['Location'][updated_locations[i][j].item()]
                an = an.split(" ")
                temp[-1].append(an)
        final_loc_out['locations'] = temp
        final_loc_results[final_loc_out['id']] = final_loc_out
  
        # print("\nVerify Learned Results:")
        # verifyResult = datanode.verifyResultsLC()
        # if verifyResult:
        #     for lc in verifyResult:
        #         if 'ifSatisfied' in verifyResult[lc] and not math.isnan(verifyResult[lc]['ifSatisfied']):
        #             print("lc %s is %i%% IfSatisfied by learned results"%(lc, verifyResult[lc]['satisfied']))
        #         else:
        #             if verifyResult[lc]['satisfied'] == verifyResult[lc]['satisfied']:
        #                 print("lc %s is %i%% satisfied by learned results"%(lc, verifyResult[lc]['satisfied']))
        #             else:
        #                 print("lc %s cannot be verified for learned results - check if lc is correct"%(lc))


        # print("\nVerify ILP Results:")
        # verifyResultILP = datanode.verifyResultsLC(key = "/ILP")
        # if verifyResultILP:
        #     for lc in verifyResultILP:
        #         if 'ifSatisfied' in verifyResultILP[lc] and not math.isnan(verifyResultILP[lc]['ifSatisfied']):
        #             print("lc %s is %i%% IfSatisfied by learned results"%(lc, verifyResultILP[lc]['satisfied']))
        #         else:
        #             if verifyResultILP[lc]['satisfied'] == verifyResultILP[lc]['satisfied']:
        #                 print("lc %s is %i%% satisfied by ilp results"%(lc, verifyResultILP[lc]['satisfied']))
        #             else:
        #                 print("lc %s cannot be verified for ilp results - check if lc is correct"%(lc))
    for key in total:
        print(key, correct[key]/total[key])
        print(key+"_before", correct_before[key]/total[key])
        print(key+" inferred", correct_inferred[key]/total_inferred[key])
        print(key + " changes", changes[key])
        if "location" in key:
            print(key + " actual", actual_correct[key]/actual_total[key])
            print(key + " actual_before", actual_correct_before[key]/actual_total[key])
            print(key + " actual_inferred", actual_inferred_correct[key]/actual_inferred_total[key])
    for key in actions_matrix_inferred:
        print(f"True key {key}")
        for key1 in actions_matrix_inferred:
            print(f"pred {key1}")
            print(f"after ILP: count {actions_matrix[key][key1]}")
            print(f"before ILP: count {actions_matrix_before[key][key1]}")
            print(f"through inferred results : count {actions_matrix_inferred[key][key1]}")
    
    torch.save(final_loc_results, "updated_loc_info.pt")
    return all_updates


updated_data = main()

# print("\nResults from model before ILP for actions_before")
# if 'actions_before' in updated_data[0]:
#     print(updated_data[0]['actions_before'].argmax(dim=-1))

# print("\nResults after ILP for action")
# if 'actions' in updated_data[0]:
#     print(updated_data[0]['actions'].argmax(dim=-1))
    
# import json
# with open("data/updated_info.json", "w") as f:
#     json.dump(updated_data, f)