import pytest
import torch

from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import atMostL, forAllL, exactL, ifL, andL, notL
from domiknows.graph import combinationC
from domiknows.program.model.pytorch import PoiModel
from domiknows.graph import EnumConcept

@pytest.fixture(name='case')
def test_case():
    import torch
    from domiknows.utils import Namespace

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    case = {
        'process': ['1'],
        'entities': [
            'a', 'b', 'c', 'd'
        ],
        'steps': [
            0, 1, 2, 3, 4, 5, 6, 7
        ],
        'locations': [
            'loc1', 'loc2', 'loc3'
        ],
    }

    ### make probs and gt for the final_decision
    final_decision_gt = torch.zeros(len(case['steps']), len(case['entities']), len(case['locations']))
    sample_decision = torch.rand(len(case['steps']), len(case['entities']), len(case['locations']), 2)
    sample_decision_p = sample_decision.softmax(dim=-1)
    sample_decision_p = sample_decision_p[:, :, :, 1]
    ### get the index of the highest value in the last dimension of the tensor sample_decision
    sample_indexes = sample_decision_p.argmax(dim=-1)
    ### put that index equal to 1 in the final_decision_gt
    for i in range(len(case['steps'])):
        for j in range(len(case['entities'])):
            final_decision_gt[i, j, sample_indexes[i, j]] = 1

    ### make probs and gt for the action_decision
    action_decision_gt = torch.zeros(len(case['steps'])-1, len(case['entities']))
    sample_action_decision = torch.rand(len(case['steps'])-1, len(case['entities']), 6)
    sample_action_decision_p = sample_action_decision.softmax(dim=-1)
    ### get the index of the highest value in the last dimension of the tensor sample_decision
    sample_action_indexes = sample_action_decision_p.argmax(dim=-1)
    ### put that index equal to 1 in the final_decision_gt
    for i in range(len(case['steps']) - 1):
        for j in range(len(case['entities'])):
            action_decision_gt[i, j] = sample_action_indexes[i, j]
    
    ### adding action decision to the case
    case['action_decision'] = action_decision_gt.to(device)
    case['action_decision_p'] = sample_action_decision_p.to(device)
    case['original_action_probs'] = sample_action_decision.reshape(-1, 6).to(device)
    ### adding final decision to the case for the location triple
    case['final_decision'] = final_decision_gt.to(device)
    case['original_probs'] = sample_decision.reshape(-1, 2).to(device)
    case['sample_decision_p'] = sample_decision_p.to(device)
    case = Namespace(case)
    return case


Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    process = Concept(name='process')
    entities = Concept(name='entities')
    steps = Concept(name='steps')
    locations = Concept(name='locations')
    location = Concept(name='location')
    step = Concept(name='step')
    entity = Concept(name='entity')
    (rel_locations_contain_location, ) = locations.contains(location)
    (rel_steps_contain_step, ) = steps.contains(step)
    (rel_entities_contain_entity, ) = entities.contains(entity)
    (rel_process_entities, rel_process_steps, rel_process_locations, ) = process.has_a(arg_e=entities, arg_s=steps, arg_l=locations)
    # (rel_process_entities, rel_process_steps, rel_process_locations, ) = process.has_a(arg1=entities, arg2=steps, arg3=locations)

    exact_before = Concept(name='exact_before')
    (rel_exact_before_step1, rel_exact_before_step2, ) = exact_before.has_a(arg_eb_s1=step, arg_eb_s2=step)

    before = Concept(name='before')
    (rel_before_step1, rel_before_step2, ) = before.has_a(arg_b_s1=step, arg_b_s2=step)

    action_pair = Concept(name='action_pair')
    (rel_action_pair_step, rel_action_pair_entity, ) = action_pair.has_a(arg_s1=step, arg_e2=entity)
    action_decision = action_pair(name='action_decision', ConceptClass=EnumConcept, 
                                  values=["create", "destroy", "exist", "move", "prior", "post"])

    decision = Concept(name='decision')
    (rel_step, rel_entity, rel_location, ) = decision.has_a(arg1=step, arg2=entity, arg3=location)

    final_decision = decision(name='final_decision')
    # LC0: For all combinations of step and entity only one location can be true
    ### the proposed interface is below
    ## for x, y in combination(step, entity):
    ##     total= 0
    ##     for z in locations:
    ##         total += final_decision(x, y, z)
    ##     assert total == 1
    forAllL(
         combinationC(step, entity)('i', 'e'), #this is the search space, cartesian product is expected between options
         exactL(
             final_decision('x', path=(('i', rel_step.reversed), ('e', rel_entity.reversed))), 1
         ), # this is the condition that should hold for every assignment
     )

    ### Given the above LC, the following LCs are automatically generated
    ### for (entity, step) in zip(case.entities, case.steps):
    ###     specific_decisions = [decision for decision in decisions if decision[step] == step and decision[entity] == entity]
    ###     atMostL(specific_decisions, 1)
    ### decision(1, a, loc1), decision(1, a, loc2), decision(1, a, loc3)
    ### (1, b, loc1), (1, b, loc2), decision(1, b, loc3)
    ### (1, c, loc1), (1, c, loc2), (1, c, loc3)
    ### (1, d, loc1), (1, d, loc2), (1, d, loc3)
    ### (2, a, loc1), (2, a, loc2), (2, a, loc3)
    ### (2, b, loc1), (2, b, loc2), (2, b, loc3)
    ### (2, c, loc1), (2, c, loc2), (2, c, loc3)
    ### and so on!

    ### LC2: If the action for entity e and step i is move, then the location should be different in step i-1 and step i
    ifL(
        # action a1 is move, i is a1's step and e is action entity
        andL(
            action_decision.move('a1'), 
            step('i', path=('a1', rel_action_pair_step)),
            entity('e', path=('a1', rel_action_pair_entity)),
            final_decision('x', path=(('i', rel_step.reversed), ('e', rel_entity.reversed)))
            ), 
        andL(
            step('j', path=('i', rel_exact_before_step2.reversed, rel_exact_before_step1)),
            notL(final_decision('y', path=(('j', rel_step.reversed), ('e', rel_entity.reversed), ('x', rel_location, rel_location.reversed))))
        ),
        active = False
    )

    ### step('i') --> step('j', path=(attr('raw') = j.attr('raw') - 1))
    ### (x != y)


def model_declaration(config, case):
    from domiknows.program.program import LearningBasedProgram
    from test_regr.sensor.pytorch.sensors import TestSensor, TestEdgeSensor

    graph.detach()
    
    process['id'] = TestSensor(expected_outputs=case.process)
    entities['list'] = TestSensor(expected_outputs=[case.entities])
    steps['list'] = TestSensor(expected_outputs=[case.steps])
    locations['list'] = TestSensor(expected_outputs=[case.locations])

    # Edge: entities, steps, and locations to process
    process[rel_process_entities.reversed, rel_process_steps.reversed, rel_process_locations.reversed] = TestSensor(
        entities['list'], steps['list'], locations['list'],
        expected_inputs= ([case.entities], [case.steps], [case.locations]),
        expected_outputs= (
            torch.ones([1, 1]), torch.ones([1, 1]),torch.ones([1, 1]), case.process
            )
        )
    
    ### Edge: entity to entities
    entity[rel_entities_contain_entity, 'raw'] = TestSensor(
        entities['list'],
        expected_inputs=([case.entities], ),
        expected_outputs=(torch.ones((len(case.entities), 1)), case.entities, )
    )

    ### Edge: step to steps
    step[rel_steps_contain_step, 'raw'] = TestSensor(
        steps['list'],
        expected_inputs=([case.steps], ),
        expected_outputs=(torch.ones((len(case.steps), 1)), case.steps, )
    )

    ### Edge: location to locations
    location[rel_locations_contain_location, 'raw'] = TestSensor(
        locations['list'],
        expected_inputs=([case.locations], ),
        expected_outputs=(torch.ones((len(case.locations), 1)), case.locations, )
    )

    ### create two 2d tensor to be 1 when step i is before step j
    ### each tensor represents the edges for the relationship
    b1s = []
    b2s = []
    for st in range(len(case.steps)):
        step1 = st + 1
        if step1 < len(case.steps):
            b1 = torch.zeros(len(case.steps))
            b1[st] = 1
            b2 = torch.zeros(len(case.steps))
            b2[step1] = 1
            b1s.append(b1)
            b2s.append(b2)                
    b1s, b2s = torch.stack(b1s), torch.stack(b2s)
    

    ### creating the before pair between steps
    exact_before[rel_exact_before_step1.reversed, rel_exact_before_step2.reversed] = TestSensor(
        step['raw'],
        expected_inputs=(case.steps, ),
        expected_outputs=(b1s, b2s, )
    )

    ### reformat the following code to be more readable
    b1s = []
    b2s = []
    for st in range(len(case.steps)):
        for step1 in range(len(case.steps)):
            if st < step1:
                b1 = torch.zeros(len(case.steps))
                b1[st] = 1
                b2 = torch.zeros(len(case.steps))
                b2[step1] = 1
                b1s.append(b1)
                b2s.append(b2)
    b1s, b2s = torch.stack(b1s), torch.stack(b2s)

    ### creating the before pair between steps
    before[rel_before_step1.reversed, rel_before_step2.reversed] = TestSensor(
        step['raw'],
        expected_inputs=(case.steps, ),
        expected_outputs=(b1s, b2s, )
    )

    ### create two matrixes to be 1 when connecting to certain type of object, between actions and step, actions and entiy, skipping the first step since it does not have an action
    num_actions = (len(case.steps)-1) * len(case.entities)
    connection_actions_steps = torch.zeros(num_actions, len(case.steps))
    connection_actions_entities = torch.zeros(num_actions, len(case.entities))
    for i in range(len(case.steps)-1):
        for j in range(len(case.entities)):
            connection_actions_steps[i*len(case.entities) + j, i+1] = 1
            connection_actions_entities[i*len(case.entities) + j, j] = 1

    action_pair[rel_action_pair_step.reversed, rel_action_pair_entity.reversed] = TestSensor(
        step['raw'], entity['raw'],
        expected_inputs=(case.steps, case.entities),
        expected_outputs=(connection_actions_steps, connection_actions_entities, )
    )

    action_pair[action_decision] = TestSensor(
        action_pair[rel_action_pair_step.reversed], action_pair[rel_action_pair_entity.reversed],
        expected_inputs=(connection_actions_steps, connection_actions_entities),
        expected_outputs=case.original_action_probs
    )

    action_pair[action_decision] = TestSensor(
        label= True,
        expected_outputs=case.action_decision.flatten()
    )
    
    ### create three matrixes to be 1 when connecting to certain type of object
    total_decision_number = len(case.steps) * len(case.entities) * len(case.locations)
    connection_steps = torch.zeros(total_decision_number, len(case.steps))
    connection_entities = torch.zeros(total_decision_number, len(case.entities))
    connection_locations = torch.zeros(total_decision_number, len(case.locations))
    dummy_names = []
    for i in range(len(case.steps)):
        for j in range(len(case.entities)):
            for k in range(len(case.locations)):
                dummy_names.append(f"{case.steps[i]}_{case.entities[j]}_{case.locations[k]}")
                connection_steps[i*len(case.entities)*len(case.locations) + j*len(case.locations) + k, i] = 1
                connection_entities[i*len(case.entities)*len(case.locations) + j*len(case.locations) + k, j] = 1
                connection_locations[i*len(case.entities)*len(case.locations) + j*len(case.locations) + k, k] = 1

    ### Edge: create decision from entity, step, and location
    decision[rel_step.reversed, rel_entity.reversed, rel_location.reversed, 'text'] = TestSensor(
        step['raw'], entity['raw'], location['raw'],
        expected_inputs=(case.steps, case.entities, case.locations),
        expected_outputs=(connection_steps, connection_entities, connection_locations, dummy_names)
    )

    ### Random probabilities for the decision for each triplet
    decision[final_decision] = TestSensor(
        decision[rel_step.reversed], decision[rel_entity.reversed], decision[rel_location.reversed],
        expected_inputs=(connection_steps, connection_entities, connection_locations),
        expected_outputs=case.original_probs
    )

    decision[final_decision] = TestSensor(
        label=True,
        expected_outputs=case['final_decision'].flatten()
    )

    lbp = LearningBasedProgram(graph, **config)
    return lbp

@pytest.mark.gurobi
def test_main_conll04(case):
    
    import torch

    lbp = model_declaration(
        {
            'Model': PoiModel,
            'poi': (process, entities, steps, locations, step, before, exact_before, location, entity, decision, final_decision, action_pair),
            'loss': None,
            'metric': None,
        }, case)
    
    data = {}

    _, _, datanode, _ = lbp.model(data)
    
    #datanode.calculateLcLoss()
    #datanode.calculateLcLoss(sample = True, sampleSize = 100, sampleGlobalLoss = False)
    datanode.verifyResultsLC()
    
    datanode.inferILPResults()
    print(f"\nPrinting DataNode: {datanode}")
    for node in datanode.findDatanodes(select=decision):
        try:
            fd_label = node.getAttribute(decision, final_decision, 'label')
            fd_ilp = node.getAttribute(final_decision, 'ILP')
            assert fd_label.to('cpu').item() == fd_ilp.item()
        except AssertionError:
            print(f"AssertionError: {fd_label.to('cpu').item()} != {fd_ilp.item()}")
            sind = node.getAttribute("arg1.reversed").argmax().item()
            print("the step of the node is: ", sind)
            eind = node.getAttribute("arg2.reversed").argmax().item()
            print("the entity of the node is: ", eind)
            lind = node.getAttribute("arg3.reversed").argmax().item()
            print("the location of the node is: ", lind)
            print("the node probs are ", node.getAttribute(final_decision, 'local/softmax').to('cpu'))
            print("the original probs are: ", case['original_probs'][sind*len(case.entities)*len(case.locations) + eind*len(case.locations) + lind])
            print("the sample_prediction_prob is ", case['sample_decision_p'][sind, eind])
            raise
    
    for node in datanode.findDatanodes(select=action_pair):
        try:
            assert node.getAttribute(action_decision, 'label').to('cpu').item() == node.getAttribute(action_decision, 'ILP').argmax()
        except AssertionError:
            print(f"AssertionError: {node.getAttribute(action_decision, 'label').to('cpu').item()} != {node.getAttribute(action_decision, 'ILP').argmax().item()}")
            sind = node.getAttribute("arg_s1.reversed").argmax().item()
            print("the step of the node is: ", sind)
            eind = node.getAttribute("arg_e2.reversed").argmax().item()
            print("the entity of the node is: ", eind)
            print("the node probs are ", node.getAttribute(action_decision, 'local/softmax').to('cpu'))
            raise
                        
if __name__ == '__main__':
    pytest.main([__file__])
# case = test_case()
# test_main_conll04(case)