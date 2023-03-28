import pytest
import math
import torch
# import sys
# sys.path.append('.')
# sys.path.append('../../..')


from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import atMostL, forAllL, combinationL
from domiknows.program.model.pytorch import PoiModel


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
            1, 2, 3, 4, 5, 6, 7
        ],
        'locations': [
            'loc1', 'loc2', 'loc3'
        ],
    }
    final_decision_gt = torch.zeros(len(case['steps']), len(case['entities']), len(case['locations']))
    for i in range(len(case['steps'])):
        for j in range(len(case['entities'])):
            ### randomly select one between locations and put that to 1
            final_decision_gt[i, j, torch.randint(0, len(case['locations']), (1,))] = 1
    final_decision_gt = final_decision_gt.to(device)
    case['final_decision'] = final_decision_gt
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
    (rel_process_entities, rel_process_steps, rel_process_locations, ) = process.has_a(arg1=entities, arg2=steps, arg3=locations)

    decision = Concept(name='decision')
    (rel_step, rel_entity, rel_location, ) = decision.has_a(arg1=step, arg2=entity, arg3=location)

    final_decision = decision(name='final_decision')
    # LC0: For all combinations of step and entity only one location can be true
    ### the proposed interface is below
    forAllL(
         combinationL(step('i'), entity('e')), #this is the search space, cartesian product is expected between options
         atMostL( 
             final_decision('x', path=(('i', rel_step.reversed), ('e', rel_entity.reversed))), 1
         ), # this is the condition that should hold for every assignment
     )
        

def model_declaration(config, case):
    from domiknows.program.program import LearningBasedProgram
    from test_regr.sensor.pytorch.sensors import TestSensor, TestEdgeSensor

    graph.detach()
    
    # process['id'] = TestSensor(expected_outputs=case.process)
    entities['list'] = TestSensor(expected_outputs=[case.entities])
    steps['list'] = TestSensor(expected_outputs=[case.steps])
    locations['list'] = TestSensor(expected_outputs=[case.locations])

    # Edge: entities, steps, and locations to process
    process[rel_process_entities.reversed, rel_process_steps.reversed, rel_process_locations.reversed, 'id'] = TestSensor(
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
        expected_outputs=torch.rand(total_decision_number, 2)
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
            'poi': (process, entities, steps, locations, step, location, entity, decision, final_decision),
            'loss': None,
            'metric': None,
        }, case)
    
    data = {}

    _, _, datanode, _ = lbp.model(data)
    print(datanode)
                        
if __name__ == '__main__':
    pytest.main([__file__])
# case = test_case()
# test_main_conll04(case)

