import sys

sys.path.append('.')
sys.path.append('../..')

import pytest
from reader import CityReader

def model_declaration():
    from regr.sensor.pytorch.sensors import ReaderSensor
    from regr.program import LearningBasedProgram
    from regr.program.model.pytorch import PoiModel

    from graph import graph, world, city, world_contains_city, cityLink, city1, city2, firestationCity, neighbor

    from sensors import DummyCityEdgeSensor, DummyCityLearner, DummyCityLabelSensor
    from sensors import DummyCityLinkEdgeSensor, DummyCityLinkCandidateGenerator, DummyNeighborLearner, DummyCityLinkLabelSensor

    graph.detach()

    # --- City
    
    world['raw'] = ReaderSensor(keyword='world')
    city['raw'] = ReaderSensor(keyword='city')
    world_contains_city['forward'] = DummyCityEdgeSensor('raw', mode='forward', keyword='world_contains_city_edge', edges=[city['raw']])
    
    city[firestationCity] = DummyCityLearner('raw', edges=[world_contains_city['forward']])
    city[firestationCity] = DummyCityLabelSensor(label=True)

    # --- CityLink
    
    city1['backward'] = DummyCityLinkEdgeSensor('raw', mode='backward', keyword='city1', edges=[world_contains_city['forward']])
    city2['backward'] = DummyCityLinkEdgeSensor('raw', mode='backward', keyword='city2', edges=[world_contains_city['forward']])
    cityLink['raw'] = DummyCityLinkCandidateGenerator(edges=[city1['backward'], city2['backward']])
    
    cityLink[neighbor] = DummyNeighborLearner('raw')
    cityLink[neighbor] = DummyCityLinkLabelSensor(label=True)

    
    program = LearningBasedProgram(graph, PoiModel)
    return program


@pytest.mark.gurobi
def test_graph_coloring_main():
    from graph import city, neighbor, firestationCity
    lbp = model_declaration()

    dataset = CityReader().run()  # Adding the info on the reader

    for datanode in lbp.eval(dataset=dataset, inference=True):
        assert datanode != None
        assert len(datanode.getChildDataNodes()) == 9

        _dataset = next(CityReader().run())
        for child_node in datanode.getChildDataNodes():
            assert child_node.ontologyNode == city
            assert child_node.getAttribute('<' + firestationCity.name + '>')[0] == 0
            assert child_node.getAttribute('<' + firestationCity.name + '>')[1] == 1

        # call solver
        conceptsRelations = (firestationCity, neighbor)  
        datanode.inferILPConstrains(*conceptsRelations, fun=None, minimizeObjective=True) 
        
        result = []
        for child_node in datanode.getChildDataNodes():
            s = child_node.getAttribute('raw')
            f = child_node.getAttribute(firestationCity, 'ILP').item()
            if f > 0:
                r = (s, True)
            else:
                r = (s, False)
            result.append(r)
        
        for child_index, child_node in enumerate(datanode.getChildDataNodes()):
            if child_index + 1 == 1:
                assert child_node.getAttribute(firestationCity, 'ILP').item() == 0 #1
            elif child_index + 1 == 5: #6
                assert child_node.getAttribute(firestationCity, 'ILP').item() == 1
            else:
                assert child_node.getAttribute(firestationCity, 'ILP').item() == 0

if __name__ == '__main__':
    pytest.main([__file__])
