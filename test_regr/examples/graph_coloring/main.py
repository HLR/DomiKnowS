import sys

sys.path.append('.')
sys.path.append('../..')

import pytest
from reader import CityReader

def model_declaration():
    from regr.sensor.pytorch.sensors import ReaderSensor
    from regr.program import LearningBasedProgram
    from regr.program.model.pytorch import PoiModel

    from graph import graph, world, city, world_contains_city, neighbor, city1, city2, firestationCity

    from sensors import DummyCityEdgeSensor, DummyCityLearner, DummyCityLabelSensor
    from sensors import DummyCityLinkEdgeSensor
    from regr.sensor.pytorch.query_sensor import CandidateReaderSensor

    graph.detach()

    # --- City
    
    world['raw'] = ReaderSensor(keyword='world')
    city['raw'] = ReaderSensor(keyword='city')
    city['index'] = ReaderSensor(keyword='city') # "index" key Required by CandidateReaderSensor ?
    world_contains_city['forward'] = DummyCityEdgeSensor('raw', mode='forward', to='world_contains_city_edge', edges=[city['raw']])

    # --- Neighbor
    
    # Not used ? - maybe should be used in define_inputs of query_sensor ?
    city1['backward'] = DummyCityLinkEdgeSensor('raw', mode='backward', to='city1', edges=[world_contains_city['forward']])
    city2['backward'] = DummyCityLinkEdgeSensor('raw', mode='backward', to='city2', edges=[world_contains_city['forward']])
    
    def readNeighbors(data, datanodes_edges, index, datanode_concept1, datanode_concept2):
        if index[1] + 1 in data[index[0] + 1]: # data contain 'links' from reader
            return 1
        else:
            return 0
        
    # "raw" is it right key?
    # First argument required ?!!
    neighbor['raw'] = CandidateReaderSensor(edges=[city1['backward'], city2['backward']], label=False, forward=readNeighbors, keyword='links')
    
    # --- Learners
    
    city[firestationCity] = DummyCityLearner('raw', edges=[world_contains_city['forward'], neighbor['raw']])
    city[firestationCity] = DummyCityLabelSensor(label=True)
    
    program = LearningBasedProgram(graph, PoiModel)
    return program


@pytest.mark.gurobi
def test_graph_coloring_main():
    from graph import city, neighbor, firestationCity
    lbp = model_declaration()

    dataset = CityReader().run()  # Adding the info on the reader
    #neighbor['raw'].fill_data(dataset) # Does not work
    
    for datanode in lbp.populate(dataset=dataset, inference=True):
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
                assert child_node.getAttribute(firestationCity, 'ILP').item() == 1
            elif child_index + 1 == 6:
                assert child_node.getAttribute(firestationCity, 'ILP').item() == 1
            else:
                assert child_node.getAttribute(firestationCity, 'ILP').item() == 0

if __name__ == '__main__':
    pytest.main([__file__])
