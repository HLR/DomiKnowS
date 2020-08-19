import sys
import pytest

sys.path.append('.')
sys.path.append('../..')


def model_declaration():
    from regr.sensor.pytorch.sensors import ReaderSensor, TorchEdgeReaderSensor, ForwardEdgeSensor, ConstantSensor
    from regr.sensor.pytorch.query_sensor import CandidateSensor, CandidateReaderSensor
    from regr.program import LearningBasedProgram
    from regr.program.model.pytorch import PoiModel

    from graph2 import graph2, world, city, world_contains_city, cityLink, city1, city2, firestationCity
    from sensors import DummyCityLearner

    graph2.detach()

    # --- City
    world['index'] = ReaderSensor(keyword='world')
    world_contains_city['forward'] = TorchEdgeReaderSensor(to='index', keyword='city', mode='forward')

    # --- Neighbor
    city1['backward'] = ForwardEdgeSensor('index', to='city1', mode='backward')
    city2['backward'] = ForwardEdgeSensor('index', to='city2', mode='backward')
    
    def readCitylinks(datanodes_edges, datanode_concept1, datanode_concept2):
        return True
    
    cityLink['index'] = CandidateSensor(forward=readCitylinks, edges=[city1['backward'], city2['backward']])

    def readNeighbors(links, current_neighbers, city1, city2, _):
        if city1.getAttribute('index') in links[int(city2.getAttribute('index'))]:
            return True
        else:
            return False
        
    cityLink['neighbor'] = CandidateReaderSensor('index', keyword='links', forward=readNeighbors, edges=[city1['backward'], city2['backward']])

    # --- Learners
    city[firestationCity] = DummyCityLearner('index', edges=[world_contains_city['forward'], cityLink['neighbor']])
    city[firestationCity] = ConstantSensor(data=None, label=True)
    
    program = LearningBasedProgram(graph2, PoiModel)
    return program


@pytest.mark.gurobi
def test_graph_coloring_main():
    from reader import CityReader
    from graph2 import city, firestationCity, cityLink

    lbp = model_declaration()

    dataset = CityReader().run()  # Adding the info on the reader

    for datanode in lbp.populate(dataset=dataset, inference=True):
        assert datanode != None
        assert len(datanode.getChildDataNodes()) == 9

        for child_node in datanode.getChildDataNodes():
            assert child_node.ontologyNode == city
            assert child_node.getAttribute('<' + firestationCity.name + '>')[0] == 0
            assert child_node.getAttribute('<' + firestationCity.name + '>')[1] == 1

        # call solver
        conceptsRelations = (firestationCity, cityLink)  
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
