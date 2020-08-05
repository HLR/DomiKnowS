import sys
import pytest

sys.path.append('.')
sys.path.append('../..')


def model_declaration():
    from regr.sensor.pytorch.sensors import TorchSensor, ReaderSensor, TorchEdgeReaderSensor, ForwardEdgeSensor, ConstantSensor
    from regr.sensor.pytorch.query_sensor import CandidateReaderSensor
    from regr.program import LearningBasedProgram
    from regr.program.model.pytorch import model_helper, PoiModel

    from graph import graph, world, city, world_contains_city, neighbor, city1, city2, firestationCity
    from sensors import DummyCityLearner

    graph.detach()

    # --- City
    world['index'] = ReaderSensor(keyword='world')
    world_contains_city['forward'] = TorchEdgeReaderSensor(to='index', keyword='city', mode='forward')

    # --- Neighbor
    city1['backward'] = ForwardEdgeSensor('index', to='city1', mode='backward')
    city2['backward'] = ForwardEdgeSensor('index', to='city2', mode='backward')

    def readNeighbors(links, current_neighbers, city1, city2):
        if city1.getAttribute('index') in links[int(city2.getAttribute('index'))]:
            return True
        else:
            return False

    neighbor['index'] = CandidateReaderSensor(keyword='links', forward=readNeighbors)

    # --- Learners
    city[firestationCity] = DummyCityLearner('index', edges=[world_contains_city['forward']])
    
    program = LearningBasedProgram(graph, model_helper(PoiModel, poi=[
        (city[firestationCity], [next(city[firestationCity].find(TorchSensor))]),
        (neighbor['index'], [next(neighbor['index'].find(TorchSensor))])]))
    return program


@pytest.mark.gurobi
def test_graph_coloring_main():
    from reader import CityReader
    from graph import city, neighbor, firestationCity

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
        conceptsRelations = (firestationCity, neighbor)  
        datanode.inferILPConstrains(*conceptsRelations, fun=None, minimizeObjective=True) 

        result = []
        for child_node in datanode.getChildDataNodes():
            s = child_node.getAttribute('index')
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
