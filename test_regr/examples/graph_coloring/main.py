import sys
import pytest

sys.path.append('.')
sys.path.append('../..')


def model_declaration():
    import torch

    from regr.sensor.pytorch.sensors import ReaderSensor
    from regr.sensor.pytorch.relation_sensors import EdgeSensor, CompositionCandidateReaderSensor
    from regr.program import LearningBasedProgram
    from regr.program.model.pytorch import model_helper, PoiModel

    from graph import graph, world, city, world_contains_city, neighbor, city1, city2, firestationCity
    from sensors import DummyCityLearner

    graph.detach()

    # --- City
    world['index'] = ReaderSensor(keyword='world')
    city['index'] = ReaderSensor(keyword='city')
    city[world_contains_city] = EdgeSensor(city['index'], world['index'], relation=world_contains_city, forward=lambda x, _: torch.ones_like(x).unsqueeze(-1))


    def readNeighbors(index, data, arg1, arg2):
        city1, city2 = arg1, arg2
        if city1.getAttribute('index') in data[int(city2.getAttribute('index'))]:
            return True
        else:
            return False

    neighbor[city1.reversed, city2.reversed] = CompositionCandidateReaderSensor(city['index'], keyword='links', relations=(city1.reversed, city2.reversed), forward=readNeighbors)

    # --- Learners
    city[firestationCity] = DummyCityLearner('index')
    
    program = LearningBasedProgram(graph, model_helper(PoiModel, poi=[world, city, city[firestationCity], neighbor]))
    return program


@pytest.mark.gurobi
def test_graph_coloring_main():
    from reader import CityReader
    from graph import city, neighbor, firestationCity

    lbp = model_declaration()

    dataset = CityReader().run()  # Adding the info on the reader

    for datanode in lbp.populate(dataset=dataset):
        assert datanode != None
        assert len(datanode.getChildDataNodes()) == 9

        for child_node in datanode.getChildDataNodes():
            assert child_node.ontologyNode == city
            assert child_node.getAttribute('<' + firestationCity.name + '>')[0] == 0
            assert child_node.getAttribute('<' + firestationCity.name + '>')[1] == 1

        # call solver
        conceptsRelations = (firestationCity)  
        datanode.inferILPResults(*conceptsRelations, fun=None, minimizeObjective=True) 

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
                
    verifyResult = datanode.verifyResultsLC()


if __name__ == '__main__':
    pytest.main([__file__])
