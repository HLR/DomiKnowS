import sys
import pytest

sys.path.append('.')
sys.path.append('../..')


def model_declaration():
    import torch

    from domiknows.sensor.pytorch.sensors import ReaderSensor
    from domiknows.sensor.pytorch.query_sensor import DataNodeReaderSensor
    from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, CompositionCandidateSensor
    from domiknows.program import LearningBasedProgram
    from domiknows.program.model.pytorch import PoiModel

    from graph2 import graph2, world, city, world_contains_city, cityLink, city1, city2, firestationCity
    from sensors import DummyCityLearner

    graph2.detach()

    lcConcepts = {}
    for _, lc in graph2.logicalConstrains.items():
        if lc.headLC:  
            lcConcepts[lc.name] = lc.getLcConcepts()
    # --- City
    world['index'] = ReaderSensor(keyword='world')
    city['index'] = ReaderSensor(keyword='city')
    city[world_contains_city] = EdgeSensor(city['index'], world['index'], relation=world_contains_city, forward=lambda x, _: torch.ones_like(x).unsqueeze(-1))

    # --- Neighbor
    cityLink[city1.reversed, city2.reversed] = CompositionCandidateSensor(city['index'], relations=(city1.reversed, city2.reversed), forward=lambda *_, **__: True)

    def readNeighbors(*_, data, datanode):
        city1_node = datanode.relationLinks[city1.name][0]
        city2_node = datanode.relationLinks[city2.name][0]
        if city1_node.getAttribute('index') in data[int(city2_node.getAttribute('index'))]:
            return True
        else:
            return False
        
    cityLink['neighbor'] = DataNodeReaderSensor(city1.reversed, city2.reversed, keyword='links', forward=readNeighbors)

    # --- Learners
    city[firestationCity] = DummyCityLearner('index')
    # city[firestationCity] = ConstantSensor(data=None, label=True)
    
    program = LearningBasedProgram(graph2, PoiModel, poi=[world, city, cityLink])
    return program


@pytest.mark.gurobi
def test_graph_coloring_main():
    from reader import CityReader
    from graph2 import city, firestationCity, cityLink

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
