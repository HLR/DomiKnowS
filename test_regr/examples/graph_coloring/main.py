import sys

sys.path.append('.')
sys.path.append('../..')

import pytest
from reader import CityReader



def model_declaration(config):
    from regr.sensor.pytorch.sensors import ReaderSensor
    from regr.program import LearningBasedProgram
    from regr.program.model.pytorch import PoiModel


    from graph import graph, world, city, neighbor, world_contains_city, neighbor_city1, neighbor_city2, firestationCity

    from sensors import DummyLearner, DummyEdgeSensor, CustomReader, DummyLabelSensor

    graph.detach()

    world['raw'] = ReaderSensor(keyword='world')

    # Edge: sentence to word forward
    world_contains_city['forward'] = DummyEdgeSensor(
        'raw', mode='forward', keyword='raw')

    neighbor['raw'] = CustomReader(keyword='raw')
    neighbor['raw'] = CustomReader(keyword='raw')

    city[firestationCity] = DummyLearner('raw')
    city[firestationCity] = DummyLabelSensor(label=True)



    program = LearningBasedProgram(graph, PoiModel)
    return program



@pytest.mark.gurobi
def test_graph_coloring_main():
    from config import CONFIG
    lbp = model_declaration(CONFIG.Model)

    dataset = CityReader().run() # Adding the info on the reader

    for datanode in lbp.eval(dataset=dataset, inference=True):
        print(datanode)
        # call solver
        # conceptsRelations = [] # TODO: please fill this
        # tokenResult, pairResult, tripleResult = datanode.inferILPConstrains(*conceptsRelations, fun=None)

    print('I am here!')


if __name__ == '__main__':
    pytest.main([__file__])
