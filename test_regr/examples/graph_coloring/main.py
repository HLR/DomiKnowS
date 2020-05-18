import sys

sys.path.append('.')
sys.path.append('../..')

import pytest



def model_declaration(config):
    from regr.sensor.pytorch.sensors import ReaderSensor
    

    from graph import graph, world, city, neighbor, world_contains_city, neighbor_city1, neighbor_city2, firestationCity

    from sensors import DummyLearner, DummyEdgeSensor, CustomReader

    graph.detach()

    world['raw'] = ReaderSensor(keyword='world')

    # Edge: sentence to word forward
    world_contains_city['forward'] = DummyEdgeSensor(
        'raw', mode='forward', keyword='raw')

    neighbor['raw'] = CustomReader(keyword='raw')

    city[firestationCity] = DummyLearner('raw')
    city[firestationCity] = ReaderSensor(keyword='raw', label=True)


    program = config.program.Type(graph, **config.program)
    return program



@pytest.mark.gurobi
def test_graph_coloring_main():
    from config import CONFIG
    lbp = model_declaration(CONFIG.Model)

    # dataset = None # FIXME: shouldn't this example anyway based on a iterable object as data source?
    # for output in lbp.eval(dataset=dataset, inference=True):
    #     print(output)

    # using an underlying call
    loss, metric, datanode = lbp.model({}, inference=True)
    conceptsRelations = [] # TODO: please fill this
    tokenResult, pairResult, tripleResult = datanode.inferILPConstrains(*conceptsRelations, fun=None)
    print('I am here!')


if __name__ == '__main__':
    pytest.main([__file__])
