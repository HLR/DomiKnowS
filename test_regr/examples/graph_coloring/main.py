import sys

sys.path.append('.')
sys.path.append('../..')

import pytest



def model_declaration(config, case):
    from regr.sensor.pytorch.sensors import ReaderSensor

    from graph import graph, world, city, neighbor, world_contains_city, neighbor_city1, neighbor_city2, firestationCity

    from sensors import DummyLearner, DummyEdgeSensor, CustomReader

    graph.detach()

    world['raw'] = ReaderSensor(keyword='world')

    # Edge: sentence to word forward
    world_contains_city['forward'] = DummyEdgeSensor(
        'raw', mode='forward', keyword='raw')

    neighbor['raw'] = CustomReader(keyword='raw')
    neighbor['raw'] = CustomReader(keyword='raw')

    city[firestationCity] = DummyLearner('raw')
    city[firestationCity] = DummyLearner('raw')

    lbp = LearningBasedProgram(graph, **config)
    return lbp



@pytest.mark.gurobi
def test_main_conll04(case):
    from config import CONFIG
    from emr.data import ConllDataLoader

    training_set = ConllDataLoader(CONFIG.Data.train_path,
                                   batch_size=CONFIG.Train.batch_size,
                                   skip_none=CONFIG.Data.skip_none)
    lbp = model_declaration(CONFIG.Model, case)
    data = next(iter(training_set))

    _, _, datanode = lbp.model(data)

    for child_node in datanode.getChildDataNodes():
        if child_node.ontologyNode.name == 'word':
            assert child_node.getAttribute('raw') == case.word.raw[child_node.instanceID]

            for child_node1 in child_node.getChildDataNodes():
                if child_node1.ontologyNode.name == 'char':
                    assert True
                else:
                    assert False

            assert len(child_node.getChildDataNodes()) == len(case.char.raw[child_node.instanceID])

            assert len(child_node.getRelationLinks(relationName="pair")) == 4

            assert (child_node.getAttribute('emb') == case.word.emb[child_node.instanceID]).all()
            assert (child_node.getAttribute('<people>') == case.word.people[child_node.instanceID]).all()
            assert (child_node.getAttribute('<organization>') == case.word.organization[child_node.instanceID]).all()
            assert (child_node.getAttribute('<location>') == case.word.location[child_node.instanceID]).all()
            assert (child_node.getAttribute('<other>') == case.word.other[child_node.instanceID]).all()
            assert (child_node.getAttribute('<O>') == case.word.O[child_node.instanceID]).all()
        elif child_node.ontologyNode.name == 'phrase':
            assert (child_node.getAttribute('emb') == case.phrase.emb[child_node.instanceID]).all()
            assert (child_node.getAttribute('<people>') == case.phrase.people[child_node.instanceID]).all()
        else:
            assert False, 'There should be only word and phrases. {} is unexpected.'.format(
                child_node.ontologyNode.name)

    conceptsRelations = ['people', 'organization', 'location', 'other', 'O']
    tokenResult, pairResult, tripleResult = datanode.inferILPConstrains(*conceptsRelations, fun=None)

    assert tokenResult['people'][0] == 1
    assert sum(tokenResult['people']) == 1
    assert tokenResult['organization'][3] == 1
    assert sum(tokenResult['organization']) == 1
    assert sum(tokenResult['location']) == 0
    assert sum(tokenResult['other']) == 0
    assert tokenResult['O'][1] == 1
    assert tokenResult['O'][2] == 1
    assert sum(tokenResult['O']) == 2


if __name__ == '__main__':
    pytest.main([__file__])
