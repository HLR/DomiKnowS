import sys
sys.path.append('.')
sys.path.append('../..')

import pytest


@pytest.fixture(name='case')
def test_case():
    import torch
    from emr.utils import Namespace

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    case = {
        'char': {
            'raw': [['J', 'o', 'h', 'n'], ['w', 'o', 'r', 'k', 's'], ['f', 'o', 'r'], ['I', 'B', 'M']]
        },
        'word': {
            'raw': ['John', 'works', 'for', 'IBM'],
            'emb': torch.randn(4, 2048, device=device),
            #                             John        works       for           IBM
            'people':       torch.tensor([[0.3, 0.7], [0.9, 0.1], [0.98, 0.02], [0.40, 0.6]], device=device),
            'organization': torch.tensor([[0.5, 0.5], [0.8, 0.2], [0.97, 0.03], [0.09, 0.91]], device=device),
            'location':     torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.95, 0.05], [0.50, 0.50]], device=device),
            'other':        torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.90, 0.10], [0.70, 0.30]], device=device),
            'O':            torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.10, 0.90], [0.90, 0.10]], device=device),
        },
        'phrase': {
            'raw': [(0, 0), (1, 2), (3, 3)],  # ['John', 'works for', 'IBM'],
            'emb': torch.randn(3, 2048, device=device),
            'people': torch.tensor([[0.3, 0.7], [0.9, 0.1], [0.40, 0.6]], device=device),

        },
        'pair': {
            'emb': torch.randn(4, 4, 2048, device=device),
            'work_for': torch.rand(4, 4, 2, device=device), # TODO: add examable values
            
            'work_for': torch.tensor([[[0.60, 0.40],         [0.80, 0.20],         [0.80, 0.20], [0.37, 0.63]],      # John
                                      [[1.00, float("nan")], [1.00, float("nan")], [0.60, 0.40], [0.70, 0.30]],  # works
                                      [[0.98, 0.02],         [0.70, 0.03],         [0.95, 0.05], [0.90, 0.10]],  # for
                                      [[0.35, 0.65],         [0.80, 0.20],         [0.90, 0.10], [0.70, 0.30]],  # IBM
                                     ], device=device),
            
            'live_in': torch.rand(4, 4, 2, device=device), # TODO: add examable values
            'located_in': torch.rand(4, 4, 2, device=device), # TODO: add examable values
            'orgbase_on': torch.rand(4, 4, 2, device=device), # TODO: add examable values
            'kill': torch.rand(4, 4, 2, device=device), # TODO: add examable values
        }
    }
    case = Namespace(case)
    return case


def model_declaration(config, case):
    from emr.graph.torch import LearningBasedProgram
    from regr.sensor.pytorch.sensors import ReaderSensor

    from graph import graph, sentence, word, char, phrase, pair
    from graph import people, organization, location, other, o
    from graph import work_for, located_in, live_in, orgbase_on, kill
    from graph import rel_sentence_contains_word, rel_phrase_contains_word, rel_word_contains_char, rel_pair_word1, rel_pair_word2
    from emr.sensors.Sensors import TestSensor, TestEdgeSensor

    graph.detach()

    sentence['raw'] = ReaderSensor(keyword='token')
    sentence['raw'] = ReaderSensor(keyword='token', label=True)  # just to trigger calculation

    # Edge: sentence to word forward
    rel_sentence_contains_word['forward'] = TestEdgeSensor(
        'raw', mode='forward', keyword='raw',
        expected_outputs=case.word.raw)
    word['emb'] = TestSensor(
        'raw', edges=[rel_sentence_contains_word['forward']],
        expected_inputs=[case.word.raw,],
        expected_outputs=case.word.emb)

    # Edge: word to char forward
    rel_word_contains_char['forward'] = TestEdgeSensor(
        'raw', mode='forward', keyword='raw',
        edges=[rel_sentence_contains_word['forward']],
        expected_inputs=[case.word.raw,],
        expected_outputs=case.char.raw)
    char['emb'] = TestSensor(
        'raw', edges=[rel_word_contains_char['forward']],
        expected_inputs=[case.char.raw,],
        expected_outputs=case.word.emb)
    char['emb'] = ReaderSensor(keyword='token', label=True)  # just to trigger calculation

    # Edge: word to phrase backward
    rel_phrase_contains_word['backward'] = TestEdgeSensor(
        'raw', mode='backward', keyword='raw',
        edges=[rel_sentence_contains_word['forward']],
        expected_inputs=[case.word.raw,],
        expected_outputs=case.phrase.raw)
    phrase['emb'] = TestSensor(
        'raw', edges=[rel_phrase_contains_word['backward']],
        expected_inputs=[case.phrase.raw,],
        expected_outputs=case.phrase.emb)
    phrase['emb'] = ReaderSensor(keyword='token', label=True)  # just to trigger calculation

    # Edge: pair backward
    rel_pair_word1['backward'] = TestEdgeSensor(
        'emb', mode='backward', keyword='word1_emb',
        expected_inputs=[case.word.emb,],
        expected_outputs=case.word.emb)
    rel_pair_word2['backward'] = TestEdgeSensor(
        'emb', mode='backward', keyword='word2_emb',
        expected_inputs=[case.word.emb,],
        expected_outputs=case.word.emb)

    pair['emb'] = TestSensor(
        'word1_emb', 'word2_emb',
        edges=[rel_pair_word1['backward'], rel_pair_word2['backward']],
        expected_inputs=[case.word.emb, case.word.emb],
        expected_outputs=case.pair.emb)

    word[people] = ReaderSensor(keyword='Peop', label=True)
    word[organization] = ReaderSensor(keyword='Org', label=True)
    word[location] = ReaderSensor(keyword='Loc', label=True)
    word[other] = ReaderSensor(keyword='Other', label=True)
    word[o] = ReaderSensor(keyword='O', label=True)

    word[people] = TestSensor(
        'emb', input_dim=2048, output_dim=2,
        expected_inputs=[case.word.emb,],
        expected_outputs=case.word.people)
    word[organization] = TestSensor(
        'emb', input_dim=2048, output_dim=2,
        expected_inputs=[case.word.emb,],
        expected_outputs=case.word.organization)
    word[location] = TestSensor(
        'emb', input_dim=2048, output_dim=2,
        expected_inputs=[case.word.emb,],
        expected_outputs=case.word.location)
    word[other] = TestSensor(
        'emb', input_dim=2048, output_dim=2,
        expected_inputs=[case.word.emb,],
        expected_outputs=case.word.other)
    word[o] = TestSensor(
        'emb', input_dim=2048, output_dim=2,
        expected_inputs=[case.word.emb,],
        expected_outputs=case.word.O)

    phrase[people] = ReaderSensor(keyword='Peop', label=True)
    phrase[people] = TestSensor(
        'emb', input_dim=2048, output_dim=2,
        expected_inputs=[case.phrase.emb,],
        expected_outputs=case.phrase.people)

    pair[work_for] = ReaderSensor(keyword='Work_For', label=True)
    pair[located_in] = ReaderSensor(keyword='Located_In', label=True)
    pair[live_in] = ReaderSensor(keyword='Live_In', label=True)
    pair[orgbase_on] = ReaderSensor(keyword='OrgBased_In', label=True)
    pair[kill] = ReaderSensor(keyword='Kill', label=True)

    pair[work_for] = TestSensor(
        'emb', input_dim=2048, output_dim=2,
        expected_inputs=[case.pair.emb,],
        expected_outputs=case.pair.work_for)
    pair[located_in] = TestSensor(
        'emb', input_dim=2048, output_dim=2,
        expected_inputs=[case.pair.emb,],
        expected_outputs=case.pair.located_in)
    pair[live_in] = TestSensor(
        'emb', input_dim=2048, output_dim=2,
        expected_inputs=[case.pair.emb,],
        expected_outputs=case.pair.live_in)
    pair[orgbase_on] = TestSensor(
        'emb', input_dim=2048, output_dim=2,
        expected_inputs=[case.pair.emb,],
        expected_outputs=case.pair.orgbase_on)
    pair[orgbase_on] = TestSensor(
        'emb', input_dim=2048, output_dim=2,
        expected_inputs=[case.pair.emb,],
        expected_outputs=case.pair.kill)

    lbp = LearningBasedProgram(graph, **config)
    return lbp

def test_graph_naming():
    from graph import graph, sentence, word, char, phrase, pair
    from graph import people, organization, location, other, o
    from graph import work_for, located_in, live_in, orgbase_on, kill
    from graph import rel_sentence_contains_word, rel_phrase_contains_word, rel_word_contains_char, rel_pair_word1, rel_pair_word2

    # graph
    assert graph.name == 'global'

    # concepts
    assert sentence.name == 'sentence'
    assert word.name == 'word'
    assert char.name == 'char'
    assert phrase.name == 'phrase'
    assert pair.name == 'pair'

    assert people.name == 'people'
    assert organization.name == 'organization'
    assert location.name == 'location'
    assert other.name == 'other'
    assert o.name == 'O'  # Note: here is different Cap: `o = word(name='O')`

    assert work_for.name == 'work_for'
    assert located_in.name == 'located_in'
    assert live_in.name == 'live_in'
    assert orgbase_on.name == 'orgbase_on'
    assert kill.name == 'kill'

    # relation: default named
    assert rel_sentence_contains_word.name == 'sentence-contains-0-word'
    assert rel_phrase_contains_word.name == 'phrase-contains-0-word'
    assert rel_word_contains_char.name == 'word-contains-0-char'

    # relation: explicitly named
    # `(rel_pair_word1, rel_pair_word2, ) = pair.has_a(arg1=word, arg2=word)`
    assert rel_pair_word1.name == 'arg1'
    assert rel_pair_word2.name == 'arg2'


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
                    
            assert len(child_node.getRelationLinks(relationName = "pair")) == 4
            
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
            assert False, 'There should be only word and phrases. {} is unexpected.'.format(child_node.ontologyNode.name)

    conceptsRelations = ['people', 'organization', 'location', 'other', 'O', 'work_for']
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
    
    assert pairResult['work_for'][0][3] == 1
    assert sum(pairResult['work_for'][0]) == 1
    assert sum(pairResult['work_for'][1]) == 0
    assert sum(pairResult['work_for'][2]) == 0
    assert sum(pairResult['work_for'][3]) == 0

if __name__ == '__main__':
    pytest.main([__file__])
