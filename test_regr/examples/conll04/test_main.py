import sys
sys.path.append(".")
sys.path.append("../..")

import pytest


@pytest.fixture(name='case')
def test_case():
    import torch
    from emr.utils import Namespace

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    case = {
        'word': {
            'raw': ["John", "works", "for", "IBM"],
            'emb': torch.randn(4, 2048, device=device),
            #                             John        works       for           IBM
            'people':       torch.tensor([[0.3, 0.7], [0.9, 0.1], [0.98, 0.02], [0.40, 0.6]], device=device),
            'organization': torch.tensor([[0.5, 0.5], [0.8, 0.2], [0.97, 0.03], [0.09, 0.91]], device=device),
            'location':     torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.95, 0.05], [0.50, 0.50]], device=device),
            'other':        torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.90, 0.10], [0.70, 0.30]], device=device),
            'o':            torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.10, 0.90], [0.90, 0.10]], device=device)
        }
    }
    case = Namespace(case)
    return case


def model_declaration(config, case):
    from emr.graph.torch import LearningBasedProgram
    from regr.sensor.pytorch.sensors import ReaderSensor

    from graph import graph, sentence, word, char, people, organization, location, other, o
    from graph import rel_sentence_contains_word, rel_phrase_contains_word, rel_word_contains_char
    from emr.sensors.Sensors import TestSensor, DummyWordEmb, DummyCharEmb, DummyFullyConnectedLearner
    from emr.sensors.Sensors import DummyEdgeStoW, DummyEdgeWtoC

    graph.detach()

    sentence['raw'] = ReaderSensor(keyword='token')
    sentence['raw'] = ReaderSensor(keyword='token', label=True)  # just to trigger calculation

    # Edge: sentence to word forward
    rel_sentence_contains_word['forward'] = DummyEdgeStoW("raw", mode="forward", keyword="raw")
    # alternatives to DummyEdgeStoW:
    #   DummyEdgeStoW: ["John", "works", "for", "IBM"]

    word['emb'] = DummyWordEmb('raw', edges=[rel_sentence_contains_word['forward']],
                               expected_inputs=[case.word.raw,],
                               expected_outputs=case.word.emb
                              )
    word['raw2'] = TestSensor('raw', edges=[rel_sentence_contains_word['forward']],
                               expected_outputs=case.word.raw
                               )

    # Edge: word to char forward
    rel_word_contains_char['forward'] = DummyEdgeWtoC("raw2", mode="forward", keyword="raw2")
    # alternatives to DummyEdgeWtoC:
    #   DummyEdgeWtoC: [["J", "o", "h", "n"], ["w", "o", "r", "k", "s"], ["f", "o", "r"], ["I", "B", "M"]]
    #   DummyEdgeWtoCOpt2: ["J", "o", "h", "n"]
    #   DummyEdgeWtoCOpt3: ["J", "o", "h", "n", " ", "w", "o", "r", "k", "s", " ", "f", "o", "r", " ", "I", "B", "M"]
    char['emb'] = DummyCharEmb('raw2', edges=[rel_word_contains_char['forward']])
    char['emb'] = ReaderSensor(keyword='token', label=True)  # just to trigger calculation

    # Edge: backward
    # TODO: need help implementing BILTransformer, MaxAggregationSensor 
    #rel_phrase_contains_word['backward'] = BILTransformer('raw_ready', 'boundary', mode="backward")
    #phrase['emb'] = MaxAggregationSensor("raw_ready", edge=rel_phrase_contains_word['backward'], map_key="encode")
    #phrase['emb'] = ReaderSensor(keyword='token', label=True)  # just to trigger calculation

    word[people] = ReaderSensor(keyword='Peop', label=True)
    word[organization] = ReaderSensor(keyword='Org', label=True)
    word[location] = ReaderSensor(keyword='Loc', label=True)
    word[other] = ReaderSensor(keyword='Other', label=True)
    word[o] = ReaderSensor(keyword='O', label=True)

    word[people] = DummyFullyConnectedLearner('emb', input_dim=2048, output_dim=2,
                                         expected_inputs=[case.word.emb,],
                                         expected_outputs=case.word.people)
    word[organization] = DummyFullyConnectedLearner('emb', input_dim=2048, output_dim=2,
                                         expected_inputs=[case.word.emb,],
                                         expected_outputs=case.word.organization)
    word[location] = DummyFullyConnectedLearner('emb', input_dim=2048, output_dim=2,
                                         expected_inputs=[case.word.emb,],
                                         expected_outputs=case.word.location)
    word[other] = DummyFullyConnectedLearner('emb', input_dim=2048, output_dim=2,
                                         expected_inputs=[case.word.emb,],
                                         expected_outputs=case.word.other)
    word[o] = DummyFullyConnectedLearner('emb', input_dim=2048, output_dim=2,
                                         expected_inputs=[case.word.emb,],
                                         expected_outputs=case.word.o)

    lbp = LearningBasedProgram(graph, **config)
    return lbp


def test_main_conll04(case):
    from config import Config as config
    from emr.data import ConllDataLoader

    training_set = ConllDataLoader(config.Data.train_path,
                                   batch_size=config.Train.batch_size,
                                   skip_none=config.Data.skip_none)
    lbp = model_declaration(config.Model, case)
    data = next(iter(training_set))
    _, _, datanode = lbp.model(data)
    sentencenode = datanode
    assert sentencenode.getChildDataNodes() != None
    for word_idx, word_node in enumerate(sentencenode.getChildDataNodes()):
        assert word_node.ontologyNode.name == 'word'

        assert word_node.getAttribute('raw') == case.word.raw[word_idx]
        assert (word_node.getAttribute('emb') == case.word.emb[word_idx]).all()
        assert (word_node.getAttribute('<people>') == case.word.people[word_idx]).all()
        assert (word_node.getAttribute('<organization>') == case.word.organization[word_idx]).all()
        assert (word_node.getAttribute('<location>') == case.word.location[word_idx]).all()
        assert (word_node.getAttribute('<other>') == case.word.other[word_idx]).all()
        assert (word_node.getAttribute('<O>') == case.word.o[word_idx]).all()

if __name__ == '__main__':
    pytest.main([__file__])
