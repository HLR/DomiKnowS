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

    from graph import graph, sentence, word, people, organization, location, other, o
    from graph import rel_sentence_contains_word
    from emr.sensors.Sensors import DummyEdgeStoW, DummyWordEmb, DummyFullyConnectedLearner

    graph.detach()

    sentence['raw'] = ReaderSensor(keyword='token')
    sentence['raw'] = ReaderSensor(keyword='token', label=True)  # sentence checkpoint

    rel_sentence_contains_word['forward'] = DummyEdgeStoW("raw", mode="forward", keyword="raw")
    word['emb'] = DummyWordEmb('raw', edges=[rel_sentence_contains_word['forward']],
                               expected_inputs=[case.word.raw,],
                               expected_outputs=case.word.emb
                              )

    # sentence['emb'] = SentenceWord2VecSensor('raw')
    # rel_sentence_contains_word['forward'] = SentenceToWordEmb('emb', mode="forward", keyword="emb")
    # word['emb'] = WordEmbedding('emb', edges=[rel_sentence_contains_word['forward']])
    # word['encode'] = LSTMLearner('features', input_dim=5236, hidden_dim=240, num_layers=1, bidirectional=True)
    # word['feature'] = MLPLearner(word['ctx_emb'], in_features=200, out_features=200)

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


@pytest.mark.gurobi
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
    for concept, word_nodes in sentencenode.getChildInstanceNodes().items():
        assert concept.name == 'word'
        for word_idx, word_node in enumerate(word_nodes):
            assert word_node.getAttribute('raw') == case.word.raw[word_idx]
            assert (word_node.getAttribute('emb') == case.word.emb[word_idx]).all()
            assert (word_node.getAttribute('<people>') == case.word.people[word_idx]).all()
            assert (word_node.getAttribute('<organization>') == case.word.organization[word_idx]).all()
            assert (word_node.getAttribute('<location>') == case.word.location[word_idx]).all()
            assert (word_node.getAttribute('<other>') == case.word.other[word_idx]).all()
            assert (word_node.getAttribute('<O>') == case.word.o[word_idx]).all()

if __name__ == '__main__':
    pytest.main([__file__])
