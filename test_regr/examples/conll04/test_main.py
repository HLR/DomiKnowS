import sys
sys.path.append(".")
sys.path.append("../..")

import pytest
import torch

from emr.utils import Namespace


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_case = {
    'word': {
        'raw': ["John", "works", "for", "IBM"],
        'emb': torch.randn(4, 2048, device=device),
        'people': torch.tensor([[0.1, 0.9], [0.6, 0.4], [0.9, 0.1], [0.4, 0.6]], device=device)
    }
}
test_case = Namespace(test_case)


def model_declaration(config):
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
                               expected_inputs=[test_case.word.raw,],
                               expected_outputs=test_case.word.emb
                              )

    # sentence['emb'] = SentenceWord2VecSensor('raw')
    # rel_sentence_contains_word['forward'] = SentenceToWordEmb('emb', mode="forward", keyword="emb")
    # word['emb'] = WordEmbedding('emb', edges=[rel_sentence_contains_word['forward']])
    # word['encode'] = LSTMLearner('features', input_dim=5236, hidden_dim=240, num_layers=1, bidirectional=True)
    # word['feature'] = MLPLearner(word['ctx_emb'], in_features=200, out_features=200)

    word[people] = ReaderSensor(keyword='Peop', label=True)
    # word[organization] = LabelSensor('Org')
    # word[location] = LabelSensor('Loc')
    # word[other] = LabelSensor('Other')
    # word[o] = LabelSensor('O')

    word[people] = DummyFullyConnectedLearner('emb', input_dim=2048, output_dim=2,
                                         expected_inputs=[test_case.word.emb,],
                                         expected_outputs=test_case.word.people)
    # word[organization] = LRLearner(word['feature'], in_features=200)
    # word[location] = LRLearner(word['feature'], in_features=200)
    # word[other] = LRLearner(word['feature'], in_features=200)
    # word[o] = LRLearner(word['feature'], in_features=200)

    lbp = LearningBasedProgram(graph, **config)
    return lbp


#### The main entrance of the program.
@pytest.mark.gurobi
def test_main_conll04():
    from config import Config as config
    from emr.data import ConllDataLoader

    training_set = ConllDataLoader(config.Data.train_path,
                                   batch_size=config.Train.batch_size,
                                   skip_none=config.Data.skip_none)
    lbp = model_declaration(config.Model)
    data = next(iter(training_set))
    _, _, datanode = lbp.model(data)
    print(datanode)
    for concept, concept_node in datanode.getChildInstanceNodes().items():
        print(concept.name)
        # FIXME: some problem here! The first word get 4x2 tensor, while other three get none!
        for word_node in concept_node:
            print(word_node.getAttributes())
            print(word_node.getAttribute('<people>'))
    #lbp.train(training_set)

if __name__ == '__main__':
    test_main_conll04()
