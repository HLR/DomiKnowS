import sys
sys.path.append(".")
sys.path.append("../..")

import pytest


def model_declaration(config):
    from emr.graph.torch import LearningBasedProgram
    #from emr.sensor.sensor import DataSensor, LabelSensor, CartesianSensor
    #from emr.sensor.learner import EmbedderLearner, RNNLearner, MLPLearner, LRLearner
    from regr.sensor.pytorch.sensors import ReaderSensor

    from graph import graph, sentence, word, people, organization, location, other, o
    from graph import rel_sentence_contains_word

    graph.detach()

    sentence['raw'] = ReaderSensor(keyword='raw')
    sentence['raw'] = ReaderSensor(keyword='raw', label=True)  # a checkpoint exam

    # sentence['emb'] = SentenceWord2VecSensor('raw')
    # rel_sentence_contains_word['forward'] = SentenceToWordEmb('emb', mode="forward", keyword="emb")
    # word['emb'] = WordEmbedding('emb', edges=[rel_sentence_contains_word['forward']])
    # word['encode'] = LSTMLearner('features', input_dim=5236, hidden_dim=240, num_layers=1, bidirectional=True)
    # word['feature'] = MLPLearner(word['ctx_emb'], in_features=200, out_features=200)

    # word[people] = LabelSensor('Peop')
    # word[organization] = LabelSensor('Org')
    # word[location] = LabelSensor('Loc')
    # word[other] = LabelSensor('Other')
    # word[o] = LabelSensor('O')

    # word[people] = LRLearner(word['feature'], in_features=200)
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
    from emr.utils import seed
    from emr.data import ConllDataLoader

    training_set = ConllDataLoader(config.Data.train_path,
                                   batch_size=config.Train.batch_size,
                                   skip_none=config.Data.skip_none)
    lbp = model_declaration(config.Model)
    lbp.train(training_set, None)

if __name__ == '__main__':
    test_main_conll04()
