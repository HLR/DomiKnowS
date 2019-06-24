import os
from regr.graph.allennlp import AllenNlpGraph
from regr.sensor.allennlp.sensor import SequenceSensor, TokenInSequenceSensor, LabelSensor, CartesianProductSensor
from regr.sensor.allennlp.learner import W2VLearner, RNNLearner, LogisticRegressionLearner
from allennlp.data import Vocabulary

from .data import Conll04SensorReader as Reader
from .config import Config
from .utils import seed


def ontology_declaration():
    from .graph_simple import graph
    return graph


def model_declaration(graph, reader, config):
    # reset the graph
    graph.detach()

    # retrieve concepts you need in this model
    sentence = graph['linguistic/sentence']
    phrase = graph['linguistic/phrase']
    pair = graph['linguistic/pair']

    people = graph['application/people']
    organization = graph['application/organization']

    work_for = graph['application/work_for']

    # connect sensors and learners
    # features
    sentence['raw'] = SequenceSensor(reader, 'sentence')
    phrase['raw'] = TokenInSequenceSensor(sentence['raw'])
    phrase['w2v'] = W2VLearner(config.embedding_dim, phrase['raw'])
    phrase['emb'] = RNNLearner(config.embedding_dim, phrase['w2v'])
    pair['emb'] = CartesianProductSensor(phrase['emb'])

    # concept label
    people['label'] = LabelSensor(reader, 'Peop', output_only=True)
    organization['label'] = LabelSensor(reader, 'Org', output_only=True)

    # concept prediction
    people['label'] = LogisticRegressionLearner(config.embedding_dim * 2, phrase['emb'])
    organization['label'] = LogisticRegressionLearner(config.embedding_dim * 2, phrase['emb'])

    # composed-concept label
    work_for['label'] = LabelSensor(reader, 'Work_For', output_only=True)

    # composed-concept prediction
    work_for['label'] = LogisticRegressionLearner(config.embedding_dim * 4, pair['emb'])

    # wrap with allennlp model
    lbp = AllenNlpGraph(graph)
    return lbp


def main():
    # 0. Prepare Data
    reader = Reader()

    # 1. Ontology Declaration
    graph = ontology_declaration()

    # 2. Model Declaration
    lbp = model_declaration(graph, reader, Config.Model)

    # 2.5/3. Train and save the model (Explicit inference done automatically)
    seed() # initial the random seeds of all subsystems
    lbp.train(Config.Data, Config.Train)
    lbp.save('/tmp/emr')
