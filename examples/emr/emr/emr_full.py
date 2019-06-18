import os
from regr.graph.allennlp import AllenNlpGraph
from regr.sensor.allennlp.sensor import PhraseSequenceSensor, LabelSequenceSensor, CartesianProductSensor
from regr.sensor.allennlp.learner import W2VLearner, RNNLearner, LogisticRegressionLearner
from allennlp.data import Vocabulary

from .graph import graph
from .data import Conll04SensorReader as Reader
from .config import Config
from .utils import seed

def data_preparation(config):
    reader = Reader()
    train_dataset = reader.read(os.path.join(config.relative_path, config.train_path))
    valid_dataset = reader.read(os.path.join(config.relative_path, config.valid_path))
    vocab = Vocabulary.from_instances(train_dataset + valid_dataset)
    return reader, vocab, train_dataset, valid_dataset

def ontology_declaration():
    return graph


def model_declaration(graph, vocab, config):
    # reset the graph
    graph.detach()

    # retrieve concepts you need in this model
    phrase = graph['linguistic/phrase']
    pair = graph['linguistic/pair']

    people = graph['application/people']
    organization = graph['application/organization']
    location = graph['application/location']
    other = graph['application/other']
    o = graph['application/O']

    work_for = graph['application/work_for']
    located_in = graph['application/located_in']
    live_in = graph['application/live_in']
    orgbase_on = graph['application/orgbase_on']

    # connect sensors and learners
    # features
    phrase['raw'] = PhraseSequenceSensor(vocab, 'sentence', 'phrase')
    phrase['w2v'] = W2VLearner(config.embedding_dim, phrase['raw'])
    phrase['emb'] = RNNLearner(config.embedding_dim, phrase['w2v'])
    pair['emb'] = CartesianProductSensor(phrase['emb'])

    # concept label
    people['label'] = LabelSequenceSensor('Peop', output_only=True)
    organization['label'] = LabelSequenceSensor('Org', output_only=True)
    location['label'] = LabelSequenceSensor('Loc', output_only=True)
    other['label'] = LabelSequenceSensor('Other', output_only=True)
    o['label'] = LabelSequenceSensor('O', output_only=True)

    # concept prediction
    people['label'] = LogisticRegressionLearner(config.embedding_dim * 2, phrase['emb'])
    organization['label'] = LogisticRegressionLearner(config.embedding_dim * 2, phrase['emb'])
    location['label'] = LogisticRegressionLearner(config.embedding_dim * 2, phrase['emb'])
    other['label'] = LogisticRegressionLearner(config.embedding_dim * 2, phrase['emb'])
    o['label'] = LogisticRegressionLearner(config.embedding_dim * 2, phrase['emb'])

    # composed-concept label
    work_for['label'] = LabelSequenceSensor('Work_For', output_only=True)
    live_in['label'] = LabelSequenceSensor('Live_In', output_only=True)
    located_in['label'] = LabelSequenceSensor('Located_In', output_only=True)
    orgbase_on['label'] = LabelSequenceSensor('OrgBased_In', output_only=True)

    # composed-concept prediction
    work_for['label'] = LogisticRegressionLearner(config.embedding_dim * 4, pair['emb'])
    live_in['label'] = LogisticRegressionLearner(config.embedding_dim * 4, pair['emb'])
    located_in['label'] = LogisticRegressionLearner(config.embedding_dim * 4, pair['emb'])
    orgbase_on['label'] = LogisticRegressionLearner(config.embedding_dim * 4, pair['emb'])

    # wrap with allennlp model
    lbp = AllenNlpGraph(graph, vocab)
    return lbp


def main():
    # 0. Prepare Data
    reader, vocab, train_dataset, valid_dataset = data_preparation(Config.Data)

    # 1. Ontology Declaration
    graph = ontology_declaration()

    # 2. Model Declaration
    lbp = model_declaration(graph, vocab, Config.Model)

    # 2.5/3. Train and save the model (Explicit inference done automatically)
    seed() # initial the random seeds of all subsystems
    lbp.train(train_dataset, valid_dataset, Config.Train)
    lbp.save('/tmp/emr')
