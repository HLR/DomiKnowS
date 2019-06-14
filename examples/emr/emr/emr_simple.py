import os
from regr.graph.allennlp import AllenNlpGraph
from regr.sensor.allennlp.sensor import TokenSequenceSensor, LabelSequenceSensor
from regr.sensor.allennlp.learner import W2VLearner, RNNLearner, LRLearner, CPCatLearner
from allennlp.data import Vocabulary
from .data import Conll04SensorReader as Reader


# data setting
relative_path = "data/EntityMentionRelation"
train_path = "conll04_train.corp"
valid_path = "conll04_test.corp"

# model setting
EMBEDDING_DIM = 8

# training setting
LR = 0.001
WD = 0.0001
BATCH = 8
EPOCH = 50
PATIENCE = None


def ontology_declaration():
    from .graph_simple import graph
    return graph


def model_declaration(graph, reader, vocab):
    graph.detach()

    # retrieve concepts you need in this model
    phrase = graph['linguistic/phrase']
    pair = graph['linguistic/pair']

    people = graph['application/people']
    organization = graph['application/organization']
    work_for = graph['application/work_for']

    # connect sensors and learners
    # features
    phrase['raw'] = TokenSequenceSensor(reader, 'sentence')
    phrase['w2v'] = W2VLearner(vocab.get_vocab_size('tokens'), EMBEDDING_DIM, 'tokens', phrase['raw'])  # 'tokens' is from reader
    phrase['emb'] = RNNLearner(EMBEDDING_DIM, phrase['w2v'])
    pair['emb'] = CPCatLearner(phrase['emb'])

    # concept label
    people['label'] = LabelSequenceSensor(reader, 'Peop', output_only=True)
    organization['label'] = LabelSequenceSensor(reader, 'Org', output_only=True)

    # concept prediction
    people['label'] = LRLearner(EMBEDDING_DIM * 2, phrase['emb'])
    organization['label'] = LRLearner(EMBEDDING_DIM * 2, phrase['emb'])

    # composed-concept label
    work_for['label'] = LabelSequenceSensor(reader, 'Work_For', output_only=True)

    # composed-concept prediction
    work_for['label'] = LRLearner(EMBEDDING_DIM * 4, pair['emb'])

    return graph


# envionment setup

#import logging
# logging.basicConfig(level=logging.INFO)

def seed1():
    import random
    import numpy as np
    import torch

    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)


seed1()


def main():
    # 0. Prepare Data
    reader = Reader()
    train_dataset = reader.read(os.path.join(relative_path, train_path))
    valid_dataset = reader.read(os.path.join(relative_path, valid_path))
    vocab = Vocabulary.from_instances(train_dataset + valid_dataset)

    # 1. Ontology Declaration
    graph = ontology_declaration()

    # 2. Model Declaration
    graph = model_declaration(graph, reader, vocab)
    lbp = AllenNlpGraph(graph, vocab)

    # 2.5/3. Train and save the model (Explicit inference done automatically)
    lbp.train(train_dataset, valid_dataset)
    lbp.save('/tmp/emr_simple')
