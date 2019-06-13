import os
from regr.graph import Graph, Concept, Relation
from regr.graph.allennlp import AllenNlpGraph
from regr.sensor.allennlp.sensor import TokenSequenceSensor, LabelSequenceSensor
from regr.sensor.allennlp.learner import W2VLearner, RNNLearner, LRLearner
from emr.data import Conll04SensorReader as Reader
from allennlp.data import Vocabulary


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

Graph.clear()
Concept.clear()
Relation.clear()


def ontology_declaration():
    with Graph('global') as graph:
        graph.ontology = 'http://ontology.ihmc.us/ML/EMR.owl'
        with Graph('linguistic') as ling_graph:
            phrase = Concept(name='phrase')
            pair = Concept(name='pair')
            pair.has_a(phrase, phrase)
        with Graph('application') as app_graph:
            people = Concept(name='people')
            organization = Concept(name='organization')
            people.is_a(phrase)
            organization.is_a(phrase)
            people.not_a(organization)
            organization.not_a(people)
            work_for = Concept(name='work_for')
            work_for.is_a(pair)
            work_for.has_a(employee=people, employer=organization)

    return graph


def model_declaration(graph, reader, vocab):
    graph.detach()

    phrase = graph['linguistic/phrase']
    people = graph['application/people']
    organization = graph['application/organization']

    phrase['sentence'] = TokenSequenceSensor(reader, 'sentence')
    phrase['pos_tag'] = LabelSequenceSensor(reader, 'pos')
    phrase['w2v'] = W2VLearner(vocab.get_vocab_size(
        'tokens'), 32, 'tokens', phrase['sentence'])  # 'tokens' is from reader
    phrase['emb'] = RNNLearner(32, phrase['w2v'])

    people['label'] = LabelSequenceSensor(reader, 'Peop', output_only=True)
    organization['label'] = LabelSequenceSensor(
        reader, 'Org', output_only=True)

    people['label'] = LRLearner(64, phrase['emb'])
    organization['label'] = LRLearner(64, phrase['emb'])

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
    lbp.save('/tmp/emr')


if __name__ == '__main__':
    main()
