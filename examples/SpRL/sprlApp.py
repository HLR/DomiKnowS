import os
import torch
from allennlp.models.model import Model
from regr.graph.allennlp import AllenNlpGraph
from regr.sensor.allennlp.sensor import PhraseSequenceSensor, LabelSequenceSensor, CartesianProductSensor
from regr.sensor.allennlp.learner import W2VLearner, RNNLearner, LogisticRegressionLearner
from allennlp.data import Vocabulary



if __package__ is None or __package__ == '':
    # uses current directory visibility
    from data import Data, Conll04BinaryReader as Reader
    from graph import graph
else:
    # uses current package visibility
    from .data import Data, Conll04BinaryReader as Reader
    from .models import get_trainer, datainput, word2vec, word2vec_rnn, fullyconnected, cartesianprod_concat, logsm
    from .spGraph import graph


# data setting
relative_path = "examples/SpRL/data/newSprl2017_all.xml"
train_path = "newSprl2017_all.xml"
valid_path = "newSprl2017_all.xml"

# model setting
EMBEDDING_DIM = 8

# training setting
LR = 0.001
WD = 0.0001
BATCH = 8
EPOCH = 50
PATIENCE = None


# develop by an ML programmer to wire nodes in the graph and ML Models
def model_declaration(graph, vocab, config):    # initialize the graph
    graph.detach()  # release anything binded before new assignment

    # get concepts from graph
    phrase = graph['linguistic/phrase']
    pair = graph['linguistic/pair']
    # concepts
    tr = graph['application/tr']
    lm = graph['application/lm']
    o = graph['application/O']

    # composed concepts
    sp_tr = graph['application/sp_tr']

    phrase['raw'] = PhraseSequenceSensor(vocab, 'sentence', 'phrase')
    phrase['w2v'] = W2VLearner(config.embedding_dim, phrase['raw'])
    phrase['emb'] = RNNLearner(config.embedding_dim, phrase['w2v'])
    pair['emb'] = CartesianProductSensor(phrase['emb'])



    # concept labels

    # concept label
    tr['label'] = LabelSequenceSensor('tr', output_only=True)
    lm['label'] = LabelSequenceSensor('lm', output_only=True)
    o['label'] = LabelSequenceSensor('O', output_only=True)

    tr['label'] = LogisticRegressionLearner(config.embedding_dim * 2, phrase['emb'])
    lm['label'] = LogisticRegressionLearner(config.embedding_dim * 2, phrase['emb'])
    o['label'] = LogisticRegressionLearner(config.embedding_dim * 2, phrase['emb'])

    sp_tr['label'] = LabelSequenceSensor('sp_tr', output_only=True)

    # composed-concept prediction
    sp_tr['label'] = LogisticRegressionLearner(config.embedding_dim * 4, pair['emb'])

    # building model
    # embedding


    lbp= AllenNlpGraph(graph, vocab)

    return lbp


# envionment setup
def ontology_declaration():
    from .spGraph import splang_Graph
    return splang_Graph
#import logging
# logging.basicConfig(level=logging.INFO)


def seed():
    import random
    import numpy as np
    import torch

    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)


seed()


def main():
    Config = {
        'Data': {  # data setting
            'relative_path': "data/EntityMentionRelation",
            'train_path': "conll04_train.corp",
            'valid_path': "conll04_test.corp"
        },
        'Model': {  # model setting
            'embedding_dim': 8
        },
        'Train': {
            'lr': 0.001,
            'wd': 0.0001,
            'batch': 8,
            'epoch': 50,
            'patience': None
        }
    }
    reader = Reader()
    train_dataset = reader.read(os.path.join(relative_path, train_path))
    valid_dataset = reader.read(
        os.path.join(relative_path, valid_path))

    vocab = Vocabulary.from_instances(train_dataset + valid_dataset)

    # 1. Ontology Declaration
    graph = ontology_declaration()

    # 2. Model Declaration
    lbp = model_declaration(graph, vocab, Config.Model)

    # 2.5/3. Train and save the model (Explicit inference done automatically)
    seed()  # initial the random seeds of all subsystems
    lbp.train(train_dataset, valid_dataset, Config.Train)
    lbp.save('/tmp/srl')


if __name__ == '__main__':
    main()
