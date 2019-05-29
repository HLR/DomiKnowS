import os
import torch
from allennlp.models.model import Model
from regr import Graph
from regr.scaffold import Scaffold, AllennlpScaffold

if __package__ is None or __package__ == '':
    # uses current directory visibility
    from data import Data, Conll04BinaryReader as Reader
    from models import get_trainer, datainput, word2vec, word2vec_rnn, fullyconnected, cartesianprod_concat, logsm
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
def make_model(graph: Graph,
               data: Data,
               scaffold: Scaffold
               ) -> Model:
    # initialize the graph
    graph.release()  # release anything binded before new assignment

    # get concepts from graph
    phrase = graph.linguistic.phrase
    # concepts
    tr = graph.application.tr
    lm = graph.application.lm
    O = graph.application.O
    # composed
    pair = graph.linguistic.pair
    # composed concepts
    sp_tr = graph.application.sp_tr

    # data
    scaffold.assign(phrase, 'index', datainput(data['sentence']))
    # concept labels
    scaffold.assign(tr, 'label', datainput(data['Peop']))
    scaffold.assign(lm, 'label', datainput(data['Org']))
    scaffold.assign(O, 'label', datainput(data['O']))
    # composed concept labels
    scaffold.assign(sp_tr, 'label', datainput(data['Work_For']))

    # building model
    # embedding
    scaffold.assign(phrase, 'emb',
                    word2vec_rnn(
                        phrase['index'],
                        data.vocab.get_vocab_size('tokens'),
                        EMBEDDING_DIM,
                        'tokens' # token name related to data reader
                    ))
    # predictor
    scaffold.assign(tr, 'label',
                    logsm(
                        phrase['emb'],
                        EMBEDDING_DIM * 2,
                        2
                    ))
    scaffold.assign(lm, 'label',
                    logsm(
                        phrase['emb'],
                        EMBEDDING_DIM * 2,
                        2
                    ))


    scaffold.assign(O, 'label',
                    logsm(
                        phrase['emb'],
                        EMBEDDING_DIM * 2,
                        2
                    ))
    # TODO: pair['emb'] should be infer from phrase['emb'] according to their relationship
    # but we specify it here to make it a bit easier for implementation
    # composed embedding
    scaffold.assign(pair, 'emb',
                    cartesianprod_concat(
                        phrase['emb']
                    ))
    # composed predictor
    scaffold.assign(sp_tr, 'label',
                    logsm(
                        pair['emb'],
                        EMBEDDING_DIM * 4,
                        2
                    ))



    # now every ['label'] has multiple assignment,
    # and the loss should come from the inconsistency here

    # get the model
    ModelCls = scaffold.build(graph)  # or should it be model = graph.build()
    # NB: Link in the graph make be use to provide non parameterized
    #     transformation, what is a core feature of our graph.
    #     Is there a better semantic interface design?
    model = ModelCls(data.vocab)

    return model


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
    # data
    reader = Reader()
    train_dataset = reader.read(os.path.join(relative_path, train_path))
    valid_dataset = reader.read(os.path.join(relative_path, valid_path))
    data = Data(train_dataset, valid_dataset)

    scaffold = AllennlpScaffold()

    # model from graph
    model = make_model(graph, data, scaffold)

    # trainer for model
    trainer = get_trainer(graph, model, data, scaffold,
                          lr=LR, wd=WD, batch=BATCH, epoch=EPOCH, patience=PATIENCE)

    # train the model
    trainer.train()

    # save the model
    with open("/tmp/model_emr.th", 'wb') as fout:
        torch.save(model.state_dict(), fout)
    data.vocab.save_to_files("/tmp/vocab_emr")


if __name__ == '__main__':
    main()
