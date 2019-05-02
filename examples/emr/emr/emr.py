import os
import torch
from allennlp.models.model import Model
from regr import Graph
from regr.scaffold import Scaffold, AllennlpScaffold

if __package__ is None or __package__ == '':
    # uses current directory visibility
    from data import Data, EMRPeopWorkforOrgReader
    from models import get_trainer, datainput, word2vec, fullyconnected, cartesianprod_concat
    from graph import graph
else:
    # uses current package visibility
    from .data import Data, EMRPeopWorkforOrgReader
    from .models import get_trainer, datainput, word2vec, fullyconnected, cartesianprod_concat
    from .graph import graph


# data setting
relative_path = "data/EntityMentionRelation"
train_path = "conll04_train.corp"
valid_path = "conll04_test.corp"

# model setting
EMBEDDING_DIM = 16

# training setting
LR = 1
BATCH = 128
EPOCH = 1000
PATIENCE = 10


# develop by an ML programmer to wire nodes in the graph and ML Models
def make_model(graph: Graph,
               data: Data,
               scaffold: Scaffold
               ) -> Model:
    # get concepts from graph
    word = graph.word
    people = graph.people
    organization = graph.organization
    workfor = graph.workfor
    pair = graph.pair

    # binding
    graph.release()  # release anything binded before new assignment

    # filling in data and label
    scaffold.assign(word, 'index', datainput(data['sentence']))
    scaffold.assign(people, 'label', datainput(data['Peop_labels']))
    scaffold.assign(organization, 'label', datainput(data['Org_labels']))
    scaffold.assign(workfor, 'label', datainput(data['relation_labels']))

    # building model
    scaffold.assign(word, 'w2v',
                    word2vec(
                        word['index'],
                        data.vocab.get_vocab_size('tokens'),
                        EMBEDDING_DIM,
                        'tokens'
                    ))
    scaffold.assign(people, 'label',
                    fullyconnected(
                        word['w2v'],
                        EMBEDDING_DIM,
                        2
                    ))
    scaffold.assign(organization, 'label',
                    fullyconnected(
                        word['w2v'],
                        EMBEDDING_DIM,
                        2
                    ))
    # TODO: pair['w2v'] should be infer from word['w2v'] according to their relationship
    # but we specify it here to make it a bit easier for implementation
    scaffold.assign(pair, 'w2v',
                    cartesianprod_concat(
                        word['w2v']
                    ))
    scaffold.assign(workfor, 'label',
                    fullyconnected(
                        pair['w2v'],
                        EMBEDDING_DIM * 2,
                        2
                    ))
    # now people['label'] has multiple assignment,
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
    reader = EMRPeopWorkforOrgReader()
    train_dataset = reader.read(os.path.join(relative_path, train_path))
    valid_dataset = reader.read(os.path.join(relative_path, valid_path))
    data = Data(train_dataset, valid_dataset)

    scaffold = AllennlpScaffold()

    # model from graph
    model = make_model(graph, data, scaffold)

    # trainer for model
    trainer = get_trainer(graph, model, data, scaffold)

    # train the model
    trainer.train()

    # save the model
    with open("/tmp/model_emr.th", 'wb') as fout:
        torch.save(model.state_dict(), fout)
    data.vocab.save_to_files("/tmp/vocab_emr")


if __name__ == '__main__':
    main()
