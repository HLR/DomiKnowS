import os
import torch
from allennlp.models.model import Model
from regr.graph import Graph
from regr.scaffold import Scaffold, AllennlpScaffold

if __package__ is None or __package__ == '':
    # uses current directory visibility
    from data import Data, Conll04TokenBinaryReader
    from models import get_trainer, datainput, word2vec, fullyconnected
    from graph import graph
else:
    # uses current package visibility
    from .data import Data, Conll04TokenBinaryReader
    from .models import get_trainer, datainput, word2vec, fullyconnected
    from .graph import graph


# App setting
entity_label_configs = {'people': {'entity_name': 'people',
                                   'label_name': 'Peop'},
                        'organization': {'entity_name': 'organization',
                                         'label_name': 'Org'},
                        'location': {'entity_name': 'location',
                                     'label_name': 'Loc'},
                        'other': {'entity_name': 'other',
                                  'label_name': 'Other'},
                        }

# data setting
relative_path = "data/EntityMentionRelation"
train_path = "conll04_train.corp"
valid_path = "conll04_test.corp"

# model setting
EMBEDDING_DIM = 64

# training setting
LR = 0.001
WD = 0.0001
BATCH = 128
EPOCH = 200
PATIENCE = None


# develop by an ML programmer to wire nodes in the graph and ML Models
def make_model(graph: Graph,
               data: Data,
               scaffold: Scaffold,
               entity_name: str,
               label_name: str
               ) -> Model:
    # initialize the graph
    graph.release()  # release anything binded before new assignment

    # get concepts from graph
    phrase = graph.linguistic.phrase
    # graph.people / graph.organization
    entity = graph.application[entity_name]

    # filling in data and label
    scaffold.assign(phrase, 'index', datainput(data['sentence']))
    scaffold.assign(entity, 'label', datainput(data[label_name]))

    # building model
    scaffold.assign(phrase, 'emb',
                    word2vec(
                        phrase['index'],
                        data.vocab.get_vocab_size('tokens'),
                        EMBEDDING_DIM,
                        'tokens'
                    ))
    scaffold.assign(entity, 'label',
                    fullyconnected(
                        phrase['emb'],
                        EMBEDDING_DIM,
                        2
                    ))
    # now entity['label'] has multiple assignment,
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
    # config
    entity = 'people'
    config = entity_label_configs[entity]

    # data
    reader = Conll04TokenBinaryReader({config['label_name'],})
    train_dataset = reader.read(os.path.join(relative_path, train_path))
    valid_dataset = reader.read(os.path.join(relative_path, valid_path))
    data = Data(train_dataset, valid_dataset)

    scaffold = AllennlpScaffold()

    # model from graph
    model = make_model(graph, data, scaffold, config['entity_name'], config['label_name'])

    # trainer for model
    batch = BATCH * 24  # multiply by average len, so compare to sentence level experiments
    trainer = get_trainer(graph, model, data, scaffold,
                          lr=LR, wd=WD, batch=batch, epoch=EPOCH, patience=PATIENCE)

    # train the model
    trainer.train()

    # save the model
    with open("/tmp/model_ner.th", 'wb') as fout:
        torch.save(model.state_dict(), fout)
    data.vocab.save_to_files("/tmp/vocab_ner")


if __name__ == '__main__':
    main()
