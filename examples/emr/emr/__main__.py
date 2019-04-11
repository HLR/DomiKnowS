import os
from allennlp.data.vocabulary import Vocabulary
import torch

from .data import Conll04DatasetReader
from .model import get_model, get_trainer
# NB: the loss_func will be replace by graph
from .graph import graph
from .graph import phrase
from .graph import people, organization, location, other, o
from .graph import work_for, located_in, live_in, orgbase_on

relative_path = "data/EntityMentionRelation"

EMBEDDING_DIM = 16
HIDDEN_DIM = 8
LR = 0.1
BATCH = 16
EPOCH = 1000
PATIENCE = 10

#import logging
# logging.basicConfig(level=logging.INFO)

torch.manual_seed(1)


def get_graph_loss_func(model):
    from allennlp.nn.util import sequence_cross_entropy_with_logits

    def graph_loss_func(**data):
        logits = data['logits']
        labels = data['labels']
        mask = data['metric_mask']
        return sequence_cross_entropy_with_logits(logits, labels, mask)

        # TODO: this part is not really running now
        graph.release()

        # model.output['logits'] - (batch, len, class)
        # bind output logits
        people['prob'] = logits[:, :, 0]
        organization['prob'] = logits[:, :, 1]
        location['prob'] = logits[:, :, 2]
        other['prob'] = logits[:, :, 3]
        o['prob'] = logits[:, :, 4]
        # bind labels
        labels_onehot = to_onehot(labels)
        people['prob'] = labels_onehot[:, :, 0]
        organization['prob'] = labels_onehot[:, :, 1]
        location['prob'] = labels_onehot[:, :, 2]
        other['prob'] = labels_onehot[:, :, 3]
        o['prob'] = labels_onehot[:, :, 4]

        return graph()
    return graph_loss_func


def main():
    # prepare data
    reader = Conll04DatasetReader()
    train_dataset = reader.read(os.path.join(
        relative_path, 'conll04_train.corp'))
    validation_dataset = reader.read(os.path.join(
        relative_path, 'conll04_test.corp'))
    vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

    # get model
    model = get_model(vocab, EMBEDDING_DIM, HIDDEN_DIM)

    # TODO:
    # bind the output of model to the graph
    # and
    # get a loss_func out of a graph
    # loss_func = ...
    loss_func = get_graph_loss_func(model)

    # get trainer
    trainer = get_trainer(model, loss_func, train_dataset, validation_dataset,
                          LR, BATCH, EPOCH, PATIENCE)

    # train the model
    trainer.train()

    # save model
    with open("/tmp/model.th", 'wb') as fout:
        torch.save(model.state_dict(), fout)
    vocab.save_to_files("/tmp/vocabulary")


if __name__ == '__main__':
    main()
