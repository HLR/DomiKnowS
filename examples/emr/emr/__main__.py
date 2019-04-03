import os
from allennlp.data.vocabulary import Vocabulary
import torch

from .conll04reader import Conll04DatasetReader
from .model import get_model, get_trainer


relative_path = "data/EntityMentionRelation"

EMBEDDING_DIM = 16
HIDDEN_DIM = 8
LR = 0.1
BATCH = 16
EPOCH = 1000
PATIENCE = 10

#import logging
#logging.basicConfig(level=logging.INFO)

torch.manual_seed(1)


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

    # get trainer
    trainer = get_trainer(model, vocab, train_dataset, validation_dataset,
                          LR, BATCH, EPOCH, PATIENCE)

    # train the model
    trainer.train()

    # save model
    with open("/tmp/model.th", 'wb') as fout:
        torch.save(model.state_dict(), fout)
    vocab.save_to_files("/tmp/vocabulary")


if __name__ == '__main__':
    main()
