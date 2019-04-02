import os
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.training.trainer import Trainer
import torch
import torch.optim as optim

from .conll04reader import Conll04DatasetReader
from .model import LstmTagger


work_dir = "emr"
relative_path = "data/EntityMentionRelation"

EMBEDDING_DIM = 12
HIDDEN_DIM = 16

#import logging
#logging.basicConfig(level=logging.INFO)

def main():
    torch.manual_seed(1)
    # prepare data
    reader = Conll04DatasetReader()
    train_dataset = reader.read(os.path.join(
        work_dir, relative_path, 'conll04_train.corp'))
    validation_dataset = reader.read(os.path.join(
        work_dir, relative_path, 'conll04_test.corp'))
    vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

    # prepare model
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_DIM)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(
        EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
    model = LstmTagger(word_embeddings, lstm, vocab)

    # prepare GPU
    if torch.cuda.is_available():
        device = 0
        model = model.cuda(device)
    else:
        device = -1

    # prepare optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    iterator = BucketIterator(batch_size=128, sorting_keys=[
                              ("sentence", "num_tokens")])
    iterator.index_with(vocab)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      patience=100,
                      num_epochs=5000,
                      cuda_device=device)

    # do the training
    trainer.train()

    # save model
    with open("/tmp/model.th", 'wb') as fout:
        torch.save(model.state_dict(), fout)
    vocab.save_to_files("/tmp/vocabulary")


if __name__ == '__main__':
    main()
