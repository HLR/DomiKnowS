import abc
from typing import List, Dict
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask
import torch


class BaseTagger(Model):
    __metaclass__ = abc.ABCMeta

    import logging
    logger = logging.getLogger(__name__)

    def __init__(self, vocab: Vocabulary) -> None:
        super().__init__(vocab)

    @abc.abstractmethod
    def _forward(self, **data): pass

    def _update_metrics(self, **data):
        for metric_name, metric in self.metrics.items():
            metric(data['logits'], data['labels'], data['metric_mask'])
            data[metric_name] = metric.get_metric(False) # no reset
        return data

    def _update_loss(self, **data):
        if self.loss_func is not None:
            data['loss'] = self.loss_func(**data)
        return data

    def forward(self, **data) -> Dict[str, torch.Tensor]:
        text_mask = get_text_field_mask(data['sentence'])
        self.logger.debug(text_mask)
        data['text_mask'] = text_mask

        data = self._forward(**data)

        if 'labels' in data and data['labels'] is not None:
            labels_mask = data['labels_mask']
            self.logger.debug(labels_mask)
            # label mask apply to metric and loss
            metric_mask = text_mask & labels_mask.type_as(text_mask)
            self.logger.debug(metric_mask)
            data['metric_mask'] = metric_mask

            data = self._update_metrics(**data)
            data = self._update_loss(**data)

        return data

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        output = {}
        for metric_name, metric in self.metrics.items():
            output[metric_name] = metric.get_metric(reset)
        return output


class Tagger(BaseTagger):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(
            in_features=encoder.get_output_dim(),
            out_features=vocab.get_vocab_size('labels'))
        self.metrics = {'accuracy': CategoricalAccuracy()}

    def _forward(self, **data) -> Dict[str, torch.Tensor]:
        sentence = data['sentence']
        mask = data['text_mask']
        
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.hidden2tag(encoder_out)
        data['logits'] = logits
        return data


# loss function
from allennlp.nn.util import sequence_cross_entropy_with_logits


def sequence_cross_entropy_with_logits_loss_func(**data):
    logits = data['logits']
    labels = data['labels']
    mask = data['metric_mask']
    return sequence_cross_entropy_with_logits(logits, labels, mask)


# model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper


def get_model(vocab, emb_dim, hid_dim):
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=emb_dim)
    word_embeddings = BasicTextFieldEmbedder({'tokens': token_embedding})
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(
        emb_dim, hid_dim, batch_first=True))
    model = Tagger(word_embeddings, lstm, vocab)
    return model


# trainer
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import BucketIterator
import torch.optim as optim


def get_trainer(model, loss_func, train_dataset, validation_dataset, lr=0.1, batch=16, epoch=1000, patience=10):
    model.loss_func = loss_func

    # prepare GPU
    if torch.cuda.is_available():
        device = 0
        model = model.cuda(device)
    else:
        device = -1

    # prepare optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)
    iterator = BucketIterator(batch_size=batch, sorting_keys=[
                              ('sentence', 'num_tokens')])
    iterator.index_with(model.vocab)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      patience=patience,
                      num_epochs=epoch,
                      cuda_device=device)

    return trainer
