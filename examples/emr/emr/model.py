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
    def _forward(self,
                 sentence: Dict[str, torch.Tensor],
                 mask: torch.Tensor): pass

    def _update_metrics(self, output, labels, mask):
        for metric_name, metric in self.metrics.items():
            metric(output['logits'], labels, mask)
            output[metric_name] = metric.get_metric(False)

    def _update_loss(self, output, labels, mask):
        if self.loss_func is not None:
            output['loss'] = self.loss_func(output["logits"], labels, mask)

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None,
                labels_mask: List[torch.LongTensor] = None,
                ) -> Dict[str, torch.Tensor]:
        text_mask = get_text_field_mask(sentence)
        self.logger.debug(text_mask)

        output = self._forward(sentence, text_mask)

        if labels is not None:
            self.logger.debug(labels_mask)
            # label mask apply to metric and loss
            metric_mask = text_mask & labels_mask.type_as(text_mask)
            self.logger.debug(metric_mask)

            self._update_metrics(output, labels, metric_mask)
            self._update_loss(output, labels, metric_mask)

        return output

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
        self.metrics = {"accuracy": CategoricalAccuracy()}

    def _forward(self,
                 sentence: Dict[str, torch.Tensor],
                 mask: torch.Tensor,
                 ) -> Dict[str, torch.Tensor]:
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.hidden2tag(encoder_out)
        output = {'logits': logits}
        return output


# loss function
from allennlp.nn.util import sequence_cross_entropy_with_logits


def sequence_cross_entropy_with_logits_loss_func(logits, labels, mask):
    return sequence_cross_entropy_with_logits(logits, labels, mask)


# model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper


def get_model(vocab, emb_dim, hid_dim):
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=emb_dim)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(
        emb_dim, hid_dim, batch_first=True))
    model = Tagger(word_embeddings, lstm, vocab)
    return model


# trainer
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import BucketIterator
import torch.optim as optim


def get_trainer(model, loss_func, vocab, train_dataset, validation_dataset, lr=0.1, batch=16, epoch=1000, patience=10):
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
                              ("sentence", "num_tokens")])
    iterator.index_with(vocab)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      patience=patience,
                      num_epochs=epoch,
                      cuda_device=device)

    return trainer
