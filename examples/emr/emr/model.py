from typing import List, Dict
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
import torch


class LstmTagger(Model):
    import logging
    logger = logging.getLogger(__name__)

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
        self.accuracy = CategoricalAccuracy()

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None,
                labels_mask: List[torch.LongTensor] = None,
                ) -> Dict[str, torch.Tensor]:
        text_mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, text_mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        if labels is not None:
            LstmTagger.logger.debug(labels_mask)
            LstmTagger.logger.debug(text_mask)
            # label mask apply to metric and loss
            metric_mask = text_mask & labels_mask.type_as(text_mask)
            LstmTagger.logger.debug(metric_mask)
            self.accuracy(tag_logits, labels, metric_mask)
            output["loss"] = sequence_cross_entropy_with_logits(
                tag_logits, labels, metric_mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper


def get_model(vocab, emb_dim, hid_dim):
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=emb_dim)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(
        emb_dim, hid_dim, batch_first=True))
    model = LstmTagger(word_embeddings, lstm, vocab)
    return model


from allennlp.training.trainer import Trainer
from allennlp.data.iterators import BucketIterator
import torch.optim as optim


def get_trainer(model, vocab, train_dataset, validation_dataset, lr=0.1, batch=16, epoch=1000, patience=10):
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
