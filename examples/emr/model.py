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
