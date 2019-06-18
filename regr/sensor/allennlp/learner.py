from typing import List, Dict, Any, NoReturn
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from torch.nn import Module, Dropout, Sequential, GRU, Linear, LogSoftmax
from ...graph import Property
from .base import SinglePreLearner, SinglePreMaskedLearner
from .sensor import PhraseSequenceSensor


class W2VLearner(SinglePreMaskedLearner):
    class W2V(Module):
        def __init__(self, sensor, embedding_dim, dropout=0.5):
            Module.__init__(self)

            self.token_embedding = Embedding(
                num_embeddings=sensor.vocab.get_vocab_size(sensor.tokenname),
                embedding_dim=embedding_dim)
            self.word_embeddings = BasicTextFieldEmbedder({sensor.tokenname: self.token_embedding})
            self.dropout = Dropout(dropout)

        def forward(self, x, mask):
            return self.dropout(self.word_embeddings(x))

    def __init__(
        self,
        embedding_dim: int,
        *pres: List[Property]
    ) -> NoReturn:
        pre = pres[0]
        for sensor in pre.values():
            # FIXME: pre is a property! how do we know which sensor? using the first PhraseSequenceSensor
            if isinstance(sensor, PhraseSequenceSensor):
                break
        else:
            raise TypeError('{} takes a PhraseSequenceSensor as pre-required sensor, while a {} instance is given.'.format(type(self), type(pre)))

        module = W2VLearner.W2V(sensor, embedding_dim)
        SinglePreMaskedLearner.__init__(self, module, *pres)


class RNNLearner(SinglePreMaskedLearner):
    class DropoutRNN(Module):
        def __init__(self, embedding_dim, dropout=0.5, bidirectional=True):
            Module.__init__(self)

            from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
            self.rnn = PytorchSeq2SeqWrapper(GRU(embedding_dim,
                                                 embedding_dim,
                                                 batch_first=True,
                                                 dropout=dropout,
                                                 bidirectional=bidirectional))
            self.dropout = Dropout(dropout)

        def forward(self, x, mask):
            return self.dropout(self.rnn(x, mask))

    def __init__(
        self,
        embedding_dim: int,
        *pres: List[Property]
    ) -> NoReturn:
        module = RNNLearner.DropoutRNN(embedding_dim)
        SinglePreMaskedLearner.__init__(self, module, *pres)


class LogisticRegressionLearner(SinglePreLearner):
    def __init__(
        self,
        input_dim: int,
        *pres: List[Property]
    ) -> NoReturn:
        fc = Linear(in_features=input_dim, out_features=2)
        sm = LogSoftmax(dim=-1)
        module = Sequential(fc, sm)
        SinglePreLearner.__init__(self, module, *pres)
