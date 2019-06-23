from typing import List, Dict, Any, NoReturn
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from torch.nn import Module, Dropout, Sequential, GRU, Linear, LogSoftmax
from ...graph import Property
from .base import SinglePreLearner, SinglePreMaskedLearner
from .sensor import TokenInSequenceSensor, SinglePreMaskedSensor


class W2VLearner(SinglePreMaskedLearner):
    class W2V(Module):
        def __init__(self, pre_sensor, embedding_dim, dropout=0.5):
            Module.__init__(self)

            self.token_embedding = Embedding(
                num_embeddings=0, # later load or extend
                embedding_dim=embedding_dim,
                vocab_namespace=pre_sensor.fullname
            )
            self.word_embeddings = BasicTextFieldEmbedder({pre_sensor.fullname: self.token_embedding})
            self.dropout = Dropout(dropout)

        def forward(self, x, mask):
            return self.dropout(self.word_embeddings(x))

    def __init__(
        self,
        embedding_dim: int,
        pre: Property
    ) -> NoReturn:
        for name, sensor in pre.find(TokenInSequenceSensor):
            break
        else:
            raise TypeError('{} takes a TokenInSequenceSensor as pre-required sensor, what cannot be found in a {} instance is given.'.format(self.fullname, type(pre)))

        module = W2VLearner.W2V(sensor, embedding_dim)
        SinglePreMaskedLearner.__init__(self, module, pre)
        self.sequence = sensor.pre

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        return self.module(context[self.sequence.fullname], self.get_mask(context)) # need sequence as input


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
        pre: Property
    ) -> NoReturn:
        module = RNNLearner.DropoutRNN(embedding_dim)
        SinglePreMaskedLearner.__init__(self, module, pre)


class LogisticRegressionLearner(SinglePreLearner, SinglePreMaskedSensor):
    def __init__(
        self,
        input_dim: int,
        pre: Property
    ) -> NoReturn:
        fc = Linear(in_features=input_dim, out_features=2)
        sm = LogSoftmax(dim=-1)
        module = Sequential(fc, sm)
        SinglePreLearner.__init__(self, module, pre)
