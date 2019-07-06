from typing import List, Dict, Any, NoReturn
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from torch.nn import Module, Dropout, Sequential, GRU, Linear, LogSoftmax
from ...utils import prod
from ...graph import Property
from .base import AllenNlpLearner, SinglePreLearner, SinglePreMaskedLearner
from .sensor import SentenceSensor, SinglePreMaskedSensor, SentenceEmbedderSensor


class SentenceEmbedderLearner(SentenceEmbedderSensor, AllenNlpLearner):
    def create_module(self):
        self.embedding = Embedding(
            num_embeddings=0, # later load or extend
            embedding_dim=self.embedding_dim,
            vocab_namespace=self.key,
            trainable=True,
        )
        return BasicTextFieldEmbedder({self.key: self.embedding})


class RNNLearner(SinglePreMaskedLearner):
    class DropoutRNN(Module):
        def __init__(self, embedding_dim, layers=1, dropout=0.5, bidirectional=True):
            Module.__init__(self)

            from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
            self.rnn = PytorchSeq2SeqWrapper(GRU(embedding_dim,
                                                 embedding_dim,
                                                 num_layers=layers,
                                                 batch_first=True,
                                                 dropout=dropout,
                                                 bidirectional=bidirectional))
            self.dropout = Dropout(dropout)

        def forward(self, x, mask):
            return self.dropout(self.rnn(x, mask))

    def create_module(self):
        return RNNLearner.DropoutRNN(prod(self.pre_dim), self.layers, self.dropout, self.bidirectional)

    def update_output_dim(self):
        if self.bidirectional:
            self.output_dim = (prod(self.pre_dim) * 2,)
        else:
            self.output_dim = (prod(self.pre_dim),)

    def __init__(
        self,
        pre: Property,
        output_only: bool=False,
        layers: int=1,
        dropout: float=0.5,
        bidirectional: bool=True
    ) -> NoReturn:
        self.layers = layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        SinglePreMaskedLearner.__init__(self, pre, output_only=output_only)


class MLPLearner(SinglePreLearner, SinglePreMaskedSensor):
    def create_module(self):
        dims = self.dims.copy()
        dims.insert(0, prod(self.pre_dim))
        layers = []
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            layers.append(Linear(in_features=dim_in, out_features=dim_out))
        module = Sequential(*layers)
        return module

    def update_output_dim(self):
        self.output_dim = (self.dims[-1],)

    def __init__(
        self,
        dims: List[int],
        pre: Property,
        output_only: bool=False
    ) -> NoReturn:
        self.dims = dims # make sure before create_module
        SinglePreLearner.__init__(self, pre, output_only=output_only)


class LogisticRegressionLearner(SinglePreLearner, SinglePreMaskedSensor):
    def create_module(self):
        fc = Linear(in_features=prod(self.pre_dim), out_features=2)
        sm = LogSoftmax(dim=-1)
        module = Sequential(fc,
                            #sm,
                           )
        return module

    def update_output_dim(self):
        self.output_dim = (2,)

    def __init__(
        self,
        pre: Property,
        output_only: bool=False
    ) -> NoReturn:
        SinglePreLearner.__init__(self, pre, output_only=output_only)
