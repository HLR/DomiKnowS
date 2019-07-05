from typing import List, Dict, Any, NoReturn
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from torch.nn import Module, Dropout, Sequential, GRU, Linear, LogSoftmax
from ...graph import Property
from .base import SinglePreLearner, SinglePreMaskedLearner
from .sensor import SentenceSensor, SinglePreMaskedSensor


class SentenceEmbedderLearner(SinglePreMaskedLearner):
    def __init__(
        self,
        key: str,
        embedding_dim: int,
        pre,
    ) -> NoReturn:
        self.embedding = Embedding(
            num_embeddings=0, # later load or extend
            embedding_dim=embedding_dim,
            vocab_namespace=key
        )
        module = BasicTextFieldEmbedder({key: self.embedding})
        SinglePreMaskedLearner.__init__(self, module, pre)

        for name, pre_sensor in pre.find(SentenceSensor):
            break
        else:
            raise TypeError()

        self.key = key
        self.tokens_key = pre_sensor.key # used by reader.update_textfield()
        pre_sensor.add_embedder(key, self)

    def update_context(
        self,
        context: Dict[str, Any],
        force=False
    ) -> Dict[str, Any]:
        if self.fullname in context and isinstance(context[self.fullname], dict):
            context[self.fullname + '_index'] = context[self.fullname] # reserve
            force = True
        return SinglePreMaskedLearner.update_context(self, context, force)

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        return self.module(context[self.fullname])

    def get_mask(self, context: Dict[str, Any]):
        # TODO: make sure update_context has been called
        return get_text_field_mask(context[self.fullname + '_index'])


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

    def __init__(
        self,
        embedding_dim: int,
        pre: Property,
        layers: int=1,
        dropout: float=0.5,
        bidirectional: bool=True
    ) -> NoReturn:
        module = RNNLearner.DropoutRNN(embedding_dim, layers, dropout, bidirectional)
        SinglePreMaskedLearner.__init__(self, module, pre)


class MLPLearner(SinglePreLearner, SinglePreMaskedSensor):
    def __init__(
        self,
        dims: List[int],
        pre: Property,
    ) -> NoReturn:
        layers = []
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            layers.append(Linear(in_features=dim_in, out_features=dim_out))
        module = Sequential(*layers)
        SinglePreLearner.__init__(self, module, pre)


class LogisticRegressionLearner(SinglePreLearner, SinglePreMaskedSensor):
    def __init__(
        self,
        input_dim: int,
        pre: Property
    ) -> NoReturn:
        fc = Linear(in_features=input_dim, out_features=2)
        sm = LogSoftmax(dim=-1)
        module = Sequential(fc,
                            #sm,
                           )
        SinglePreLearner.__init__(self, module, pre)
