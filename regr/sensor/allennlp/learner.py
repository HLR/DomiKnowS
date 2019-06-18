from typing import List, Dict, Optional, Any, NoReturn
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
import torch
from torch.nn import Module, Dropout, Sequential, GRU, Linear, LogSoftmax
from ...graph import Property
from .. import Learner, Sensor
from .sensor import AllenNlpModuleSensor, PhraseSequenceSensor


class AllenNlpLearner(AllenNlpModuleSensor, Learner):
    def parameters(self):
        return self.module.parameters()


class PreArgsLearner(AllenNlpLearner):
    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        try:
            return self.module(*(context[pre.fullname] for pre in self.pres))
        except:
            print('Error during forward with sensor {}'.format(self.fullname))
            print('module:', self.module)
            raise
        return None


class SinglePreLearner(PreArgsLearner):
    def __init__(
        self,
        module: Module,
        *pres: List[Property]
    ) -> NoReturn:
        if len(pres) != 1:
            raise ValueError(
                '{} take one pre-required sensor, {} given.'.format(type(self), len(pres)))
        PreArgsLearner.__init__(self, module, *pres)
        self.pre = self.pres[0]


class W2VLearner(SinglePreLearner):
    def __init__(
        self,
        embedding_dim: int,
        *pres: List[Property]
    ) -> NoReturn:
        pre = pres[0]
        for presensor in pre.values():
            # FIXME: pre is a property! how do we know which sensor? using the first PhraseSequenceSensor
            if isinstance(presensor, PhraseSequenceSensor):
                break
        else:
            raise TypeError('{} takes a PhraseSequenceSensor as pre-required sensor, while a {} instance is given.'.format(type(self), type(pre)))

        self.token_embedding = Embedding(
            num_embeddings=presensor.vocab.get_vocab_size(presensor.tokenname),
            embedding_dim=embedding_dim)
        word_embeddings = BasicTextFieldEmbedder({presensor.tokenname: self.token_embedding})
        dropout = Dropout(0.5)
        module = Sequential(word_embeddings, dropout)

        SinglePreLearner.__init__(self, module, *pres)


class MaskedSinglePreLearner(SinglePreLearner):
    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        # TODO: something like context['mask'] = self.sup.sup.findbase().get_mask()
        return self.module(context[self.pre.fullname], context['mask'])


class RNNLearner(MaskedSinglePreLearner):
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
        MaskedSinglePreLearner.__init__(self, module, *pres)


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
