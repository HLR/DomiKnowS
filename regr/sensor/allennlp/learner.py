from typing import List, Dict, Optional, Any, NoReturn
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from torch.nn import Module, Dropout, Sequential, GRU, Linear, LogSigmoid
from .base import BaseAllenNlpLearner
from ..sensor import Sensor


class SinglePreLearner(BaseAllenNlpLearner):
    def __init__(
        self,
        *pres: List[Sensor],
        module: Optional[Module]=None
    ) -> NoReturn:
        if len(pres) != 1:
            raise ValueError('{} take one pre-required sensor, {} given.'.format(type(self), len(pres)))
        BaseAllenNlpLearner.__init__(self, *pres, module=module)
        
    def forward(
        context: Dict[str, Any]
    ) -> Any:
        return self.module( context[self.pres[0].fullname] )

class W2VLearner(SinglePreLearner):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        token_name: str,
        *pres: List[Sensor]
    ) -> NoReturn:
        token_embedding = Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim)
        word_embeddings = BasicTextFieldEmbedder({
            token_name: token_embedding})
        word_embeddings.add_module('tkn', word_embeddings)
        dropout = Dropout(0.5)
        module = Sequential(word_embeddings, dropout)
        
        SinglePreLearner.__init__(self, *pres, module=module)


class RNNLearner(SinglePreLearner):
    def __init__(
        self,
        embedding_dim: int,
        *pres: List[Sensor]
    ) -> NoReturn:
        from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
        rnn = PytorchSeq2SeqWrapper(GRU(embedding_dim,
                                        embedding_dim,
                                        batch_first=True,
                                        dropout=0.5,
                                        bidirectional=True))
        dropout = Dropout(0.5)
        module = Sequential(rnn, dropout)
        
        SinglePreLearner.__init__(self, *pres, module=module)

    def forward(
        context: Dict[str, Any]
    ) -> Any:
        # TODO: something like context['mask'] = self.sup.sup.findbase().get_mask()
        return self.module( context[self.pres[0].fullname], context['mask'] )

class LRLearner(SinglePreLearner):
    def __init__(
        self,
        input_dim: int,
        *pres: List[Sensor]
    ) -> NoReturn:
        fc = Linear(
            in_features=input_dim,
            out_features=1)
        ls = LogSigmoid()
        
        module = Sequential(fc, ls)
        
        SinglePreLearner.__init__(self, *pres, module=module)
