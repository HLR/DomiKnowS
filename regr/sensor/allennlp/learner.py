from typing import List, Dict, Optional, Any, NoReturn
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
import torch
from torch.nn import Module, Dropout, Sequential, GRU, Linear, LogSoftmax
from .. import Learner, Sensor
from . import AllenNlpSensor


class AllenNlpLearner(AllenNlpSensor, Learner):
    def __init__(
        self,
        module: Module,
        *pres: List[Sensor]
    ) -> NoReturn:
        Learner.__init__(self)
        self.module = module
        self.pres = pres
        for pre in pres:
            for name, sensor in pre.items():
                if isinstance(sensor, AllenNlpLearner) and (not (hasattr(sensor, 'output_only') and sensor.output_only)):
                    module.add_module(sensor.fullname, sensor.module)

    def update_context(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        for pre in self.pres:
            for name, sensor in pre.items():
                # choose one result to update finally
                # TODO: consider priority or confidence or merge somehow
                if isinstance(sensor, AllenNlpSensor) and (not (hasattr(sensor, 'output_only') and sensor.output_only)):
                    context = sensor(context)
                    break
            else:  # no break
                raise RuntimeError('Not able to find a sensor for {} as prereqiured by {}'.format(
                    pre.fullname, self.fullname))

        return AllenNlpSensor.update_context(self, context)

    def __call__(
        self, *args, **kwargs
    ) -> Any:
        return AllenNlpSensor.__call__(self, *args, **kwargs)

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        return Learner.forward(self, context)


class SinglePreLearner(AllenNlpLearner):
    def __init__(
        self,
        module: Module,
        *pres: List[Sensor]
    ) -> NoReturn:
        if len(pres) != 1:
            raise ValueError(
                '{} take one pre-required sensor, {} given.'.format(type(self), len(pres)))
        AllenNlpLearner.__init__(self, module, *pres)
        self.pre = self.pres[0]

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        return self.module(context[self.pre.fullname])


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
        #dropout = Dropout(0.5)
        module = Sequential(word_embeddings,
                            # dropout
                            )

        SinglePreLearner.__init__(self, module, *pres)


class MaskedSinglePreLearner(SinglePreLearner):
    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        # TODO: something like context['mask'] = self.sup.sup.findbase().get_mask()
        return self.module(context[self.pre.fullname], context['mask'])


class RNNLearner(MaskedSinglePreLearner):
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
        #dropout = Dropout(0.5)
        # module = Sequential(rnn,
        # dropout
        #                   )

        MaskedSinglePreLearner.__init__(self, rnn, *pres)


class LRLearner(MaskedSinglePreLearner):
    class MaskedSequenceLRModule(Module):
        def __init__(self, input_dim):
            Module.__init__(self)
            self.fc = Linear(in_features=input_dim, out_features=2)
            self.sm = LogSoftmax(dim=-1)

        def forward(self, x, mask):
            return self.sm(self.fc(x))

    def __init__(
        self,
        input_dim: int,
        *pres: List[Sensor]
    ) -> NoReturn:
        module = LRLearner.MaskedSequenceLRModule(input_dim)
        MaskedSinglePreLearner.__init__(self, module, *pres)


class CPCatLearner(SinglePreLearner):
    class Cpcat(Module):
        def forward(self, x, y):  # (b,l1,f1) x (b,l2,f2) -> (b, l1, l2, f1+f2)
            xs = x.size()
            ys = y.size()
            assert xs[0] == ys[0]
            # torch cat is not broadcasting, do repeat manually
            xx = x.view(xs[0], xs[1], 1, xs[2]).repeat(1, 1, ys[1], 1)
            yy = y.view(ys[0], 1, ys[1], ys[2]).repeat(1, xs[1], 1, 1)
            return torch.cat([xx, yy], dim=3)

    class SelfCpcat(Cpcat):
        def forward(self, x):
            return CPCatLearner.Cpcat.forward(self, x, x)

    def __init__(
        self,
        *pres: List[Sensor]
    ) -> NoReturn:
        module = CPCatLearner.SelfCpcat()
        SinglePreLearner.__init__(self, module, *pres)
