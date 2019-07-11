from typing import List, Dict, Any, Optional, NoReturn
import torch
from torch.nn import Module
from allennlp.nn.util import get_text_field_mask
from ...graph import Property
from .base import AllenNlpReaderSensor, AllenNlpModuleSensor, MaskedSensor


class SentenceSensor(AllenNlpReaderSensor):
    pass


class PhraseSensor(AllenNlpReaderSensor):
    pass


class PhraseSequenceSensor(AllenNlpReaderSensor, MaskedSensor):
    def __init__(
        self,
        vocab,
        fieldname: str,
        tokenname: str='tokens',
        output_only: Optional[bool]=False
    ) -> NoReturn:
        AllenNlpReaderSensor.__init__(self, fieldname, output_only=output_only) # *pres=[]
        self.vocab = vocab
        self.tokenname = tokenname

    def get_mask(self, context: Dict[str, Any]):
        return get_text_field_mask(context[self.fieldname])


class LabelSensor(AllenNlpReaderSensor):
    pass


class LabelSequenceSensor(AllenNlpReaderSensor):
    pass


class CartesianProductSensor(AllenNlpModuleSensor):
    class CP(Module):
        def forward(self, x, y):  # (b,l1,f1) x (b,l2,f2) -> (b, l1, l2, f1+f2)
            xs = x.size()
            ys = y.size()
            assert xs[0] == ys[0]
            # torch cat is not broadcasting, do repeat manually
            xx = x.view(xs[0], xs[1], 1, xs[2]).repeat(1, 1, ys[1], 1)
            yy = y.view(ys[0], 1, ys[1], ys[2]).repeat(1, xs[1], 1, 1)
            return torch.cat([xx, yy], dim=3)

    class SelfCP(CP):
        def forward(self, x):
            return CartesianProductSensor.CP.forward(self, x, x)

    def __init__(
        self,
        *pres: List[Property]
    ) -> NoReturn:
        if len(pres) != 1:
            raise ValueError(
                '{} take one pre-required sensor, {} given.'.format(type(self), len(pres)))
        module = CartesianProductSensor.SelfCP()
        AllenNlpModuleSensor.__init__(self, module, *pres)
        self.pre = self.pres[0]

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        return self.module(context[self.pre.fullname])
