from typing import List, Dict, Any, Optional, NoReturn
import torch
from torch.nn import Module
from allennlp.nn.util import get_text_field_mask
from allennlp.data.token_indexers import SingleIdTokenIndexer
from ...graph import Property
from .base import AllenNlpReaderSensor, AllenNlpModuleSensor, SinglePreMaskedSensor, MaskedSensor


class SequenceSensor(MaskedSensor):
    def __init__(
        self,
        reader,
        key: str,
        output_only: Optional[bool]=False
    ) -> NoReturn:
        AllenNlpReaderSensor.__init__(self, reader, key, output_only=output_only) # *pres=[]
        self.tokens = []

    def add_token(self, sensor):
        self.tokens.append(sensor)
        self.reader.token_indexers_cls[self][sensor] = lambda s: SingleIdTokenIndexer(namespace=s.get_fullname('_'))

    def get_mask(self, context: Dict[str, Any]):
        return get_text_field_mask(context[self.fullname])


class TokenInSequenceSensor(SinglePreMaskedSensor):
    def __init__(
        self,
        pre,
        output_only: Optional[bool]=False
    ) -> NoReturn:
        SinglePreMaskedSensor.__init__(self, pre, output_only=output_only)
        for name, sensor in pre.find(SequenceSensor):
            break
        else:
            raise TypeError('{} takes a SequenceSensor as pre-required sensor, what cannot be found in a {} instance is given.'.format(self.fullname, type(pre)))
        sensor.add_token(self)

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        seq_dict = context[self.pre.fullname]
        return seq_dict[self.get_fullname('_')]


class LabelSensor(AllenNlpReaderSensor):
    def __init__(
        self,
        reader,
        key: str,
        output_only: bool=True
    ) -> NoReturn:
        AllenNlpReaderSensor.__init__(self, reader, key, output_only=output_only)


class CartesianProductSensor(AllenNlpModuleSensor, MaskedSensor):
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

    def get_mask(self, context: Dict[str, Any]):
        for name, sensor in self.pre.find(MaskedSensor):
            break
        else:
            print(self.pre)
            raise RuntimeError('{} require at least one pre-required sensor to be MaskedSensor.'.format(self.fullname))

        mask = sensor.get_mask(context).float()
        ms = mask.size()
        mask = mask.view(ms[0], ms[1], 1).matmul(
            mask.view(ms[0], 1, ms[1]))  # (b,l,l)
        return mask
