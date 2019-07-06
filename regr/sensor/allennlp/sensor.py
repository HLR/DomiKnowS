from typing import List, Dict, Any, NoReturn
from collections import OrderedDict
import torch
from torch.nn import Module
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from ...utils import prod
from ...graph import Property
from .base import ReaderSensor, ModuleSensor, SinglePreMaskedSensor, MaskedSensor, PreArgsModuleSensor


class SentenceSensor(ReaderSensor):
    def __init__(
        self,
        reader,
        key: str,
        output_only: bool=False
    ) -> NoReturn:
        ReaderSensor.__init__(self, reader, key, output_dim=(), output_only=output_only) # *pres=[]
        self.embedders = OrderedDict() # list of SentenceEmbedderLearner

    def add_embedder(self, key, embedder):
        self.reader.claim(key, embedder)
        self.embedders[key] = embedder

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        # This sensor it self can do nothing
        # mayby with self.embedders something more can happen?
        return None


class LabelSensor(ReaderSensor):
    def __init__(
        self,
        reader,
        key: str,
        output_only: bool=True
    ) -> NoReturn:
        ReaderSensor.__init__(self, reader, key, output_dim=(), output_only=output_only)


class ConcatSensor(PreArgsModuleSensor, MaskedSensor):
    class Concat(Module):
        def forward(self, *x):
            # TODO: flatten
            return torch.cat(x, dim=-1)

    def create_module(self):
        return ConcatSensor.Concat()

    def update_output_dim(self):
        output_dim = 0
        for pre_dim in self.pre_dims:
            if len(pre_dim) == 0:
                output_dim += 1
            else:
                output_dim += prod(pre_dim) # assume flatten
        self.output_dim = (output_dim,)

    def get_mask(self, context: Dict[str, Any]):
        for pre in self.pres:
            for name, sensor in pre.find(MaskedSensor):
                return sensor.get_mask(context)
            else:
                # not found
                continue
            # found
            break
        else:
            raise RuntimeError('{} require at least one pre-required sensor to be MaskedSensor.'.format(self.fullname))

        return None # not going to here


class CartesianProductSensor(PreArgsModuleSensor, SinglePreMaskedSensor):
    class CP(Module):
        def forward(self, x, y):  # (b,l1,f1) x (b,l2,f2) -> (b, l1, l2, f1+f2)
            # TODO: flatten
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

    def create_module(self):
        return CartesianProductSensor.SelfCP()

    def update_output_dim(self):
        if len(self.pre_dim) == 0:
            output_dim = 2
        else:
            output_dim = prod(self.pre_dim) * 2 # assume flatten
        self.output_dim = (output_dim,)

    def __init__(
        self,
        pre: Property,
        output_only: bool=False
    ) -> NoReturn:
        self.pre = pre
        PreArgsModuleSensor.__init__(self, pre, output_only=output_only)


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


class SentenceEmbedderSensor(SinglePreMaskedSensor, ModuleSensor):
    def create_module(self):
        self.embedding = Embedding(
            num_embeddings=0, # later load or extend
            embedding_dim=self.embedding_dim,
            vocab_namespace=self.key,
            trainable=False,
        )
        return BasicTextFieldEmbedder({self.key: self.embedding})

    def __init__(
        self,
        key: str,
        embedding_dim: int,
        pre,
        output_only: bool=False
    ) -> NoReturn:
        self.key = key
        self.embedding_dim = embedding_dim
        self.pre = pre
        ModuleSensor.__init__(self, pre, output_only=output_only)

        for name, pre_sensor in pre.find(SentenceSensor):
            pre_sensor.add_embedder(key, self)
            self.tokens_key = pre_sensor.key # used by reader.update_textfield()
            break
        else:
            raise TypeError()

    def update_context(
        self,
        context: Dict[str, Any],
        force=False
    ) -> Dict[str, Any]:
        if self.fullname in context and isinstance(context[self.fullname], dict):
            context[self.fullname + '_index'] = context[self.fullname] # reserve
            force = True
        return SinglePreMaskedSensor.update_context(self, context, force)

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        return self.module(context[self.fullname])

    def get_mask(self, context: Dict[str, Any]):
        # TODO: make sure update_context has been called
        return get_text_field_mask(context[self.fullname + '_index'])
