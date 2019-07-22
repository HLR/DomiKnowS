from typing import List, Dict, Any, NoReturn
from collections import OrderedDict
import torch
from torch.nn import Module
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from ...graph import Property
from ...utils import prod, guess_device
from .base import ReaderSensor, ModuleSensor, SinglePreMaskedSensor, MaskedSensor, PreArgsModuleSensor, SinglePreArgMaskedPairSensor
from .module import Concat, SelfCartesianProduct, SelfCartesianProduct3, NGram, PairTokenDistance, PairTokenDependencyRelation, PairTokenDependencyDistance, LowestCommonAncestor


class SentenceSensor(ReaderSensor):
    def __init__(
        self,
        reader,
        key: str,
        output_only: bool=False
    ) -> NoReturn:
        ReaderSensor.__init__(self, reader, key, output_only=output_only) # *pres=[]
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

    @property
    def output_dim(self):
        return ()


class LabelSensor(ReaderSensor):
    def __init__(
        self,
        reader,
        key: str,
        output_only: bool=True
    ) -> NoReturn:
        ReaderSensor.__init__(self, reader, key, output_only=output_only)

    @property
    def output_dim(self):
        return ()

class ConcatSensor(PreArgsModuleSensor, MaskedSensor):
    def create_module(self):
        return Concat()

    @property
    def output_dim(self):
        output_dim = 0
        for pre_dim in self.pre_dims:
            if len(pre_dim) == 0:
                output_dim += 1
            else:
                output_dim += prod(pre_dim) # assume flatten
        return (output_dim,)

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


class CartesianProductSensor(SinglePreArgMaskedPairSensor):
    def create_module(self):
        return SelfCartesianProduct()

    @property
    def output_dim(self):
        if len(self.pre_dim) == 0:
            output_dim = 2
        else:
            output_dim = prod(self.pre_dim) * 2 # assume flatten
        return (output_dim,)

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

class CartesianProduct3Sensor(SinglePreArgMaskedPairSensor):
    def create_module(self):
        return SelfCartesianProduct3()

    @property
    def output_dim(self):
        if len(self.pre_dim) == 0:
            output_dim = 3
        else:
            output_dim = prod(self.pre_dim) * 3 # assume flatten
        return (output_dim,)

    def get_mask(self, context: Dict[str, Any]):
        for name, sensor in self.pre.find(MaskedSensor):
            break
        else:
            print(self.pre)
            raise RuntimeError('{} require at least one pre-required sensor to be MaskedSensor.'.format(self.fullname))

        mask = sensor.get_mask(context).float()
        ms = mask.size()
        #(b,l,l)
        mask1 = mask.view(ms[0], ms[1], 1).matmul(mask.view(ms[0], 1, ms[1]))
        mask2 = mask1.view(ms[0], ms[1], ms[1], 1).matmul(mask.view(ms[0], 1, 1, ms[1]))
        
        return mask2

class SentenceEmbedderSensor(SinglePreMaskedSensor, ModuleSensor):
    def create_module(self):
        self.embedding = Embedding(
            num_embeddings=0, # later load or extend
            embedding_dim=self.embedding_dim,
            pretrained_file=self.pretrained_file,
            vocab_namespace=self.key,
            trainable=False,
        )
        return BasicTextFieldEmbedder({self.key: self.embedding})

    def __init__(
        self,
        key: str,
        embedding_dim: int,
        pre,
        pretrained_file: str=None,
        output_only: bool=False
    ) -> NoReturn:
        self.key = key
        self.embedding_dim = embedding_dim
        self.pretrained_file = pretrained_file
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


class NGramSensor(PreArgsModuleSensor, SinglePreMaskedSensor):
    def create_module(self):
        return NGram(self.ngram)

    @property
    def output_dim(self):
        #import pdb; pdb.set_trace()
        return tuple(dim * self.ngram for dim in self.pre_dim)

    def __init__(
        self,
        ngram: int,
        pre: Property,
        output_only: bool=False
    ) -> NoReturn:
        self.ngram = ngram
        PreArgsModuleSensor.__init__(self, pre, output_only=output_only)


class TokenDistantSensor(SinglePreArgMaskedPairSensor):
    def create_module(self):
        return PairTokenDistance(self.emb_num, self.window)

    def __init__(
        self,
        emb_num: int,
        window: int,
        pre: Property,
        output_only: bool=False
    ) -> NoReturn:
        self.emb_num = emb_num
        self.window = window
        SinglePreArgMaskedPairSensor.__init__(self, pre, output_only=output_only)

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        device, _ = guess_device(context).most_common(1)[0]
        with torch.cuda.device(device):
            return super().forward(context)


class TokenDepSensor(SinglePreArgMaskedPairSensor):
    def create_module(self):
        return PairTokenDependencyRelation()

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        #import pdb; pdb.set_trace()
        device, _ = guess_device(context).most_common(1)[0]
        with torch.cuda.device(device):
            return super().forward(context)


class TokenLcaSensor(SinglePreArgMaskedPairSensor):
    def create_module(self):
        return LowestCommonAncestor()

    @property
    def output_dim(self):
        return self.pre_dims[1]

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        return PreArgsModuleSensor.forward(self, context)


class TokenDepDistSensor(SinglePreArgMaskedPairSensor):
    def create_module(self):
        return PairTokenDependencyDistance(self.emb_num, self.window)

    def __init__(
        self,
        emb_num: int,
        window: int,
        pre: Property,
        output_only: bool=False
    ) -> NoReturn:
        self.emb_num = emb_num
        self.window = window
        SinglePreArgMaskedPairSensor.__init__(self, pre, output_only=output_only)

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        device, _ = guess_device(context).most_common(1)[0]
        with torch.cuda.device(device):
            return super().forward(context)
