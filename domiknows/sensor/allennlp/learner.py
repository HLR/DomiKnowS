from typing import List, Dict, Any, Union, NoReturn
from collections.abc import Iterable
import numpy as np
import torch
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from torch.nn import Module, Dropout, Sequential, Linear, ZeroPad2d, Conv2d, LogSoftmax, ReLU
from ...utils import prod
from ...graph import Property
from .base import AllenNlpLearner, SinglePreLearner, SinglePreMaskedLearner, MaskedSensor
from .sensor import SentenceSensor, SinglePreMaskedSensor, SentenceEmbedderSensor
from .module import DropoutRNN, Permute, Uncap


class SentenceEmbedderLearner(SentenceEmbedderSensor, AllenNlpLearner):
    def create_module(self):
        self.embedding = Embedding(
            num_embeddings=0, # later load or extend
            embedding_dim=self.embedding_dim,
            pretrained_file=self.pretrained_file,
            vocab_namespace=self.key,
            trainable=True,
        )
        return BasicTextFieldEmbedder({self.key: self.embedding})


class TripletEmbedderLearner(SentenceEmbedderLearner):
    def create_module(self):
        embedder = super().create_module()
        uncapper = Uncap(3, embedder)
        return uncapper

    @property
    def output_dim(self):
        return (self.embedding_dim,)

    def forward(
        self,
        data_item: Dict[str, Any]
    ) -> Any:
        return self.module(data_item[self.fullname], self.get_mask(data_item))

    def get_mask(self, data_item: Dict[str, Any]):
        en_mask = super().get_mask(data_item)
        #import pdb; pdb.set_trace()
        batch, en_len = en_mask.shape
        un_len = en_len ** (1. / 3)
        un_len = int(np.round(un_len))
        # (b, ul, ul, ul)
        un_mask = torch.zeros_like(en_mask).reshape((batch, un_len, un_len, un_len))
        # (b,)
        ul = en_mask.sum(dim=1).float() ** (1. / 3)
        ul = ul.round().int()
        for mm, ull in zip(un_mask, ul):
            mm[:ull,:ull,:ull] = 1
        return un_mask

class RNNLearner(SinglePreMaskedLearner):
    def create_module(self):
        return DropoutRNN(prod(self.pre_dim), self.layers, self.dropout, self.bidirectional)

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
        dims = list(self.dims) # convert or copy
        if isinstance(self.activation, list):
            activations = self.activation
        else:
            activations = [self.activation,] * len(dims)
        if isinstance(self.dropout, list):
            dropouts = self.dropout
        else:
            dropouts = [self.dropout,] * len(dims)

        dims.insert(0, prod(self.pre_dim))
        for i in range(len(dims) - 1):
            if dims[i + 1] is None:
                dims[i + 1] = dims[i]
        layers = []
        for dim_in, dim_out, activation, dropout in zip(dims[:-1], dims[1:], activations, dropouts):
            layers.append(Linear(in_features=dim_in, out_features=dim_out))
            if activation is not None:
                layers.append(activation)
            layers.append(Dropout(dropout))

        module = Sequential(*layers)
        return module

    @property
    def output_dim(self):
        dims = list(self.dims) # convert or copy
        dims.insert(0, prod(self.pre_dim))
        for i in range(len(dims) - 1):
            if dims[i + 1] is None:
                dims[i + 1] = dims[i]
        return (dims[-1],)

    def __init__(
        self,
        dims: List[int],
        pre: Property,
        activation: Union[Module, List[Module]]=ReLU(),
        dropout: Union[float, List[float]]=0.5,
        output_only: bool=False
    ) -> NoReturn:
        self.dims = dims
        self.activation = activation
        self.dropout = dropout
        SinglePreLearner.__init__(self, pre, output_only=output_only)


class ConvLearner(MLPLearner):
    def create_module(self):
        dims = list(self.dims) # convert or copy
        if isinstance(self.kernel_size, Iterable):
            kernel_sizes = self.kernel_size
        else:
            kernel_sizes = [self.kernel_size,] * len(dims)
        if isinstance(self.activation, Iterable):
            activations = self.activation
        else:
            activations = [self.activation,] * len(dims)
        if isinstance(self.dropout, Iterable):
            dropouts = self.dropout
        else:
            dropouts = [self.dropout,] * len(dims)

        dims.insert(0, prod(self.pre_dim))
        for i in range(len(dims) - 1):
            if dims[i + 1] is None:
                dims[i + 1] = dims[i]
        layers = []
        for dim_in, dim_out, kernel_size, activation, dropout in zip(dims[:-1], dims[1:], kernel_sizes, activations, dropouts):
            layers.append(Permute(0, 3, 1, 2)) # (b,l,l,c) -> (b,c,l,l)
            padding_t = padding_l = int(np.floor((float(kernel_size) - 1) / 2))
            padding_b = padding_r = int(np.ceil((float(kernel_size) - 1) / 2))
            layers.append(ZeroPad2d(padding=(padding_l, padding_r, padding_t, padding_b)))
            layers.append(Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=kernel_size))
            layers.append(Permute(0, 2, 3, 1)) # (b,c,l,l) -> (b,l,l,c)
            if activation is not None:
                layers.append(activation)
            layers.append(Dropout(dropout))

        module = Sequential(*layers)
        return module

    def __init__(
        self,
        dims: List[int],
        kernel_size: Union[int, List[int]],
        pre: Property,
        activation: Union[Module, List[Module]]=ReLU(),
        dropout: Union[float, List[float]]=0.5,
        output_only: bool=False
    ) -> NoReturn:
        self.dims = dims
        self.kernel_size = kernel_size
        self.activation = activation
        self.dropout = dropout
        SinglePreLearner.__init__(self, pre, output_only=output_only)


class LogisticRegressionLearner(SinglePreLearner, SinglePreMaskedSensor):
    def create_module(self):
        fc = Linear(in_features=prod(self.pre_dim), out_features=2)
        sm = LogSoftmax(dim=-1)
        module = Sequential(fc,
                            #sm,
                           )
        return module

    @property
    def output_dim(self):
        return (2,)

    def __init__(
        self,
        pre: Property,
        output_only: bool=False
    ) -> NoReturn:
        SinglePreLearner.__init__(self, pre, output_only=output_only)


class SoftmaxLogitLearner(SinglePreLearner, SinglePreMaskedSensor):
    def create_module(self):
        fc = Linear(in_features=prod(self.pre_dim), out_features=self.classes)
        sm = LogSoftmax(dim=-1)
        module = Sequential(fc,
                            #sm,
                           )
        return module

    @property
    def update_output_dim(self):
        return (self.classes,)

    def __init__(
        self,
        classes: int,
        pre: Property,
        output_only: bool=False
    ) -> NoReturn:
        self.classes = classes
        SinglePreLearner.__init__(self, pre, output_only=output_only)
