from typing import List, Dict, Any, NoReturn
from collections import OrderedDict
import torch
from torch.nn import Module
import numpy as np
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from ...utils import prod, guess_device, find_base
from ...graph import Property
from .base import ReaderSensor, ModuleSensor, SinglePreMaskedSensor, MaskedSensor, PreArgsModuleSensor, SinglePreArgMaskedPairSensor


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
    class Concat(Module):
        def forward(self, *x):
            # TODO: flatten
            return torch.cat(x, dim=-1)

    def create_module(self):
        return ConcatSensor.Concat()

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

    @property
    def output_dim(self):
        if len(self.pre_dim) == 0:
            output_dim = 2
        else:
            output_dim = prod(self.pre_dim) * 2 # assume flatten
        return (output_dim,)


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
    class NGram(Module):
        def __init__(self, ngram):
            Module.__init__(self)
            self.ngram = ngram

        def forward(self, x):
            #import pdb; pdb.set_trace()
            shifted = []
            size = x.size() # (b, l, c)
            for i in torch.arange(self.ngram):
                shifted_x = torch.zeros((size[0], size[1]+self.ngram, size[2]), device=x.device)
                shifted_x[:, i:i-self.ngram, :] = x
                shifted.append(shifted_x)
            new_x = torch.cat(shifted, dim=-1)
            offset = int((self.ngram-1) / 2)
            return new_x[:, offset:offset-self.ngram, :]

    def create_module(self):
        return NGramSensor.NGram(self.ngram)

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
    class Dist(Module):
        def __init__(self, emb_num, window):
            Module.__init__(self)
            self.emb_num = emb_num # must define emb_num (to have lb and ub) before window
            self.window = window

        def get_output_dim(self):
            return self.emb_num

        @property
        def window(self):
            return self._window

        @window.setter
        def window(self, window):
            self._window = window
            ul = np.floor(window / 2)
            self._base = find_base(ul, self.ub - 1)

        @property
        def base(self):
            return self._base

        @property
        def emb_num(self):
            return self._emb_num

        @emb_num.setter
        def emb_num(self, emb_num):
            self._emb_num = emb_num
            self._lb = -np.floor((emb_num - 1) / 2)
            self._ub = np.ceil((emb_num - 1) / 2)

        @property
        def lb(self):
            return self._lb

        @property
        def ub(self):
            return self._ub

        def forward(self, x):
            batch = len(x)
            length = max(len(xx) for xx in x)
            #(l*2)
            dist = torch.arange(-length + 1, length, device=torch.cuda.current_device())
            rows = []
            for i in range(length):
                rows.append(dist[i:i + length].view(1, -1))
            #(l, l)
            dist = torch.cat(tuple(reversed(rows)))
            #print(dist)
            sign = dist.sign()
            dist = dist.abs()
            dist = dist.float()
            dist = (dist.log() / np.log(self.base) + 1).floor()
            #print(dist)
            dist[dist < 0] = 0
            dist = dist * sign.to(dtype=dist.dtype, device=dist.device)
            dist[dist < self.lb] = self.lb
            dist[dist > self.ub] = self.ub
            dist = dist - self.lb
            dist = dist.long()
            #print(dist)
            #(n, n)
            eye = torch.eye(self.emb_num, device=dist.device)
            #import pdb; pdb.set_trace()
            #(1, l, l, n)
            dist = eye.index_select(0, dist.view(-1)).view(1, length, length, -1)
            #(b, l, l, n)
            dist = dist.repeat(batch, 1, 1, 1)
            return dist

    def create_module(self):
        return TokenDistantSensor.Dist(self.emb_num, self.window)

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
    class Dep(Module):
        def get_output_dim(self):
            return 6

        def forward(self, x):
            #import pdb; pdb.set_trace()
            batch = len(x)
            length = max(len(xx) for xx in x)
            #(b,l,l,f)
            dep = torch.zeros((batch, length, length, self.get_output_dim()), device=torch.cuda.current_device())
            for i, span in enumerate(x):
                for j, token in enumerate(span):
                    # children
                    for r_token in token.children:
                        dep[i, j, r_token.i, 0] = 1
                    # lefts
                    for r_token in token.lefts:
                        dep[i, j, r_token.i, 1] = 1
                    # rights
                    for r_token in token.rights:
                        dep[i, j, r_token.i, 2] = 1
                    # conjuncts
                    for r_token in token.conjuncts:
                        dep[i, j, r_token.i, 3] = 1
                    # conjuncts
                    for r_token in token.conjuncts:
                        dep[i, j, r_token.i, 4] = 1
                    # ancestors
                    for r_token in token.ancestors:
                        dep[i, j, r_token.i, 5] = 1

            return dep

    def create_module(self):
        return TokenDepSensor.Dep()

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        #import pdb; pdb.set_trace()
        device, _ = guess_device(context).most_common(1)[0]
        with torch.cuda.device(device):
            return super().forward(context)


class TokenLcaSensor(SinglePreArgMaskedPairSensor):
    class LCA(Module):
        def forward(self, token_lists, features):
            #import pdb; pdb.set_trace()
            batch = features.shape[0]
            length = features.shape[1]
            feat = features.shape[2]

            docs = []
            for token_list in token_lists:
                doc = None
                for token in token_list:
                    if doc is None:
                        doc = token.doc
                    else:
                        assert doc == token.doc
                docs.append(doc)

            # (b,lx,lx)
            lcas = torch.zeros((batch, length, length), device=features.device, dtype=torch.long)
            for doc, lca in zip(docs, lcas):
                # (l,l) ~ [-1, l-1]
                lca_np = doc.get_lca_matrix()
                # (lx,lx) ~ [0, l]
                lca[:lca_np.shape[0], :lca_np.shape[0]] = torch.as_tensor(lca_np) + 1
            # (b, lx+1, f)
            features = torch.cat((torch.zeros((batch, 1, feat), device=features.device), features), dim=1)
            # (b, lx*lx)
            lcas = lcas.view(batch, length * length)
            lcas_emb = []
            for feature, lca in zip(features, lcas):
                # [(1, lx*lx, f)...]
                lcas_emb.append(feature.index_select(0, lca).view(1, length * length, feat))
            # (b, lx, lx, f)
            lcas_emb = torch.cat(lcas_emb, dim=0).view(batch, length, length, feat)

            return lcas_emb

    def create_module(self):
        return TokenLcaSensor.LCA()

    @property
    def output_dim(self):
        return self.pre_dims[1]

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        return PreArgsModuleSensor.forward(self, context)


class TokenDepDistSensor(SinglePreArgMaskedPairSensor):
    class DepDist(Module):
        def __init__(self, emb_num):
            Module.__init__(self)
            self.emb_num = emb_num
            #self.emb_dim = 64
            self.window = 128
            # leave one for zero-distance, making one-distance another unique (b**0===1 <- b!=0)
            self.base = find_base(self.window, self.emb_num - 1)
            #self.embedding = torch.nn.Embedding(self.emb_num, self.emb_dim)

        def get_output_dim(self):
            return self.emb_num * 2

        def forward(self, token_lists):
            #import pdb; pdb.set_trace()
            batch = len(token_lists)
            length = max(len(token_list) for token_list in token_lists)

            docs = []
            for token_list in token_lists:
                doc = None
                for token in token_list:
                    if doc is None:
                        doc = token.doc
                    else:
                        assert doc == token.doc
                docs.append(doc)

            # (b,lx,lx)
            lcas = torch.zeros((batch, length, length), device=torch.cuda.current_device(), dtype=torch.long)
            for doc, lca in zip(docs, lcas):
                # (l,l) ~ [-1, l-1]
                lca_np = doc.get_lca_matrix()
                # (lx,lx) ~ [-1, l-1]
                lca[:lca_np.shape[0], :lca_np.shape[0]] = torch.as_tensor(lca_np)
            # (b,lx,lx) ~ [0, l-1] U {inf}
            lcas[lcas==-1] = torch.as_tensor(np.inf)
            # (lx,)
            index = torch.arange(length, device=lcas.device)
            # (b,lx,lx) = (b,lx,lx) - (lx,1)
            right_dist = (lcas - index).abs()

            right_dist = right_dist.float()
            # (b,lx,lx) ~ [0, emb_num-2] U {-inf} + 1 ~ [1, emb_num-1] U {-inf}
            right_dist = (right_dist.log() / np.log(self.base) + 1).floor()
            # (b,lx,lx) ~ [0, emb_num-1]
            right_dist[right_dist < 0] = 0
            right_dist[right_dist > self.emb_num - 1] = self.emb_num - 1
            right_dist = right_dist.long()

            #left_dist = right_dist.transpose(1, 2)

            # (lx, lx)
            eye = torch.eye(self.emb_num, device=lcas.device)
            # (b,lx,lx, emb_num)
            #left_oh = eye.index_select(0, left_dist.view(-1)).view(batch, length, length, -1)
            right_oh = eye.index_select(0, right_dist.view(-1)).view(batch, length, length, -1)
            left_oh = right_oh.transpose(1, 2)

            # (b,lx,lx,emb_dim)
            #left_emb = self.embedding(left_dist)
            # (b,lx,lx,emb_dim)
            #right_emb = self.embedding(right_dist)

            # (b,lx,lx, emb_num*2)
            dist = torch.cat((left_oh, right_oh), dim=-1)
            return dist

    def create_module(self):
        return TokenDepDistSensor.DepDist(self.emb_num)

    def __init__(
        self,
        emb_num: int,
        pre: Property,
        output_only: bool=False
    ) -> NoReturn:
        self.emb_num = emb_num
        SinglePreArgMaskedPairSensor.__init__(self, pre, output_only=output_only)

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        device, _ = guess_device(context).most_common(1)[0]
        with torch.cuda.device(device):
            return super().forward(context)
