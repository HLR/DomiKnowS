import numpy as np
import torch
from torch.nn import Module, GRU, Dropout
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from ...utils import find_base
import itertools


class WrapperModule(Module):
    # use a wrapper to keep pre-requireds and avoid side-effect of sequencial or other modules
    def __init__(self, module, pres):
        Module.__init__(self)
        self.main_module = module
        for pre in pres:
            from .base import ModuleSensor
            for sensor in pre.find(ModuleSensor, lambda s: not s.output_only):
                self.add_module(sensor.fullname, sensor.module)

    def forward(self, *args, **kwargs):
        return self.main_module(*args, **kwargs)

    def get_output_dim(self):
        return self.main_module.get_output_dim()


class BaseModule(Module):
    def __init__(self, output_dim):
        Module.__init__(self)
        self.output_dim = output_dim
        self.default_device = None

    def get_output_dim(self):
        return self.output_dim


class DropoutRNN(BaseModule):
    def __init__(self, embedding_dim, layers=1, dropout=0.5, bidirectional=True):
        output_dim = embedding_dim
        if bidirectional:
            output_dim *= 2
        BaseModule.__init__(self, output_dim)

        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional

        self.rnn = PytorchSeq2SeqWrapper(GRU(embedding_dim,
                                             embedding_dim,
                                             num_layers=layers,
                                             batch_first=True,
                                             dropout=dropout,
                                             bidirectional=bidirectional))
        self.dropout = Dropout(dropout)

    def forward(self, x, mask):
        return self.dropout(self.rnn(x, mask))


class Concat(Module):
    def forward(self, *x):
        # TODO: flatten
        return torch.cat(x, dim=-1)


class CartesianProduct(Module):
    def forward(self, x, y):  # (b,l1,f1) x (b,l2,f2) -> (b, l1, l2, f1+f2)
        # TODO: flatten
        xs = x.size()
        ys = y.size()
        assert xs[0] == ys[0]
        # torch cat is not broadcasting, do repeat manually
        xx = x.view(xs[0], xs[1], 1, xs[2]).repeat(1, 1, ys[1], 1)
        yy = y.view(ys[0], 1, ys[1], ys[2]).repeat(1, xs[1], 1, 1)
        return torch.cat([xx, yy], dim=3)


class SelfCartesianProduct(CartesianProduct):
    def forward(self, x):
        return CartesianProduct.forward(self, x, x)


class CartesianProduct3(Module):
    # (b,l1,f1) x (b,l2,f2) -> (b, l1, l2, l3, f1+f2+f3)
    def forward(self, x, y, z):
        # TODO: flatten
        xs = x.size()
        ys = y.size()
        zs = z.size()
        assert xs[0] == ys[0] and ys[0] == zs[0]
        # torch cat is not broadcasting, do repeat manually
        xx = x.view(xs[0], xs[1], 1, 1, xs[2]).repeat(1, 1, ys[1], zs[1], 1)
        yy = y.view(ys[0], 1, ys[1], 1, ys[2]).repeat(1, xs[1], 1, zs[1], 1)
        zz = z.view(ys[0], 1, 1, ys[1], zs[2]).repeat(1, xs[1], ys[1], 1, 1)
        return torch.cat([xx, yy, zz], dim=-1)


class SelfCartesianProduct3(CartesianProduct3):
    def forward(self, x):
        return CartesianProduct3.forward(self, x, x, x)


class JointOldCandidate(Module):
    def forward(self, *tensors):
        #import pdb; pdb.set_trace()
        batch = tensors[0].shape[0]
        num = len(tensors)
        # b, l, c
        logits_sum = tensors[0].clone()
        # n, b, l, c
        addends = [logits.clone() for logits in tensors[1:]]
        for i, logits in enumerate(addends):
            for tensor in addends[i:]:
                # b, ..., 1, l, c
                tensor.unsqueeze_(-3)
            # b, l, ..., l, c = b, ..., l, 1, c + b, ..., 1, l, c 
            logits_sum = logits_sum.unsqueeze_(-2) + logits # this are the logits
        # b, l, ..., l ~ {0, 1}
        candidate = logits_sum.argmax(-1) # find the largest index along logits dim
        # remove self pair
        # b, l, ..., l ~ {0}
        candidate_2 = torch.zeros_like(candidate)[0]
        for indices in itertools.product(*(range(d) for d in candidate.shape[1:])):
            if len(indices) == len(set(indices)):
                # if no same index
                candidate_2[indices] = 1
        # b, l, ..., l ~ {0, 1}
        candidate_2 = candidate_2.expand(batch, *((-1,) * num))
        return candidate * candidate_2


class JointCandidate(Module):
    def forward(self, *tensors):
        #import pdb; pdb.set_trace()
        batch = tensors[0].shape[0]
        num = len(tensors)
        # b, l ~ {0, 1}
        candidate_sum = tensors[0].clone().argmax(-1)
        # n-1, b, l ~ {0, 1}
        addends = [logits.clone().argmax(-1) for logits in tensors[1:]]
        for i, candidate_tensor in enumerate(addends):
            for tensor in addends[i:]:
                # b, ..., 1, l ~ {0, 1}
                tensor.unsqueeze_(-2)
            # b, l, ..., l = b, ..., l, 1 + b, ..., 1, l
            candidate_sum = candidate_sum.unsqueeze_(-1) * candidate_tensor
        # b, l, ..., l ~ {0, 1}
        candidate = candidate_sum
        # remove self pair
        # b, l, ..., l ~ {0}
        candidate_2 = torch.zeros_like(candidate)[0]
        for indices in itertools.product(*(range(d) for d in candidate.shape[1:])):
            if len(indices) == len(set(indices)):
                # if no same index
                candidate_2[indices] = 1
        # b, l, ..., l ~ {0, 1}
        candidate_2 = candidate_2.expand(batch, *((-1,) * num))
        return candidate * candidate_2


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


class PairTokenDistance(BaseModule):
    def __init__(self, emb_num, window):
        BaseModule.__init__(self, emb_num)
        self.emb_num = emb_num # must define emb_num (to have lb and ub) before window
        self.window = window

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
        dist = torch.arange(-length + 1, length, device=self.default_device)
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


class PairTokenDependencyRelation(BaseModule):
    def __init__(self):
        BaseModule.__init__(self, 6)

    def forward(self, x):
        #import pdb; pdb.set_trace()
        batch = len(x)
        length = max(len(xx) for xx in x)
        #(b,l,l,f)
        dep = torch.zeros((batch, length, length, self.get_output_dim()), device=self.default_device)
        for i, span in enumerate(x):
            for j, token in enumerate(span):
                # children: immediate syntactic children
                for r_token in token.children:
                    dep[i, j, r_token.i, 0] = 1
                # lefts: leftward immediate children
                for r_token in token.lefts:
                    dep[i, j, r_token.i, 1] = 1
                # rights: rightward immediate children
                for r_token in token.rights:
                    dep[i, j, r_token.i, 2] = 1
                # conjuncts: coordinated tokens
                for r_token in token.conjuncts:
                    dep[i, j, r_token.i, 3] = 1
                # head: syntactic parent, or "governor"
                dep[i, j, token.head.i, 4] = 1
                # ancestors: rightmost token of this toke's syntactic descendants
                for r_token in token.ancestors:
                    dep[i, j, r_token.i, 5] = 1

        return dep


class TripPhraseDistRelation(Module):
    def forward(self, x):
        #import pdb; pdb.set_trace()
        batch, length, feat = x.shape
        device = x.device
        dist = torch.zeros(batch, length, length, length, feat * 2, device=device)

        permutations = itertools.permutations(range(0, length), 3)
        # permutations ensure no repeat index at the same time
        for lm, tr, sp in permutations:
            # (batch, lm:sp, feat)
            lm2sp = x[:, min(lm, sp):max(lm, sp), :]
            # (batch, tr:sp, feat)
            tr2sp = x[:, min(tr, sp):max(tr, sp), :]
            # (batch, feat)
            dist[:, lm, tr, sp, :] = torch.cat((lm2sp.mean(dim=1), tr2sp.mean(dim=1)), dim=1)

        return dist


class LowestCommonAncestor(Module):
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


class PairTokenDependencyDistance(BaseModule):
    def __init__(self, emb_num, window):
        BaseModule.__init__(self, emb_num * 2)

        self.emb_num = emb_num
        self.window = window
        #self.emb_dim = 64
        #self.embedding = torch.nn.Embedding(self.emb_num, self.emb_dim)

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, window):
        self._window = window
        # leave one for zero-distance, making one-distance another unique (b**0===1 <- b!=0)
        self.base = find_base(window, self.emb_num - 1)

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
        lcas = torch.zeros((batch, length, length), device=self.default_device, dtype=torch.long)

        for doc, lca in zip(docs, lcas):
            # (l,l) ~ [-1, l-1]
            lca_np = doc.get_lca_matrix()
            # (lx,lx) ~ [-1, l-1]
            lca[:lca_np.shape[0], :lca_np.shape[0]] = torch.as_tensor(lca_np)
        lcas = lcas.float()
        # (b,lx,lx) ~ [0, l-1] U {inf}
        lcas[lcas==-1] = torch.as_tensor(np.inf)
        # (lx,)
        index = torch.arange(length, device=lcas.device, dtype=torch.float)
        # (b,lx,lx) = (b,lx,lx) - (lx,1) ~ [0, l-1] U {inf}
        right_dist = (lcas - index).abs()

        # (b,lx,lx) ~ [0, emb_num-2] U {-inf, inf} + 1 ~ [1, emb_num-1] U {-inf, inf}
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


class Permute(Module):
    def __init__(self, *dims):
        Module.__init__(self)
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


class Uncap(Module):
    def __init__(self, dim, pre_module=None):
        Module.__init__(self)
        self.dim = dim
        self.pre_module = pre_module

    def forward(self, x, mask):
        if self.pre_module:
            x = self.pre_module(x)

        #import pdb; pdb.set_trace()
        batch, en_len, feat = x.shape
        device = x.device
        # note that mask is generated to match this module already

        # estimate ul
        un_len = en_len ** (1. / self.dim)
        # should always be a natural number if dim is matched as input config
        # could be some slight different due to numerical stability
        un_len = int(np.round(un_len))

        dist = torch.zeros_like(x).reshape((batch, *((un_len,) * self.dim), feat))
        # process each differnt length base on mask
        for dd, xx, mm in zip(dist, x, mask):
            # dd - (ul..., f)
            # xx - (el, f)
            # mm - (ul...,)

            # ()
            ull = mm.sum().float() ** (1. / 3)
            ull = ull.round().int()

            # convert ijk index to 1-d array index
            def encap_index(*indices):
                return sum(i * un_len ** j for j, i in enumerate(reversed(indices)))

            # headfirst solution
            # iter through all and move only the last dim together
            for indices in itertools.product(range(ull), repeat=self.dim-1):
                dd[indices] = xx[encap_index(*indices)]

            # more efficient solution:
            # try torch.unfold or torch.as_strided

        return dist
