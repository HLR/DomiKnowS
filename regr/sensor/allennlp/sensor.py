from typing import List, Dict, Any, NoReturn
from collections import OrderedDict
import torch
from torch.nn import Module
from allennlp.modules.token_embedders import Embedding, PretrainedBertEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from ...graph import Property
from ...utils import prod, guess_device
from .base import ReaderSensor, ModuleSensor, SinglePreMaskedSensor, MaskedSensor, PreArgsModuleSensor, SinglePreArgMaskedPairSensor
from .module import Concat, SelfCartesianProduct, SelfCartesianProduct3, NGram, PairTokenDistance, PairTokenDependencyRelation, PairTokenDependencyDistance, LowestCommonAncestor, TripPhraseDistRelation, JointCandidate


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
        data_item: Dict[str, Any]
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

    
class LabelMaskSensor(LabelSensor):
    pass


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

    def get_mask(self, data_item: Dict[str, Any]):
        for pre in self.pres:
            for sensor in pre.find(MaskedSensor):
                return sensor.get_mask(data_item)
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

    def get_mask(self, data_item: Dict[str, Any]):
        for sensor in self.pre.find(MaskedSensor):
            break
        else:
            print(self.pre)
            raise RuntimeError('{} require at least one pre-required sensor to be MaskedSensor.'.format(self.fullname))

        mask = sensor.get_mask(data_item).float()
        ms = mask.size()
        mask = mask.view(ms[0], ms[1], 1).matmul(
            mask.view(ms[0], 1, ms[1]))  # (b,l,l)
        return mask


class CandidateSensor(MaskedSensor):
    pass

class CandidateReaderSensor(ReaderSensor, CandidateSensor):
    pass

class JointCandidateSensor(PreArgsModuleSensor, CandidateSensor):
    def create_module(self):
        return JointCandidate()

    @property
    def output_dim(self):
        return ()

    def get_mask(self, data_item: Dict[str, Any]):
        masks = []
        for pre in self.pres:
            for sensor in pre.find(MaskedSensor):
                break
            else:
                print(self.pre)
                raise RuntimeError('{} require at least one pre-required sensor to be MaskedSensor.'.format(self.fullname))
            # (b,l)
            mask = sensor.get_mask(data_item).float()
            masks.append(mask)

        masks_num = len(masks)
        mask = masks[0]
        for i in range(1, masks_num):
            for j in range(i, masks_num):
                masks[j].unsqueeze_(-2)
            mask = mask.unsqueeze_(-1).matmul(masks[i])
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

    def get_mask(self, data_item: Dict[str, Any]):
        for sensor in self.pre.find(MaskedSensor):
            break
        else:
            print(self.pre)
            raise RuntimeError('{} require at least one pre-required sensor to be MaskedSensor.'.format(self.fullname))

        mask = sensor.get_mask(data_item).float()
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

        for pre_sensor in pre.find(SentenceSensor):
            pre_sensor.add_embedder(key, self)
            self.tokens_key = pre_sensor.key # used by reader.update_textfield()
            break
        else:
            raise TypeError()

    def update_context(
        self,
        data_item: Dict[str, Any],
        force=False
    ) -> Dict[str, Any]:
        if self.fullname in data_item and isinstance(data_item[self.fullname], dict):
            data_item[self.fullname + '_index'] = data_item[self.fullname] # reserve
            force = True
        return SinglePreMaskedSensor.update_context(self, data_item, force)

    def forward(
        self,
        data_item: Dict[str, Any]
    ) -> Any:
        return self.module(data_item[self.fullname])

    def get_mask(self, data_item: Dict[str, Any]):
        # TODO: make sure update_context has been called
        return get_text_field_mask(data_item[self.fullname + '_index'])


class SentenceBertEmbedderSensor(SentenceEmbedderSensor):
    def create_module(self):
        self.embedding = PretrainedBertEmbedder(
            pretrained_model = self.pretrained_model,
        )
        return BasicTextFieldEmbedder({self.key: self.embedding},
                                      embedder_to_indexer_map={self.key: [
                                          self.key,
                                          "{}-offsets".format(self.key),
                                          "{}-type-ids".format(self.key)]},
                                      allow_unmatched_keys=True)

    def __init__(
        self,
        key: str,
        pretrained_model: str,
        pre,
        output_only: bool=False
    ) -> NoReturn:
        self.key = key
        self.pretrained_model = pretrained_model
        ModuleSensor.__init__(self, pre, output_only=output_only)

        for pre_sensor in pre.find(SentenceSensor):
            pre_sensor.add_embedder(key, self)
            self.tokens_key = pre_sensor.key # used by reader.update_textfield()
            break
        else:
            raise TypeError()


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
        data_item: Dict[str, Any]
    ) -> Any:
        device, _ = guess_device(data_item).most_common(1)[0]
        self.module.main_module.default_device = device
        return super().forward(data_item)


class TripPhraseDistSensor(SinglePreArgMaskedPairSensor):
    def create_module(self):
        return TripPhraseDistRelation()

    @property
    def output_dim(self):
        return (self.pre_dim[0] * 2,)

    def get_mask(self, data_item: Dict[str, Any]):
        for sensor in self.pre.find(MaskedSensor):
            break
        else:
            print(self.pre)
            raise RuntimeError('{} require at least one pre-required sensor to be MaskedSensor.'.format(self.fullname))

        mask = sensor.get_mask(data_item).float()
        ms = mask.size()
        #(b,l,l)
        mask1 = mask.view(ms[0], ms[1], 1).matmul(mask.view(ms[0], 1, ms[1]))
        mask2 = mask1.view(ms[0], ms[1], ms[1], 1).matmul(mask.view(ms[0], 1, 1, ms[1]))

        return mask2


class TokenDepSensor(SinglePreArgMaskedPairSensor):
    def create_module(self):
        return PairTokenDependencyRelation()

    def forward(
        self,
        data_item: Dict[str, Any]
    ) -> Any:
        #import pdb; pdb.set_trace()
        device, _ = guess_device(data_item).most_common(1)[0]
        self.module.main_module.default_device = device
        return super().forward(data_item)


class TokenLcaSensor(SinglePreArgMaskedPairSensor):
    def create_module(self):
        return LowestCommonAncestor()

    @property
    def output_dim(self):
        return self.pre_dims[1]

    def forward(
        self,
        data_item: Dict[str, Any]
    ) -> Any:
        return PreArgsModuleSensor.forward(self, data_item)


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
        self.window  = window
        SinglePreArgMaskedPairSensor.__init__(self, pre, output_only=output_only)

    def forward(
        self,
        data_item: Dict[str, Any]
    ) -> Any:
        device, _ = guess_device(data_item).most_common(1)[0]
        self.module.main_module.default_device = device
        return super().forward(data_item)
