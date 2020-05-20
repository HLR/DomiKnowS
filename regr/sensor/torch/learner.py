import torch

from .. import Learner
from .sensor import TorchSensor, FunctionalSensor


class TorchLearner(TorchSensor, Learner):
    pass

class FunctionalLearner(FunctionalSensor, TorchLearner):
    pass


class ModuleLearner(FunctionalLearner):
    @staticmethod
    def Module(**kwargs):
        pass

    def __init__(self, pre, target=False, module=None, **kwargs):
        super().__init__(pre, target=target)
        self.module = module or self.Module(**kwargs)

    def forward_func(self, input):
        return self.module(input)

    def parameters(self):
        return self.module.parameters()


class EmbedderLearner(ModuleLearner):
    Module = torch.nn.Embedding

    def mask(self, context):
        input = next(self.get_args(context, sensor_filter=lambda s: not s.target))
        if self.module.padding_idx is not None:
            mask = input.clone().detach() != self.module.padding_idx
        else:
            mask = torch.ones_like(input)
        return mask

class NorminalEmbedderLearner(EmbedderLearner):
    def __init__(self, pre, vocab_key=None, num_embeddings=None, target=False, module=None, **kwargs):
        super().__init__(pre, target=target)
        self.vocab_key = vocab_key
        if num_embeddings is None:
            self.module = module or self.Module(num_embeddings=1, **kwargs)
            self.modele_args = kwargs
        else:
            self.module = module or self.Module(**kwargs)
            self.modele_args = kwargs

    def update_module(self, num_embeddings):
        self.module = self.Module(num_embeddings=num_embeddings, **self.modele_args)

    def mask(self, context):
        input = next(self.get_args(context, sensor_filter=lambda s: not s.target))
        if self.module.padding_idx is not None:
            mask = input.clone().detach() != self.module.padding_idx
        else:
            mask = torch.ones_like(input)
        return mask

class RNNLearner(ModuleLearner):
    Module=torch.nn.LSTM

    def forward_func(self, input):
        output, _ = super().forward_func(input)
        return output


class MLPLearner(ModuleLearner):
    Module = torch.nn.Linear


class LRLearner(ModuleLearner):
    @staticmethod
    def Module(**kwargs):
        kwargs['out_features'] = 1
        return torch.nn.Linear(**kwargs)

    def forward_func(self, input):
        return super().forward_func(input).squeeze(-1)
