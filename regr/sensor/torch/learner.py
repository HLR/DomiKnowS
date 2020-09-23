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

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.module.to(*args, **kwargs)


class EmbedderLearner(ModuleLearner):
    Module = torch.nn.Embedding

    def mask(self, data_item):
        input = next(self.get_args(data_item, sensor_filter=lambda s: not s.target))
        if self.module.padding_idx is not None:
            mask = input.clone().detach() != self.module.padding_idx
        else:
            mask = torch.ones_like(input).to(*self._to_args, **self._to_kwargs)
        return mask

class NorminalEmbedderLearner(EmbedderLearner):
    def __init__(self, pre, vocab_key=None, num_embeddings=None, target=False, module=None, **kwargs):
        self.vocab_key = vocab_key
        self.modele_args = kwargs
        if num_embeddings is None:
            super().__init__(pre, num_embeddings=2, padding_idx=0, **kwargs, target=target)
        else:
            super().__init__(pre, num_embeddings=num_embeddings, padding_idx=0, **kwargs, target=target)

    def update_module(self, num_embeddings):
        self.module = self.Module(num_embeddings=num_embeddings, **self.modele_args)

    def mask(self, data_item):
        input = next(self.get_args(data_item, sensor_filter=lambda s: not s.target))
        if self.module.padding_idx is not None:
            mask = input.clone().detach() != self.module.padding_idx
        else:
            mask = torch.ones_like(input).to(*self._to_args, **self._to_kwargs)
        return mask

    def forward_func(self, input):
        _, indexes, *_ = input
        if isinstance(indexes, list):
            max_len = max(map(len, indexes))
            def pad(output):
                return torch.cat((
                    torch.tensor(output, dtype=torch.long), 
                    torch.zeros(max_len - len(output), dtype=torch.long).to(*self._to_args, **self._to_kwargs)
                    ))
            indexes = torch.stack(tuple(map(pad, indexes)))
        indexes = indexes.to(*self._to_args, **self._to_kwargs).to(dtype=torch.long)
        return super().forward_func(indexes)

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
