import torch

from regr.sensor.learner import Learner
from emr.sensor.sensor import TorchSensor, FunctionalSensor


class TorchLearner(TorchSensor, Learner):
    pass

class FunctionalLearner(FunctionalSensor, TorchLearner):
    pass


class ModuleLearner(FunctionalLearner):
    @staticmethod
    def Module():
        pass

    def __init__(self, pre, target=False, module=None, **kwargs):
        super().__init__(pre, target=target)
        self.module = module or self.Module(**kwargs)

    def forward(self, input):
        return self.module(input)


class EmbedderLearner(ModuleLearner):
    Module = torch.nn.Embedding


class RNNLearner(ModuleLearner):
    Module=torch.nn.LSTM

    def forward(self, input):
        output, _ = self.module(input)
        return output


class MLPLearner(ModuleLearner):
    Module = torch.nn.Linear


class LRLearner(ModuleLearner):
    @staticmethod
    def Module(**kwargs):
        kwargs['out_features'] = 1
        return torch.nn.Linear(**kwargs)
