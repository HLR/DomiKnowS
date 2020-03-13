import torch

from regr.sensor.learner import Learner
from emr.sensor.sensor import TorchSensor, FunctionalSensor


class TorchLearner(TorchSensor, Learner):
    pass

class FunctionalLearner(FunctionalSensor, TorchLearner):
    pass

class EmbedderLearner(FunctionalLearner):
    def __init__(self, pre, output_only=False, **kwargs):
        super().__init__(pre, output_only=output_only)
        self.module = torch.nn.Embedding(**kwargs)

    def forward(self, input):
        return self.module(input)

class RNNLearner(FunctionalLearner):
    def __init__(self, pre, output_only=False, **kwargs):
        super().__init__(pre, output_only=output_only)
        self.module = torch.nn.LSTM(**kwargs)

    def forward(self, input):
        output, _ = self.module(input)
        return output


class MLPLearner(FunctionalLearner):
    def __init__(self, pre, output_only=False, **kwargs):
        super().__init__(pre, output_only=output_only)
        self.module = torch.nn.Linear(**kwargs)

    def forward(self, input):
        return self.module(input)


class LRLearner(FunctionalLearner):
    def __init__(self, pre, output_only=False, **kwargs):
        super().__init__(pre, output_only=output_only)
        self.module = torch.nn.Linear(**kwargs, out_features=1)

    def forward(self, input):
        return self.module(input)
