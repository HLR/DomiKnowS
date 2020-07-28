import abc
from typing import Any
import os.path
import warnings

import torch

from .sensors import TorchSensor, ModuleSensor
from .learnerModels import PyTorchFC, LSTMModel, PyTorchFCRelu



class TorchLearner(TorchSensor):
    __metaclass__ = abc.ABCMeta

    def __init__(self, *pre, edges=None, label=False):
        super(TorchLearner, self).__init__(*pre, edges=edges, label=label)
        self.model = None
        self.updated = False

    @property
    @abc.abstractmethod
    def parameters(self) -> Any:
        # self.update_parameters()
        return self.model.parameters()

    def update_parameters(self):
        if not self.updated:
            for pre in self.pres:
                for learner in self.sup.sup[pre].find(TorchLearner):
                    self.model.add_module(learner.name, module=learner.model)
            self.updated = True

    @property
    def sanitized_name(self):
        return self.fullname.replace('/', '_').replace("<","").replace(">","")

    def save(self, filepath):
        save_path = os.path.join(filepath, self.sanitized_name)
        torch.save(self.model.state_dict(), save_path)

    def load(self, filepath):
        save_path = os.path.join(filepath, self.sanitized_name)
        try:
            self.model.load_state_dict(torch.load(save_path))
            self.model.eval()
            self.model.train()
        except FileNotFoundError:
            message = f'Failed to load {self} from {save_path}. Continue not loaded.'
            warnings.warn(message)

class ModuleLearner(ModuleSensor, TorchLearner):
    def __init__(self, *pres, Module, edges=None, label=False, **kwargs):
        super().__init__(*pres, Module=Module, edges=edges, label=label, **kwargs)
        self.model = self.module
        self.updated = True  # no need to update

    def update_parameters(self):
        pass

class LSTMLearner(TorchLearner):
    def __init__(self, *pres, input_dim, hidden_dim, num_layers=1, bidirectional=False):
        super(LSTMLearner, self).__init__(*pres)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.input_dim = input_dim
        self.model = LSTMModel(input_dim=self.input_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers,
                               batch_size=1, bidirectional=self.bidirectional)
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.model.cuda()

    def forward(
            self,
    ) -> Any:
        value = self.inputs[0]
        if not torch.is_tensor(value):
            value = torch.stack(self.inputs[0])
        output = self.model(value)
        return output


class FullyConnectedLearner(TorchLearner):
    def __init__(self, *pres, input_dim, output_dim):
        super(FullyConnectedLearner, self).__init__(*pres)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = PyTorchFC(input_dim=self.input_dim, output_dim=self.output_dim)
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.model.cuda()

    def forward(
            self,
    ) -> Any:
        _tensor = self.inputs[0]
        output = self.model(_tensor)
        return output


class FullyConnected2Learner(TorchLearner):
    def __init__(self, *pres, input_dim, output_dim, edges=None):
        super(FullyConnected2Learner, self).__init__(*pres, edges=edges)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = torch.nn.Linear(self.input_dim, self.output_dim)
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.model.cuda()

    def forward(
            self,
    ) -> Any:
        _tensor = self.inputs[0]
        output = self.model(_tensor)
        return output


class FullyConnectedLearnerRelu(TorchLearner):
    def __init__(self, *pres, input_dim, output_dim):
        super(FullyConnectedLearnerRelu, self).__init__(*pres)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = PyTorchFCRelu(input_dim=self.input_dim, output_dim=self.output_dim)
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.model.cuda()

    def forward(
            self,
    ) -> Any:
        _tensor = self.inputs[0]
        output = self.model(_tensor)
        return output