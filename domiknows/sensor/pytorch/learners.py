import abc
from typing import Any
import os.path
import warnings

import torch

from .. import Learner
from .sensors import TorchSensor, FunctionalSensor, ModuleSensor
from .learnerModels import PyTorchFC, LSTMModel, PyTorchFCRelu


class TorchLearner(Learner, FunctionalSensor):
    __metaclass__ = abc.ABCMeta

    def __init__(self, *pre, edges=None, loss=None, metric=None, label=False, device='auto'):
        self.updated = False
        super(TorchLearner, self).__init__(*pre, edges=edges, label=label, device=device)
        self._loss = loss
        self._metric = metric

    @property
    def model(self):
        return None

    @property
    @abc.abstractmethod
    def parameters(self) -> Any:
        # self.update_parameters()
        if self.model is not None:
            return self.model.parameters()

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        if self.model is not None:
            self.parameters.to(device)
        self._device = device

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

    def loss(self, data_item, target):
        if self._loss is not None:
            pred = self(data_item)
            label = target(data_item)
            return self._loss(pred, label)

    def metric(self, data_item, target):
        if self._metric:
            pred = self(data_item)
            label = target(data_item)
            return self._metric(pred, label)


class ModuleLearner(ModuleSensor, TorchLearner):
    def __init__(self, *pres, module, edges=None, loss=None, metric=None, label=False, **kwargs):
        super().__init__(*pres, module=module, edges=edges, label=label, **kwargs)
        self._loss = loss
        self._metric = metric
        self.updated = True  # no need to update

    def update_parameters(self):
        pass


class LSTMLearner(TorchLearner):
    def __init__(self, *pres, input_dim, hidden_dim, num_layers=1, bidirectional=False, device='auto'):
        super(LSTMLearner, self).__init__(*pres, device=device)
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
    def __init__(self, *pres, input_dim, output_dim, device='auto'):
        super(FullyConnectedLearner, self).__init__(*pres, device=device)
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


class FullyConnectedLearnerRelu(TorchLearner):
    def __init__(self, *pres, input_dim, output_dim, device='auto'):
        super(FullyConnectedLearnerRelu, self).__init__(*pres, device=device)
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
