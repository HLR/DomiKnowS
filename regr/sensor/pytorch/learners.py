import abc
from typing import Any
import torch
from .sensors import TorchSensor
from .learnerModels import PyTorchFC, LSTMModel, PyTorchFCRelu
import os.path
from os import path

class TorchLearner(TorchSensor):
    __metaclass__ = abc.ABCMeta

    def __init__(self, *pre):
        super(TorchLearner, self).__init__(*pre)
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
                for name, learner in self.sup.sup[pre].find(TorchLearner):
                    self.model.add_module(name=name, module=learner.model)
            self.updated = True

    def save(self, filepath):
        #final_name = self.fullname.replace('/', '_')
        final_name = self.fullname.replace('/', '_').replace("<","").replace(">","")
        torch.save(self.model.state_dict(), filepath+"/"+final_name)

    def load(self, filepath):
        #final_name = self.fullname.replace('/', '_')
        final_name = self.fullname.replace('/', '_').replace("<","").replace(">","")
        if path.exists(filepath+"/"+final_name):
            self.model.load_state_dict(torch.load(filepath+"/"+final_name))
            self.model.eval()
            self.model.train()


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