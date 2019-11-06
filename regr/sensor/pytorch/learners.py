from Graphs.Sensors.mainSensors import CallingSensor
import abc
from typing import Any
import torch
from .sensors import TorchSensor


class TorchLearner(TorchSensor):
    __metaclass__ = abc.ABCMeta

    def __init__(self, *pre):
        super(TorchLearner, self).__init__(*pre)
        self.model = None
        self.updated = False

    @property
    @abc.abstractmethod
    def parameters(self) -> Any:
        self.update_parameters()
        return self.model.parameters()

    def update_parameters(self):
        if not self.updated:
            for pre in self.pres:
                for name, learner in self.sup[pre].find(TorchLearner):
                    self.model.add_module(name=name, module=learner.model)
            self.updated = True

    def save(self, filepath):
        final_name = self.fullname.replace('/', '_')
        torch.save(self.model.state_dict(), filepath+"/"+final_name)

    def load(self, filepath):
        final_name = self.fullname.replace('/', '_')
        self.model.load_state_dict(torch.load(filepath+"/"+final_name))
        self.model.eval()
