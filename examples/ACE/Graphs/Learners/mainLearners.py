from Graphs.Sensors.mainSensors import CallingSensor
import abc
from typing import Any
import torch

class CallingLearner(CallingSensor):
    __metaclass__ = abc.ABCMeta

    def __init__(self, *pre):
        super(CallingLearner, self).__init__(*pre)
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
                for name, learner in pre.find(CallingLearner):
                    self.model.add_module(name=name, module=learner.model)
            self.updated = True

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath+"/"+self.name)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath+"/"+self.name))
        self.model.eval()