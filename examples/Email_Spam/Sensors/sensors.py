from regr.sensor.pytorch.sensors import TorchSensor
import spacy
from typing import Any
import torch


class SentenceRepSensor(TorchSensor):
    def __init__(self, *pres, edges=None, label=False):
        super().__init__(*pres, edges=None, label=False)
        self.nlp = spacy.load('en_core_web_sm')

    def forward(self,) -> Any:
        email = self.nlp(self.inputs[0])
        return torch.from_numpy(email.vector).to(device=self.device)


class ForwardPresenceSensor(TorchSensor):
    def forward(self,) -> Any:
        if self.inputs[0]:
            return torch.ones(1).to(self.device)
        else:
            return torch.zeros(1).to(self.device)
