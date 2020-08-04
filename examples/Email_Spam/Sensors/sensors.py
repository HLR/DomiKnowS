from regr.sensor.pytorch.sensors import TorchSensor
import spacy
from typing import Any
import torch


class SentenceRepSensor(TorchSensor):
    def __init__(self, *pres, output=None, edges=None, label=False):
        super(SentenceRepSensor).__init__(*pres, output=None, edges=None, label=False)
        nlp = spacy.load('en_core_web_lg')

    def forward(self,) -> Any:
        email = self.nlp(self.inputs[0])
        return torch.from_numpy(email.vector)


class ForwardPresenceSensor(TorchSensor):
    def forward(self,) -> Any:
        if self.inputs[0]:
            return torch.ones(1)
        else:
            return torch.zeros(1)
