from regr.sensor.pytorch.sensors import TorchSensor, FunctionalSensor
import spacy
from typing import Any
import torch

class SentenceRepSensor(FunctionalSensor):
    def __init__(self, *pres, **kwarg):
        super().__init__(*pres, **kwarg)
        self.nlp = spacy.load('en_core_web_lg')

    def forward(self, *inputs) -> Any:
        email = self.nlp(inputs[0])
        return torch.from_numpy(email.vector).to(device=self.device).unsqueeze(0)


class ForwardPresenceSensor(FunctionalSensor):
    def forward(self, forward_body) -> Any:
        if forward_body:
            return torch.ones(1,1).to(self.device)
        else:
            return torch.zeros(1,1).to(self.device)
