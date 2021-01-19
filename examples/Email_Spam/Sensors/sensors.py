from regr.sensor.pytorch.sensors import FunctionalSensor
import spacy
from typing import Any
import torch


class SentenceRepSensor(FunctionalSensor):
    def __init__(self, *pres, edges=None, label=False):
        super().__init__(*pres, edges=None, label=False)
        self.nlp = spacy.load('en_core_web_sm')

    def forward(self, text) -> Any:
        email = list(self.nlp.pipe(text))
        return torch.tensor([it.vector for it in email], device=self.device)


class ForwardPresenceSensor(FunctionalSensor):
    def forward(self, forward_body) -> Any:
        if forward_body:
            return torch.ones(1).to(self.device)
        else:
            return torch.zeros(1).to(self.device)
