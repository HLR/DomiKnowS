from typing import Any

import spacy
import torch

from regr.sensor.pytorch.sensors import TorchSensor


class SentenceRepSensor(TorchSensor):
    def __init__(self, *pres, edges=None, label=False):
        super().__init__(*pres, edges=None, label=False)
        self.nlp = spacy.load('en_core_web_sm')
    def forward(self,) -> Any:
        email = self.nlp(self.inputs[0])
        return torch.from_numpy(email.vector)