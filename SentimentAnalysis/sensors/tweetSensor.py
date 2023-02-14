from typing import Any

import spacy
import torch

from domiknows.sensor.pytorch.sensors import FunctionalSensor

class SentenceRepSensor(FunctionalSensor):
    def __init__(self, *pres, edges=None, label=False):
        super().__init__(*pres, edges=None, label=False)
        self.nlp = spacy.load('en_core_web_lg')

    def forward(self, text) -> Any:
        email = list(self.nlp.pipe(text))
        return torch.tensor([it.vector for it in email], device=self.device)
