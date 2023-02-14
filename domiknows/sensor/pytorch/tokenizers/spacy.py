from domiknows.sensor.pytorch.relation_sensors import EdgeSensor
import torch
from typing import Any


class SpacyTokenizer(EdgeSensor):
    def __init__(self, *pres, relation, edges=None, label=False, device='auto', spacy=None):
        super().__init__(*pres, relation=relation, edges=edges, label=label, device=device)
        if not spacy:
            raise ValueError('You should select a default Spacy Pipeline')
        self.nlp = spacy

    def forward(self,) -> Any:
        text = self.nlp(self.inputs[0])
        return [token for token in text]
