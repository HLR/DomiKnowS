from regr.sensor.pytorch.sensors import TorchEdgeSensor
from typing import Any
import torch


class FlairSentenceToWord(TorchEdgeSensor):
    def forward(self,) -> Any:
        return [token for token in self.inputs[0]]


class BILTransformer(TorchEdgeSensor):
    def forward(self,) -> Any:
        phrases = []
        start = -1
        for token_it in range(len(self.inputs[1])):
            with torch.no_grad():
                values, indices = self.inputs[1][token_it].max(0)
            if start == -1 and indices == 0:
                start = token_it
            elif start == -1 and indices == 2:
                phrases.append([token_it, token_it])
            elif start != -1 and indices == 2:
                phrases.append([start, token_it])
                start = -1
            elif start != -1 and indices == 1:
                pass
        return phrases
