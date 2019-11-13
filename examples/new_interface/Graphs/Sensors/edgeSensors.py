from regr.sensor.pytorch.sensors import TorchEdgeSensor
from typing import Any


class FlairSentenceToWord(TorchEdgeSensor):
    def forward(self,) -> Any:
        return [token for token in self.inputs[0]]


class BILTransformer(TorchEdgeSensor):
    def forward(self,) -> Any:
        phrases = []
        start = None
        end = 0
        for token_it in range(len(self.inputs[1])):
            if not start and self.inputs[1][token_it] == "B":
                start = token_it
                end = token_it
            elif not start and self.inputs[1][token_it] == "L":
                phrases.append([token_it, token_it])
            elif start and self.inputs[1][token_it] == "L":
                phrases.append([start, token_it])
                start = None
                length = None
            elif start and self.inputs[1][token_it] == "I":
                end = token_it
        return phrases
