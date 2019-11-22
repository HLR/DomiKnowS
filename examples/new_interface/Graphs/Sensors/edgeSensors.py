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
        if phrases:
            return phrases
        else:
            return [[0, len(self.inputs[1])]]


class WordToPhraseTransformer(TorchEdgeSensor):
    def forward(self,) -> Any:
        value = torch.cat([item for item in self.inputs], -1)
        indexes = []
        for item in value:
            _, index = torch.max(item, 0)
            indexes.append(index)
        print(indexes)
        phrases = []
        start = -1
        previous = -1
        _index = 0
        for _index in range(len(indexes)):
            if start == -1 and indexes[_index] % 2 == 0:
                start = _index
                previous = indexes[_index]
            elif start != -1:
                if indexes[_index] != previous:
                    phrases.append([start, _index - 1])
                    if indexes[_index] % 2 == 0:
                        start = _index
                        previous = indexes[_index]
                    else:
                        start = -1
                        previous = -1
        if start != -1 and indexes[_index] % 2 == 0:
            phrases.append([start, _index])


