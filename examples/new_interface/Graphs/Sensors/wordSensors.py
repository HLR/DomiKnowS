from regr.sensor.pytorch.sensors import TorchSensor
from typing import Any


class WordEmbedding(TorchSensor):
    def forward(self,) -> Any:
        return [word.embedding.view(1, 5220) for word in self.inputs[0]]
