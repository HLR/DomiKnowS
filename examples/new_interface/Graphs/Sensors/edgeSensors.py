from regr.sensor.pytorch.sensors import TorchEdgeSensor
from typing import Any


class FlairSentenceToWord(TorchEdgeSensor):
    def forward(self,) -> Any:
        return [token for token in self.inputs[0]]
