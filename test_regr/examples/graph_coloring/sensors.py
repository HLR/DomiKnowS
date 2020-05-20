from regr.sensor.pytorch.sensors import TorchSensor, TorchEdgeSensor, ReaderSensor
from regr.sensor.pytorch.learners import TorchLearner
from typing import Any
import torch


class DummyLearner(TorchLearner):
    def forward(self,) -> Any:
        result = torch.ones(len(self.inputs[0]))
        result = -1 * result
        return result


class DummyLabelSensor(TorchSensor):
    def __init__(self, *pres, label=True):
        super().__init__(*pres, label=label)

    def forward(self,) -> Any:
        return []


class DummyEdgeSensor(TorchEdgeSensor):
    def forward(self,) -> Any:
        return self.inputs[0]


class DummyNeighborgenerator(TorchEdgeSensor):
    def forward(self,) -> Any:
        return self.inputs[0]


class DummyNeighbor(TorchSensor):
    def forward(self,) -> Any:
        pairs = []
        print("start of the dummyNeighbor sensor")
        for item in self.inputs[0]:
            for item1 in self.inputs[1]:
                pairs.append((item, item1))
        return pairs


class CustomReader(ReaderSensor):
    def forward(
        self,
    ) -> Any:
        if self.data:
            try:
                info = self.data[self.keyword]
                pairs = []
                for city, targets in info.enumerate():
                    for target in targets:
                        pairs.append((city, target))
                return pairs
            except:
                print("the key you requested from the reader doesn't exist")
                raise
        else:
            print("there is no data to operate on")
            raise Exception('not valid')


class NeighborDetector(TorchSensor):
    def forward(self,) -> Any:
        detector = []
        for pair in self.inputs[0]:
            if pair in self.inputs[1]:
                detector.append(1)
            else:
                detector.append(0)

