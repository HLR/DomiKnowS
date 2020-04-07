from typing import Any
import torch

from regr.sensor.pytorch.sensors import TorchSensor, TorchEdgeSensor


class TestSensor(TorchSensor):
    def __init__(self, *pres, output=None, edges=None, label=False, expected_inputs=None, expected_outputs=None, **kwargs):
        super().__init__(*pres, output=output, edges=edges, label=label)
        self._expected_inputs = expected_inputs
        self._expected_outputs = expected_outputs

    @property
    def expected_inputs(self):
        return self._expected_inputs

    @property
    def expected_outputs(self):
        return self._expected_outputs

    def forward(self,) -> Any:
        if self.expected_inputs is not None:
            assert self.inputs == self.expected_inputs
        return self.expected_outputs


class DummyEdgeStoW(TorchEdgeSensor):
    def forward(self,) -> Any:
        return ["John", "works", "for", "IBM"]


class DummyEdgeStoP(TorchEdgeSensor):
    def forward(self,) -> Any:
        return ["John", "works for", "IBM"]


class DummyEdgeWtoP(TorchEdgeSensor):
    def forward(self,) -> Any:
        return [(0, 0), (2,2), (3, 3)]


class DummyEdgeWtoPOpt2(TorchEdgeSensor):
    def forward(self,) -> Any:
        #self.inputs == ["John", "works", "for", "IBM"]
        # opt 2
        return [("John"), ("works", "for"), ("IBM")]


class DummyEdgeWtoPair(TorchEdgeSensor):
    def forward(self,) -> Any:
        return self.inputs[0]


class DummyEdgeWtoC(TorchEdgeSensor):
    def forward(self,) -> Any:
        # # opt 1
        # self.inputs == ["John", "works", "for", "IBM"]
        return [["J", "o", "h", "n"], ["w", "o", "r", "k", "s"], ["f", "o", "r"], ["I", "B", "M"]]
        #return ["J", "o", "h", "n", "w", "o"]

        # # opt 2
        # self.inputs == "John"
        #return ["J", "o", "h", "n"]


class DummyEdgeWtoCOpt2(TorchEdgeSensor):
    def forward(self,) -> Any:
        # # opt 2
        # self.inputs == "John"
        if self.inputs[0] == "John":
            return ["J", "o", "h", "n"]
        elif self.inputs[0] == "works":
            return ["w", "o", "r", "k", "s"]
        elif self.inputs[0] == "for":
            return ["f", "o", "r"]
        elif self.inputs[0] == "IBM":
            return ["I", "B", "M"]
        return list(self.inputs[0])


class DummyEdgeWtoCOpt3(TorchEdgeSensor):
    def forward(self,) -> Any:
        # # opt 3
        # self.inputs == "John"
        return ["J", "o", "h", "n", " ", "w", "o", "r", "k", "s", " ", "f", "o", "r", " ", "I", "B", "M"]


class DummyWordEmb(TestSensor):
    pass


class DummyCharEmb(TestSensor):
    pass


class DummyPhraseEmb(TestSensor):
    pass


class DummyFullyConnectedLearner(TestSensor):
    pass
