from regr.sensor.pytorch.sensors import TorchSensor, Sensor, NominalSensor
from typing import Any
import torch
from typing import Dict, Any


class WordEmbedding(TorchSensor):
    def forward(self, ) -> Any:
        return [word.embedding.view(1, 5220) for word in self.inputs[0]]


class BetweenIndexGenerator(TorchSensor):
    def forward(self, ) -> Any:
        result = []
        for item in self.inputs[0]:
            result.append([self.inputs[1][item[0]][1], self.inputs[2][item[1]][0]])
        return result


class PairIndexGenerator(TorchSensor):
    def forward(self, ) -> Any:
        result = []
        for item in range(len(self.inputs[0])):
            for item1 in range(len(self.inputs[1])):
                if self.inputs[0][item] == self.inputs[1][item1]:
                    continue
                if self.inputs[1][item1][0] < self.inputs[0][item][0]:
                    continue
                result.append([item, item1])
        return result


class MultiplyCatSensor(TorchSensor):
    def forward(self, ) -> Any:
        results = []
        if not len(self.inputs[0]):
            return torch.zeros(1, 1, 974, device=self.device)

        for item in self.inputs[0]:
            results.append(torch.cat([self.inputs[1][item[0]], self.inputs[2][item[1]]], dim=-1))
        return torch.stack(results)


class BetweenEncoderSensor(TorchSensor):
    def __init__(self, *pres, output=None, edges=None, inside=None, key=None):
        super().__init__(*pres, output=output, edges=edges)
        self.inside = inside
        self.key = key

    def update_pre_context(
            self,
            context: Dict[str, Any]
    ) -> Any:
        for edge in self.edges:
            for _, sensor in edge.find(Sensor):
                sensor(context=context)
        for pre in self.pres:
            for _, sensor in self.sup.sup[pre].find(Sensor):
                sensor(context=context)
        if self.inside:
            for _, sensor in self.inside[self.key].find(Sensor):
                sensor(context=context)

    def define_inputs(self):
        super().define_inputs()
        self.inputs.append(self.context_helper[self.inside[self.key].fullname])

    def forward(self, ) -> Any:
        results = []
        if not len(self.inputs[0]):
            return torch.zeros(1, 1, 480, device=self.device)

        for item in self.inputs[0]:
            if item[0] + 1 <= item[1]:
                data = self.inputs[-1][item[0]:item[1]]
            else:
                data = self.inputs[-1][item[0] + 1:item[1]]
            results.append(torch.mean(data, dim=0))
        return torch.stack(results)


class WordPosTaggerSensor(NominalSensor):
    def forward(
            self,
    ) -> Any:
        return self.inputs[0]


class PhraseEntityTagger(NominalSensor):
    def forward(
            self,
    ) -> Any:
        return self.inputs[0]