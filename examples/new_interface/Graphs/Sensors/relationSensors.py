from regr.sensor.pytorch.sensors import ReaderSensor, TorchSensor
from typing import Any
import torch


class RelationReaderSensor(ReaderSensor):
    def forward(
            self,
    ) -> Any:
        if self.data:
            try:
                if self.label:
                    return torch.tensor(self.data['relations'][self.keyword], device=self.device)
                else:
                    return self.data['relations'][self.keyword]
            except:
                print("the key you requested from the reader doesn't exist")
                raise
        else:
            print("there is no data to operate on")
            raise Exception('not valid')


class RangeCreatorSensor(TorchSensor):
    def forward(self,) -> Any:
        result = []
        for item in self.inputs[0]:
            result.append([self.inputs[1][item[0]][0], self.inputs[1][item[0]][1],
                           self.inputs[2][item[1]][0], self.inputs[2][item[1]][1]])
        return result
