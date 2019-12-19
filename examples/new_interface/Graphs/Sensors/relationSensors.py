from regr.sensor.pytorch.sensors import ReaderSensor
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
