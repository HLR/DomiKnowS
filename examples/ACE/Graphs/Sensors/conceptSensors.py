from regr.sensor.sensor import Sensor
from Graphs.Sensors.mainSensors import ReaderSensor, CallingSensor
# from .mainSensors import ReaderSensor, CallingSensor
from typing import Dict, Any
import torch


class LabelSensor(CallingSensor):

    def __init__(self, *pres, target):
        super(LabelSensor, self).__init__(*pres)
        self.target = target

    def labels(
            self,
            data_item: Dict[str, Any]
           ) -> Any:
        return data_item[self.pres[0].fullname][1]

    def forward(
        self,
        data_item: Dict[str, Any]
    ) -> Any:
        super(LabelSensor, self).forward(data_item)
        labels = self.labels(data_item)
        output = [1] * len(labels)
        for it in range(len(labels)):
            if labels[it] == self.target:
                output[it] = 0
                break
        return torch.tensor(output, device=self.device)
        



