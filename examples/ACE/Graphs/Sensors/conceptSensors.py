from regr.sensor.sensor import Sensor
from .mainSensors import ReaderSensor, CallingSensor
from typing import Dict, Any
import torch


class LabelSensor(CallingSensor):

    def __init__(self, *pres, target):
        super(LabelSensor, self).__init__(pres)
        self.target = target

    def labels(
            self,
            context: Dict[str, Any]
           ) -> Any:
        return context[self.pres[0].fullname][1]

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        super(LabelSensor, self).forward(context=context)
        labels = self.labels(context=context)
        output = torch.zeros(len(labels))
        for it in range(len(labels)):
            if labels[it] == self.target :
                output[it] = 1
                break
        return output
        


