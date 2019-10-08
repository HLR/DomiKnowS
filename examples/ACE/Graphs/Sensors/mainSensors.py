from regr.sensor.sensor import Sensor
from typing import Dict, Any

class ReaderSensor(Sensor):
    def __init__(self, reader):
        super().__init__()
        self.reader = reader


class CallingSensor(Sensor):
    def __init__(self, *pres, output=None):
        super().__init__()
        self.pres = pres
        self.output = output

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        for pre in self.pres:
            for _, sensor in pre.find(Sensor):
                sensor(context=context)
        if self.output:
            return context[self.output.fullname]
