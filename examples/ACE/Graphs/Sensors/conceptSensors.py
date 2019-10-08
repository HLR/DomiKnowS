from regr.sensor.sensor import Sensor
from .mainSensors import ReaderSensor


class LabelSensor(ReaderSensor):
    def __init__(self, reader):
        super().__init__(reader=reader)

