from typing import Dict
from .. import Sensor, Learner

class BaseAllenNlpSensor(Sensor):
    def __call__(self) -> Dict: pass

class BaseAllenNlpLearner(BaseAllenNlpSensor, Learner):
    def __init__(self, *args, **kwargs): pass

    def __call__(self) -> Dict: pass
