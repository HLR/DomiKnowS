from typing import Dict
from .base import BaseAllenNlpSensor


class TokenSensor(BaseAllenNlpSensor):
    def __call__(self) -> Dict: pass

class TokenSequenceSensor(BaseAllenNlpSensor):
    def __call__(self) -> Dict: pass

class LabelSensor(BaseAllenNlpSensor):
    def __call__(self) -> Dict: pass

class LabelSequenceSensor(BaseAllenNlpSensor):
    def __call__(self) -> Dict: pass
