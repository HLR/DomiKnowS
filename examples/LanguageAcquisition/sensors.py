
import os
# import sys

# sys.path.append('../..')
# print("sys.path - %s"%(sys.path))

from regr.sensor.pytorch.sensors import FunctionalSensor

from typing import Any
import torch
from torch import nn

class SituationRepSensor(FunctionalSensor):
    def __init__(self, *pres, **kwarg):
        super().__init__(*pres, **kwarg)
        
        # Need to load the predicates
        self.predicates = [x.strip() for x in open("./data/predicates.txt")]
        self.predicates.append("<eos>")
        
    def forward(self, *inputs) -> Any:
        
        situation = inputs[0]
        indices = [self.predicates.index(logic) for logic in self.predicates]
        tensor = torch.tensor(indices,dtype=torch.long, device='auto')
        
        return tensor.view(-1,1)


class UtteranceRepSensor(FunctionalSensor):
    def __init__(self, *pres, **kwarg):
        super().__init__(*pres, **kwarg)
        
        # Need to load the predicates
        self.vocabulary = [x.strip() for x in open("./data/vocabulary.txt")]
        self.vocabulary.insert("<sos>")
        
    def forward(self, *inputs) -> Any:
        
        utterance = inputs[0]
        indices = [self.vocabulary.index(word) for word in self.vocabulary]
        tensor = torch.tensor(indices,dtype=torch.long, device='auto')
        
        return tensor.view(-1,1)