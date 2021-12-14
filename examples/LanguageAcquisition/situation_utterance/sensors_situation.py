
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
        self.predicates = [x.strip() for x in open("../data/predicates.txt")]
        self.predicates.append("<eos>")
        
        self.max_len = 12
        
        print("Predicate:", len(self.predicates))
        
    def forward(self, *inputs) -> Any:
        
        situation = inputs[0]
        indices = [self.predicates.index(logic) for logic in situation]
        tensor = torch.tensor(indices,dtype=torch.long, device='cpu')
        
        print("Indices:", indices)
        print("Situation Tensor:", tensor)
        
        return tensor.view(-1,1)

class CategoryRepSensor(FunctionalSensor):
    def __init__(self, *pres, **kwarg):
        super().__init__(*pres, **kwarg)
        
        # Need to load the predicates
        self.categories = ['color', 'shape', 'size', 'position']
        
        print("Categories:", len(self.categories))
        
    def forward(self, *inputs) -> Any:
        
        category = inputs[0]
        indices = [self.categories.index(word) for word in category]
        tensor = torch.tensor(indices,dtype=torch.long, device='cpu')
        
        
        return tensor.view(-1,1)

class UtteranceRepSensor(FunctionalSensor):
    def __init__(self, *pres, **kwarg):
        super().__init__(*pres, **kwarg)
        
        # Need to load the predicates
        self.vocabulary = [x.strip() for x in open("../data/vocabulary.txt")]
        self.vocabulary.insert(0, "<sos>")
        
        self.max_len = 12
        
        print("Vocabulary:", len(self.vocabulary))
        
    def forward(self, *inputs) -> Any:
        
        utterance = inputs[0]
        indices = [self.vocabulary.index(word) for word in utterance]
        tensor = torch.tensor(indices,dtype=torch.long, device='cpu')
        
        print("Indices: ", indices)
        print("Utterance Tensor: ", tensor)
        
        return tensor.view(-1,1)