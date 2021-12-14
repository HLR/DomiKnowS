
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
        # self.predicates.append("<eos>")
        
        self.max_len = 12
        
        # print("Predicate:", len(self.predicates))
        
    def forward(self, *inputs) -> Any:

        situation = inputs[0][0].split()
        indices = [self.predicates.index(logic) for logic in situation]
        tensor = torch.tensor(indices,dtype=torch.long, device='cpu')
        
        print("Situation_Vector size:", tensor.size())
        tensor = tensor.unsqueeze(0)
        print("Situation_Vector size (after adding dimensions):", tensor.size())

        # print("Indices:", indices)
        # print("Situation Tensor:", tensor)
        
        return tensor # Need to add extra dimension

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
        # self.vocabulary.insert(0, "<sos>")
        self.vocab_size = len(self.vocabulary)
        
        self.max_len = 12
        
        # print("Vocabulary:", len(self.vocabulary))
        
    def forward(self, *inputs) -> Any:
        
        utterance = inputs[0][0]
        indices = [self.vocabulary.index(word) for word in utterance]
        
        # # need to pad the list of indices
        # indices = [self.vocab_size-1 for i in range(self.max_len)]
        
        # for i,w in enumerate(utterance):
        #     indices[i] = self.vocabulary.index(w)
        
        tensor = torch.tensor(indices,dtype=torch.long, device='cpu')
        
        # print("Indices: ", indices)
        # print("Utterance Tensor: ", tensor)
        
        return tensor.view(-1,1)
    
    
# Define a function that combines inputs to form the relation
def create_words(situation_text, utterance_text):
    return torch.ones((len(situation_text[0]), 1)), [situation_text[0]], [utterance_text[0]]
    
