# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:27:21 2021

@author: Juan
"""

import sys

if '../../..' not in sys.path:
    sys.path.append('../../..')
print("sys.path - %s"%(sys.path))



import torch
from torch import nn
import torch.nn.functional as F

from regr.graph import Graph, Concept, Relation
from regr.graph import ifL, notL, andL, orL
from regr.program import LearningBasedProgram, POIProgram
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.sensors import ReaderSensor
from regr.program.model.pytorch import PoiModel
from regr.program.metric import MacroAverageTracker, PRF1Tracker
from regr.program.loss import NBCrossEntropyLoss
from regr.program import POIProgram, IMLProgram, SolverPOIProgram
from regr.program.model.pytorch import PoiModel
from regr.program.metric import MacroAverageTracker, PRF1Tracker, PRF1Tracker, DatanodeCMMetric
from regr.program.loss import NBCrossEntropyLoss


from regr.graph import Property

from regr.sensor.pytorch.sensors import TorchSensor, FunctionalSensor
import spacy
from typing import Any
import torch

import logging


# Graph declaration: Need to build a simple situation-utterance model
from graph_situation import graph, situation, utterance

from model_situation import LearnerM, LearnerModel, RNNEncoder, RNNDecoder 
from sensors_situation import SituationRepSensor, UtteranceRepSensor, CategoryRepSensor
from reader_situation import PredicateReader, InteractionReader

device = 'cpu'
    
import random
        
def loss_fn(self,outputs,labels):
        '''
            Computes the negative-log likelihood function loss for the word prediction
            
            Parameters
                outputs: The output value from the forward pass
                labels: The target value for the evaluation
                 
            Returns
                NLLLoss object
        '''
        
        return nn.NLLLoss()(outputs,labels)    

def model_declaration():
    
    
    graph.detach()
    
    situation = graph['situation']
    utterance = graph['utterance']
    
    
    situation['text'] = ReaderSensor(keyword='situation')
    utterance['text'] = ReaderSensor(keyword='utterance')
    
    
    # Use functional sensor instead
    def Situation_Convert(*inputs):
        
        predicates = [x.strip() for x in open("../data/predicates.txt")]
        predicates.append("<eos>")
        
        max_len = 12
        
        situation = inputs[0][0]
        indices = [predicates.index(logic) for logic in situation]
        tensor = torch.tensor(indices,dtype=torch.long, device='cpu')
        
        print("Indices:", indices)
        print("Situation Tensor:", tensor)
        
        return tensor.view(-1,1)
    
    def Utterance_Convert(*inputs):
        
        vocabulary = [x.strip() for x in open("../data/vocabulary.txt")]
        vocabulary.insert(0, "<sos>")
        
        max_len = 12
        
        
        utterance = inputs[0][0]
        indices = [vocabulary.index(word) for word in utterance]
        tensor = torch.tensor(indices,dtype=torch.long, device='cpu')
        
        print("Indices: ", indices)
        print("Utterance Tensor: ", tensor)
        
        return tensor.view(-1,1)
    
    # situation['sit_emb'] = SituationRepSensor('text')
    # utterance['utt_emb'] = UtteranceRepSensor('text')
    
    situation['sit_emb'] = FunctionalSensor('text',forward=Situation_Convert)
    utterance['utt_emb'] = FunctionalSensor('text',forward=Utterance_Convert)
    
    # Plan the learner module
    
    # Prepare the parameters for the encoder/decoder
    encoder_dim = (36,100)
    decoder_dim = (100,23)
    learning_rate = 0.001
    max_length = 12
    vocabulary = [x.strip() for x in open("../data/vocabulary.txt")]
    predicates = [x.strip() for x in open("../data/predicates.txt")]
    
    utterance['output'] = ModuleLearner(situation['sit_emb'], module=LearnerModel(vocabulary, predicates,\
                                                                    encoder_dim, decoder_dim, learning_rate, max_length))
    program = SolverPOIProgram(graph, inferTypes=['ILP', 'local/argmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()), metric={'ILP':PRF1Tracker(DatanodeCMMetric()),'argmax':PRF1Tracker(DatanodeCMMetric('local/argmax'))})

    return program

def main():
    
    logging.basicConfig(level=logging.INFO)

    program = model_declaration()
    
    
    graph.visualize("image")
    
    situation = graph['situation']
    utterance = graph['utterance']
    
    # Load the training file
    train_filename = "../data/training_set.txt"
    test_filename = "../data/test_set.txt"
    
    # Get the list of predicates and words
    words = [x.strip() for x in open("../data/vocabulary.txt")]
    
    train_dataset = InteractionReader(train_filename,"txt")
    test_dataset = InteractionReader(train_filename,"txt")

    #device options are 'cpu', 'cuda', 'cuda:x', torch.device instance, 'auto', None
    device = 'cpu'
    
    
    program.train(train_dataset, train_epoch_num=1, Optim=torch.optim.Adam, device='cpu')
    
    program.test(test_dataset, device=device)
    
    count = 1
    for item, datanode in zip(list(iter(test_dataset)),program.populate(test_dataset)):
        
        # Print to console
        print("#"*40)
        print("Label #{:d}".format(count), item)
        print('datanode:', datanode)
        print('\tInference {}:'.format("Word Sequence"), datanode.getAttribute('utterance', 'ILP'))
        print("-"*40)
        
        
       
    
if __name__ == "__main__":
    main()
    # module_test()
