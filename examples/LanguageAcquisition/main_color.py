








# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:27:21 2021

@author: Juan
"""

import sys

if '../..' not in sys.path:
    sys.path.append('../..')
print("sys.path - %s"%(sys.path))


from reader import PredicateReader, InteractionReader

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
from graph_color import graph, predicate, Red, Blue, Yellow, Purple, Orange, Green

from model import MyModel, MyIMLModel, Net, LearnerModel
from sensors_color import SituationRepSensor, UtteranceRepSensor

device = 'auto'

def model_declaration():
    
    
    graph.detach()


    # initialize the concepts
    # predicate = graph['predicate']
    # Red = graph['Red']
    # Blue = graph['Blue']
    # Yellow = graph['Yellow']
    # Purple = graph['Purple']
    # Orange = graph['Orange']
    # Green = graph['Green']
    
    predicate['pred'] = ReaderSensor(keyword='Pred')
    
    
    def convert(x):
        
        val = torch.zeros(1,10)
        
        predicates = ['re1', 'bl1', 'ye1', 'or1', 'gr1', 'pu1']
        
        index = None
        
        if x[0] in predicates:
            index = predicates.index(x[0])
            
        val[index] = 1
        
        return val
    
    predicate['feature'] = FunctionalSensor('pred',forward=convert)
    
    
    predicate[Red] = ModuleLearner('feature', module=nn.Linear(10,2))
    predicate[Blue] = ModuleLearner('feature', module=nn.Linear(10,2))
    predicate[Yellow] = ModuleLearner('feature', module=nn.Linear(10,2))
    predicate[Purple] = ModuleLearner('feature', module=nn.Linear(10,2))
    predicate[Orange] = ModuleLearner('feature', module=nn.Linear(10,2))
    predicate[Purple] = ModuleLearner('feature', module=nn.Linear(10,2))
    
    predicate[Red] = ReaderSensor(keyword='Red',label=True)
    predicate[Blue] = ReaderSensor(keyword='Blue',label=True)
    predicate[Yellow] = ReaderSensor(keyword='Yellow',label=True)
    predicate[Purple] = ReaderSensor(keyword='Purple',label=True)
    predicate[Orange] = ReaderSensor(keyword='Orange',label=True)
    predicate[Green] = ReaderSensor(keyword='Green',label=True)
    
    
    program = SolverPOIProgram(graph, inferTypes=['ILP', 'local/argmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()), metric={'ILP':PRF1Tracker(DatanodeCMMetric()),'argmax':PRF1Tracker(DatanodeCMMetric('local/argmax'))})

    return program

def main():
    
    
    logging.basicConfig(level=logging.INFO)

    program = model_declaration()
    
    #data = [{'predicate': ['re1(t1)'], 'label': ['red']}]
    
    # Load the training file
    train_filename = "./data/training_set.txt"
    test_filename = "./data/test_set.txt"
    
    train_dataset = PredicateReader(train_filename,"txt")
    test_dataset = PredicateReader(train_filename,"txt")

    #device options are 'cpu', 'cuda', 'cuda:x', torch.device instance, 'auto', None
    device = 'auto'
    
    program.train(train_dataset, train_epoch_num=5, Optim=torch.optim.Adam, device='auto')
    #program.train(train_dataset, test_set=test_dataset, train_epoch_num=5, Optim=torch.optim.Adam, device='auto')
    
    program.test(test_dataset, device="auto")


    out = open("output.txt","w")
    
    count = 1
    for item, datanode in zip(list(iter(test_dataset)),program.populate(test_dataset)):
        
        # Console
        print("#"*40)
        print("Label #{:d}".format(count), item)
        print('datanode:', datanode)
        
        # File
        print("#"*40,file=out)
        print("Label #{:d}".format(count), item,file=out)
        print('datanode:', datanode, file=out)
        
        for n,x in zip(['Red', 'Blue', 'Yellow', 'Purple', 'Orange', 'Green'],[Red, Blue, Yellow, Purple, Orange, Green]):
            # Console
            print('\t{}:'.format(n),datanode.getAttribute(x))
            print('\tInference {}:'.format(n), datanode.getAttribute(x, 'ILP'))
            print("-"*40)
            
            # File
            print('\t{}:'.format(n),datanode.getAttribute(x),file=out)
            print('\tInference {}:'.format(n), datanode.getAttribute(x, 'ILP'), file=out)
            print("-"*40,file=out)
        count +=1
        
    out.close()
    
if __name__ == "__main__":
    main()

