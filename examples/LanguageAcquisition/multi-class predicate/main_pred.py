# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:27:21 2021

@author: Juan
"""

import sys

if '../../..' not in sys.path:
    sys.path.append('../../..')
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
from graph_pred import graph, predicate

from model import LearnerM, LearnerModel
from sensors_pred import SituationRepSensor, UtteranceRepSensor, CategoryRepSensor

device = 'cpu'
    
import random
        
        
def model_declaration():
    
    
    graph.detach()
    
    predicate = graph['predicate']
    category = graph['category']
    word = graph['word']
    
    
    
    predicate['predicate'] = ReaderSensor(keyword='predicate')
    predicate[category] = ReaderSensor(keyword='category',label=True)
    predicate[word] = ReaderSensor(keyword='word',label=True)
    
    
    predicate['feature'] = SituationRepSensor('predicate')
    # predicate['category_rep'] = CategoryRepSensor('category')
    # predicate['word_rep'] = UtteranceRepSensor('word')
    
    predicate[category] = ModuleLearner('feature',module=LearnerModel(38,60,4))
    predicate[word] = ModuleLearner('feature', module=LearnerModel(38,60,24))
    
    # predicate[category] = ModuleLearner('feature',module=LearnerM(38,60,4))
    # predicate[word] = ModuleLearner('feature', module=LearnerM(38,60,24))
    
    
    program = SolverPOIProgram(graph, inferTypes=['ILP', 'local/argmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()), metric={'ILP':PRF1Tracker(DatanodeCMMetric()),'argmax':PRF1Tracker(DatanodeCMMetric('local/argmax'))})

    return program

def main():
    
    logging.basicConfig(level=logging.INFO)

    program = model_declaration()
    
    
    graph.visualize("image")
    
    predicate = graph['predicate']
    Category = graph['category']
    Word = graph['word']
    
    #data = [{'predicate': ['re1(t1)'], 'label': ['red']}]
    
    # Load the training file
    train_filename = "../data/training_set.txt"
    test_filename = "../data/test_set.txt"
    
    # Get the list of predicates and words
    categories = ['color', 'shape', 'size', 'position']
    words = [x.strip() for x in open("../data/vocabulary.txt")]
    
    train_dataset = PredicateReader(train_filename,"txt")
    test_dataset = PredicateReader(train_filename,"txt")

    #device options are 'cpu', 'cuda', 'cuda:x', torch.device instance, 'auto', None
    device = 'auto'
    
    program.train(train_dataset, train_epoch_num=2, Optim=torch.optim.Adam, device='auto')
    #program.train(train_dataset, test_set=test_dataset, train_epoch_num=5, Optim=torch.optim.Adam, device='auto')
    
    program.test(test_dataset, device="auto")


    # out = open("output.txt","w")
    
    count = 1
    for item, datanode in zip(list(iter(test_dataset)),program.populate(test_dataset)):
        
        # Print to console
        print("#"*40)
        print("Label #{:d}".format(count), item)
        print('datanode:', datanode)
        
        # Print to file
        # print("#"*40,file=out)
        # print("Label #{:d}".format(count), item,file=out)
        # print('datanode:', datanode, file=out)
        
        for n,x in zip(['Category','Word'],[Category, Word]):
            
            # Print to console
            
            # Get the labels for each element
            infer_node = datanode.getAttribute(x,'ILP')
            print(infer_node.data == 1)
            
            print('\t{}:'.format(n),datanode.getAttribute(x))
            print('\tInference {}:'.format(n), datanode.getAttribute(x, 'ILP'))
            print("-"*40)
            
            # Print to file
            # print('\t{}:'.format(n),datanode.getAttribute(x),file=out)
            # print('\tInference {}:'.format(n), datanode.getAttribute(x, 'ILP'), file=out)
            # print("-"*40,file=out)
            
        count +=1
        
    # out.close()
    
if __name__ == "__main__":
    main()
    # module_test()
