








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
from graph_color import graph, predicate, Red, Blue, Yellow, Purple, Orange, Green

from model import LearnerM, LearnerModel
from sensors_color import SituationRepSensor, UtteranceRepSensor



device = 'cpu'

import random

def module_test():
    
    learner = LearnerM(10)
    
    # Run multiple times
    # Generate random indices
    
    optimizer = torch.optim.Adam(learner.parameters())
    criterion = nn.NLLLoss()
    num_ex_train = 100
    
    optimizer.zero_grad()
    
    # print("Train the data!")
    # print("{:10s}{:10s}{:10s}".format("Expected","Predicted","Loss"))
    
    for x in range(num_ex_train):
        
        # Initialize a tensor
        index = [random.randint(0,9)]
        index_tensor = torch.tensor(index,dtype=torch.long, device='cpu')
        
        # run the module
        output = learner(index_tensor)
        
        log_soft = F.log_softmax(output,dim=1)
        
        loss = criterion(log_soft, index_tensor)
        
        
        # print("{:<10d}{:<10d}{:<10f}".format(index[0], log_soft.topk(1)[1].item(), loss.item()))
        
        loss.backward()
        optimizer.step()
        
    # Test the model
    
    with torch.no_grad():
        
        print("\nTest the data!")
        print("{:10s}{:10s}{:10s}".format("Expected","Predicted", "Correct"))
    
        correct = 0
        num_ex_test = 10
        
        for x in range(num_ex_test):
            
            # Initialize a tensor
            index = [random.randint(0,9)]
            index_tensor = torch.tensor(index,dtype=torch.long, device='cpu')
            
            
            # run the module
            output = learner(index_tensor)
            
            log_soft = F.log_softmax(output,dim=1)
            
            a= index[0]
            b = log_soft.topk(1)[1].item()
            if a==b: correct += 1
            
            print("{:<10d}{:<10d}{:<10d}".format(a, b, int(a==b)))
        
        print("Accuracy:", (correct/num_ex_test))
        
    
        
        
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
        
        predicates = ['re1', 'bl1', 'ye1', 'or1', 'gr1', 'pu1']
        
        val = torch.tensor([predicates.index(x[0])])
        
        return val.view(1,-1)
    
    predicate['feature'] = FunctionalSensor('pred',forward=convert)
    
    
    predicate[Red] = ModuleLearner('feature', module=LearnerM(10))
    predicate[Blue] = ModuleLearner('feature', module=LearnerM(10))
    predicate[Yellow] = ModuleLearner('feature', module=LearnerM(10))
    predicate[Purple] = ModuleLearner('feature', module=LearnerM(10))
    predicate[Orange] = ModuleLearner('feature', module=LearnerM(10))
    predicate[Green] = ModuleLearner('feature', module=LearnerM(10))
    
    predicate[Red] = ReaderSensor(keyword='Red',label=True)
    predicate[Blue] = ReaderSensor(keyword='Blue',label=True)
    predicate[Yellow] = ReaderSensor(keyword='Yellow',label=True)
    predicate[Purple] = ReaderSensor(keyword='Purple',label=True)
    predicate[Orange] = ReaderSensor(keyword='Orange',label=True)
    predicate[Green] = ReaderSensor(keyword='Green',label=True)
    
    
    program = SolverPOIProgram(graph, inferTypes=['ILP', 'local/softmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()), metric={'ILP':PRF1Tracker(DatanodeCMMetric()),'argmax':PRF1Tracker(DatanodeCMMetric('local/argmax'))})

    return program

def main():
    
    logging.basicConfig(level=logging.INFO)

    program = model_declaration()
    
    
    graph.visualize("image")
    
    #data = [{'predicate': ['re1(t1)'], 'label': ['red']}]
    
    # Load the training file
    train_filename = "../data/training_set.txt"
    test_filename = "../data/test_set.txt"
    
    train_dataset = PredicateReader(train_filename,"txt")
    test_dataset = PredicateReader(test_filename,"txt")
    
    colors = ['Blue','Green', 'Orange', 'Purple', 'Red', 'Yellow']

    #device options are 'cpu', 'cuda', 'cuda:x', torch.device instance, 'auto', None
    device = 'cpu'
    
    program.train(train_dataset, train_epoch_num=3, Optim=torch.optim.Adam, device='cpu')
    #program.train(train_dataset, test_set=test_dataset, train_epoch_num=5, Optim=torch.optim.Adam, device='auto')
    
    program.test(test_dataset, device="cpu")


    # out = open("output.txt","w")
    
    count = 1
    
    correct_L = [0,0,0,0,0,0]
    total_L = [0,0,0,0,0,0]
    
    for item, datanode in zip(list(iter(test_dataset)),program.populate(test_dataset)):
        
        # Print to console
        print("#"*40)
        print("Label #{:d}".format(count), item)
        print('datanode:', datanode)
        
        
        # Print to file
        # print("#"*40,file=out)
        # print("Label #{:d}".format(count), item,file=out)
        # print('datanode:', datanode, file=out)
        
        index = 0
        for n,x in zip(colors,[Blue, Green, Orange, Purple, Red, Yellow]):
            
            if item[n][0] == 1: 
                total_L[index] += 1
            
            # Print to console
            print('\t{}:'.format(n),datanode.getAttribute(x))
            print('\tInference {}:'.format(n), datanode.getAttribute(x, 'ILP'))
            print("-"*40)
            
            # Find the inference with the max value
            if item[n][0] == 1 and datanode.getAttribute(x,'ILP').item() == item[n][0]:
                correct_L[index] += 1
                
            index += 1
                
        
            # Print to file
            # print('\t{}:'.format(n),datanode.getAttribute(x),file=out)
            # print('\tInference {}:'.format(n), datanode.getAttribute(x, 'ILP'), file=out)
            # print("-"*40,file=out)
            
        count +=1
        
        # if max_val > -1 and item[max_lbl] == [1]:
        #     correct_L.append(max_lbl)
        
    ILPmetrics = datanode.getInferMetric()
    print("\nILP metrics Total %s"%(ILPmetrics['Total']))
    # out.close()
    
    
    # Count the number of labels
    acc = [0]*len(colors)
    
    for i in range(len(colors)):
        acc[i] = correct_L[i] / total_L[i]
        
    print("\nFrequency distribution for each color")
    for i,c in enumerate(colors):
        print("\t{:s}: {:d}".format(c,total_L[i]))
    
    print("\nNumber of correct predictions per color")
    for i,c in enumerate(colors):
        print("\t{:s}: {:d}".format(c,correct_L[i]))
    
    print("\nAccuracy for each color")
    for c,accuracy in zip(colors,acc):
        print("\t{:s}: {:.2f}".format(c,accuracy))
    
if __name__ == "__main__":
    main()
    # module_test()
