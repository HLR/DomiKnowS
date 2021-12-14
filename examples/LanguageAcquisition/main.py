# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:27:21 2021

@author: Juan
"""

import sys
sys.path.append('../..')
print("sys.path - %s"%(sys.path))


from reader import InteractionReader

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

from regr.sensor.pytorch.sensors import TorchSensor, FunctionalSensor
import spacy
from typing import Any
import torch

import logging


# Graph declaration: Need to build a simple situation-utterance model
from graph import graph, situation, utterance 
from sensors import SituationRepSensor
from model import LearnerModel

device = 'auto'

def model_declaration():
    '''
        This function initializes the graph, sensors, learners, and program
        for the task
        
        Sensors
            situation['text']: ReaderSensor with the situation string
            utterance['text']: ReaderSensor with the utterance string
            
            situation['emb']: Sensor that converts text to vector embedding
    '''    
    graph.detach()
    
    situation = graph['situation']
    utterance = graph['utterance']
    
    
    situation['text'] = ReaderSensor(keyword='situation')
    utterance['text'] = ReaderSensor(keyword='utterance')
    
    
    # Use the text property from the situation to convert it into the vector embedding
    situation['situation_rep'] = SituationRepSensor('text')
    
    
    # Place the learner module here!
    # Need to build a learner module
    utterance[situation] = ModuleLearner('situation_rep',module=LearnerModel(35,100,35))
    
    program = SolverPOIProgram(graph, inferTypes=['ILP', 'local/argmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()), metric={'ILP':PRF1Tracker(DatanodeCMMetric()),'argmax':PRF1Tracker(DatanodeCMMetric('local/argmax'))})

    return program
 
    


def main():

    
    logging.basicConfig(level=logging.INFO)

    program = model_declaration()

    # Load the training file
    filename = "./data/training_set.txt"
    train_reader = InteractionReader(filename,"txt")
    test_reader = InteractionReader(filename,"txt")

    # device options are 'cpu', 'cuda', 'cuda:x', torch.device instance, 'auto', None
    device = 'auto'
    
    
    print("Start!!!!!!")

    program.train(train_reader, train_epoch_num=1, Optim=torch.optim.Adam, device=device)
    
    
    program.test(test_reader, device="auto")
    
    print("Finish!")

    for datanode in program.populate(list(iter(test_reader))):
        print('datanode:', datanode)
        # print('Spam:', datanode.getAttribute(situation))
        # print('Regular:', datanode.getAttribute(utterance))
        # print('inference spam:', datanode.getAttribute(situation, 'ILP'))
        # print('inference regular:', datanode.getAttribute(utterance, 'ILP'))


if __name__ == "__main__":
    main()