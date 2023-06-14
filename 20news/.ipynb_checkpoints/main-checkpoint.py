import argparse
import os,sys
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append(".")
sys.path.append("../..")


from domiknows.program.model_program import SolverPOIProgram
from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor, FunctionalReaderSensor, ModuleLearner
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from domiknows.program.loss import NBCrossEntropyLoss


# Enable skeleton DataNode
def main(device):
    from graph import graph, image_group_contains, image, level1, level2, level3, level4, image_group, structure

    image_group['input_ids'] = FunctionalReaderSensor(keyword='input_ids', forward=lambda data: data.unsqueeze(0) ,device=device)
    
    image_group['attention_mask'] = FunctionalReaderSensor(keyword='attention_mask', forward=lambda data: data.unsqueeze(0) ,device=device)
    
    image_group['reps'] = Module
    

    image[image_group_contains, "rep"] = JointSensor(image_group['reps'], forward=lambda x: (torch.ones(x.shape[1], 1), x.squeeze(0), x.squeeze(0)))

    def get_probs(*inputs, data):
        return data
        # return torch.softmax(data, -1)

    def get_label(*inputs, data):
        return data

    image[level1] = FunctionalReaderSensor(image_group_contains, "reps", keyword='level1', forward=get_probs, label=False)
    image[level1] = FunctionalReaderSensor(keyword='level1_label', forward=get_label, label=True)

    image[level2] = FunctionalReaderSensor(image_group_contains, "reps", keyword='level2', forward=get_probs, label=False)
    image[level2] = FunctionalReaderSensor(keyword='level2_label', forward=get_label, label=True)

    image[level3] = FunctionalReaderSensor(image_group_contains, "reps", keyword='level3', forward=get_probs, label=False)
    image[level3] = FunctionalReaderSensor(keyword='level3_label', forward=get_label, label=True)
    
    prefix = "Tasks/20news/"
    f = open(f"{prefix}logger.txt", "w")
    program = SolverPOIProgram(graph, inferTypes=[
        'ILP', 
        'local/argmax'],
        # probKey = ("local" , "meanNormalizedProbStd"),
                                poi = (image_group, image, level1, level2, level3, level4),
                                loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                 metric={
                                    # 'ILP': PRF1Tracker(DatanodeCMMetric()),
                                            'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))}, f=f)
    return program