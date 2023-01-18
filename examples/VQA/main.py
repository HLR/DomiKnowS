import argparse
import os,sys
import torch
import torch.nn as nn
import numpy as np
currentdir = os.path.dirname(os.getcwd())
root = os.path.dirname(currentdir)
sys.path.append(root)


from regr.program.model_program import SolverPOIProgram
from regr.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor, FunctionalReaderSensor
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from regr.program.loss import NBCrossEntropyLoss


# Enable skeleton DataNode
def main(device):
    from graph import graph, image_group_contains, image, level1, level2, level3, level4, image_group, structure

    image_group['reps'] = FunctionalReaderSensor(keyword='reps', forward=lambda data: data.unsqueeze(0) ,device=device)

    image[image_group_contains, "reps"] = JointSensor(image_group['reps'], forward=lambda x: (torch.ones(x.shape[1], 1), x.squeeze(0)))

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

    image[level4] = FunctionalReaderSensor(image_group_contains, "reps", keyword='level4', forward=get_probs, label=False)
    image[level4] = FunctionalReaderSensor(keyword='level4_label', forward=get_label, label=True)

    f = open("logger.txt", "w")
    program = SolverPOIProgram(graph, inferTypes=['ILP', 'local/argmax'],
                                poi = (image_group, image, level1, level2, level3, level4),
                                loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                 metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                            'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))}, f=f)
    return program
if __name__ == '__main__':
    from regr.utils import setProductionLogMode
    productionMode = True
    if productionMode:
        setProductionLogMode(no_UseTimeLog=True)
    from regr.utils import setDnSkeletonMode
    setDnSkeletonMode(True)
    import logging
    logging.basicConfig(level=logging.INFO)

    from torch.utils.data import Dataset, DataLoader
    from reader import VQADataset

    file = "val.npy"
    data = np.load(file, allow_pickle=True).item()

    dataset = VQADataset(data,)
    dataloader = DataLoader(dataset, batch_size=500)

    # test_reader = VQAReader('val.npy', 'npy')
    device = 'cpu'
    program = main(device)
    program.test(dataloader, device=device)
    # program.test(list(iter(dataloader))[:2], device=device)