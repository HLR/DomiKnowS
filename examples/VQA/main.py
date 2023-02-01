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
    program = SolverPOIProgram(graph, inferTypes=[
        # 'ILP', 
        'local/argmax'],
                                poi = (image_group, image, level1, level2, level3, level4),
                                loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                 metric={
                                    # 'ILP': PRF1Tracker(DatanodeCMMetric()),
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
    from graph import *

    file = "val.npy"
    data = np.load(file, allow_pickle=True).item()

    dataset = VQADataset(data,)
    dataloader = DataLoader(dataset, batch_size=1000)
    # dataloader = DataLoader(dataset, batch_size=20)

    # test_reader = VQAReader('val.npy', 'npy')
    device = 'cpu'
    program = main(device)
    program.test(dataloader, device=device)
    # program.test(list(iter(dataloader))[:40], device=device)

    # corrects = {
    #     'level1': 0,
    #     'level2': 0,
    #     'level3': 0,
    #     'level4': 0
    # }

    # consistent_corrects = {
    #     'level1': 0,
    #     'level2': 0,
    #     'level3': 0,
    #     'level4': 0
    # }

    # ilp_corrects = {
    #     'level1': 0,
    #     'level2': 0,
    #     'level3': 0,
    #     'level4': 0
    # }

    # totals = {
    #     'level1': 0,
    #     'level2': 0,
    #     'level3': 0,
    #     'level4': 0
    # }

    # backsteps = {'level3': 'level2', 'level4': 'level3'}

    # from tqdm import tqdm
    # for datanode in tqdm(program.populate(list(iter(dataloader))[:40], device=device)):
    # # for datanode in tqdm(program.populate(dataloader, device=device)):
    #     for child in datanode.getChildDataNodes('image'):
    #         pred = child.getAttribute('level1', 'local/argmax').argmax().item()
    #         pred_ilp = child.getAttribute('level1', 'ILP').argmax().item()
    #         label = child.getAttribute('level1', 'label').item()
    #         if pred == label:
    #             corrects['level1'] += 1
    #         if pred_ilp == label:
    #             ilp_corrects['level1'] += 1
    #         totals['level1'] += 1
    #         for _concept in ['level2', 'level3', 'level4']:
    #             label = child.getAttribute(_concept, 'label').item()
    #             if len(hierarchy[_concept]) != label:
    #                 totals[_concept] += 1
    #                 pred = child.getAttribute(_concept, 'local/argmax').argmax().item()
    #                 pred_ilp = child.getAttribute(_concept, 'ILP').argmax().item()
    #                 if _concept in backsteps:
    #                     if prior != len(hierarchy[backsteps[_concept]]):
    #                         if label == pred:
    #                             consistent_corrects[_concept] += 1
    #                 if label == pred:
    #                     corrects[_concept] += 1
    #                 if label == pred_ilp:
    #                     ilp_corrects[_concept] += 1
    #             prior = pred

    # file2 = open("logger_manual.txt", "w")
    # for _c in corrects:
    #     if totals[_c] != 0:
    #         print(f"{_c} accuracy: {corrects[_c]/totals[_c]}", file=file2)
    #         print(f"{_c} ILP accuracy: {ilp_corrects[_c]/totals[_c]}", file=file2)  
    #         if _c in backsteps:
    #             print(f"{_c} consistent accuracy: {consistent_corrects[_c]/totals[_c]}", file=file2)
    #     else:
    #         print(f"no instances for {_c}", file=file2)                 
