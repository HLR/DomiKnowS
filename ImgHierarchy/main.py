import argparse
import os,sys
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append(".")
sys.path.append("../..")


from domiknows.program.model_program import SolverPOIProgram
from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor, FunctionalReaderSensor
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from domiknows.program.loss import NBCrossEntropyLoss


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

    prefix = "Tasks/ImgHierarchy/"
    f = open(f"{prefix}logger.txt", "w")
    program = SolverPOIProgram(graph, inferTypes=[
        # 'ILP', 
        'local/argmax'],
        # probKey = ("local" , "normalizedProb"),
        # probAcc = {'level1': 0.5673, 'level2': 0.5445, 'level3': 0.4342, 'level4': 0.1768},
        poi = (image_group, image, level1, level2, level3, level4),
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
            metric={
                # 'ILP': PRF1Tracker(DatanodeCMMetric()),
                'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))}, f=f)
    return program

if __name__ == '__main__':
    from domiknows.utils import setProductionLogMode
    productionMode = True
    if productionMode:
        setProductionLogMode(no_UseTimeLog=False)
    from domiknows.utils import setDnSkeletonMode
    setDnSkeletonMode(True)
    import logging
    logging.basicConfig(level=logging.INFO)

    from torch.utils.data import Dataset, DataLoader
    from reader import VQADataset
    from graph import *
    import time

    file = "Tasks/ImgHierarchy/data/test_logits.pt"
    # file = "data_sample/val_small.npy"
    start = time.time()
    if "npy" in file:
        data = np.load(file, allow_pickle=True).item()
    else:
        data = torch.load(file)

    dataset = VQADataset(data,)
    # dataset = torch.utils.data.Subset(dataset, [i for i in range(2000)])
    dataloader = DataLoader(dataset, batch_size=50000)
    end = time.time()
    print(f"elapsed time for dataloader {end - start}")
    # dataloader = DataLoader(dataset, batch_size=20)

    # test_reader = VQAReader('val.npy', 'npy')
    device = 'cuda:1'
    program = main(device)

    # program.test(dataloader, device=device)
    # program.test(list(iter(dataloader))[:40], device=device)

    corrects = {
        'level1': 0,
        'level2': 0,
        'level3': 0,
        'level4': 0
    }

    sequential_corrects = {
        'level1': 0,
        'level2': 0,
        'level3': 0,
        'level4': 0
    }

    consistent_corrects = {
        'level1': 0,
        'level2': 0,
        'level3': 0,
        'level4': 0
    }

    ilp_corrects = {
        'level1': 0,
        'level2': 0,
        'level3': 0,
        'level4': 0
    }

    totals = {
        'level1': 0,
        'level2': 0,
        'level3': 0,
        'level4': 0
    }

    backsteps = {'level3': 'level2', 'level4': 'level3'}
    per_class_tp = {"level1": {}, "level2": {}, "level3": {}, "level4": {}}
    per_class_fp = {"level1": {}, "level2": {}, "level3": {}, "level4": {}}
    per_class_fn = {"level1": {}, "level2": {}, "level3": {}, "level4": {}}

    per_class_tp_ilp = {"level1": {}, "level2": {}, "level3": {}, "level4": {}}
    per_class_fp_ilp = {"level1": {}, "level2": {}, "level3": {}, "level4": {}}
    per_class_fn_ilp = {"level1": {}, "level2": {}, "level3": {}, "level4": {}}

    per_class_tp_sequential = {"level1": {}, "level2": {}, "level3": {}, "level4": {}}
    per_class_fp_sequential = {"level1": {}, "level2": {}, "level3": {}, "level4": {}}
    per_class_fn_sequential = {"level1": {}, "level2": {}, "level3": {}, "level4": {}}

    get_satisfaction = False
    if get_satisfaction:
        program.verifyResultsLC(dataloader, device="cpu")

    support_set = {"level1": {}, "level2": {}, "level3": {}, "level4": {}}
    total_support_set = {"level1": 0, "level2": 0, "level3": 0, "level4": 0}
    calculate_ilp = False
    calculate_sequential = False
    run_metrics = False

    ### load hierarchy_flat.json and reverse its order
    with open('Tasks/ImgHierarchy/hierarchy_flat.json') as f:
        hierarchy_flat = json.load(f)
    
    reverse_flat_hierarchy = {"level1": {}, "level2": {}, "level3": {}}
    for _level in hierarchy_flat.keys():
        if _level == "level4":
            continue
        reverse_flat_hierarchy[_level] = {}
        for key, value in hierarchy_flat[_level].items():
            for val in value:
                reverse_flat_hierarchy[_level][val] = key
    from tqdm import tqdm

    run_consistent_metric = False
    consistent_all = 0
    consistent_positive = 0
    consistent_positive_ilp = 0
    if run_consistent_metric:
        for datanode in tqdm(program.populate(dataloader, device=device)):
            for child in datanode.getChildDataNodes('image'):
                consistent_all += 4
                consistent_labels = []
                consistent_preds = []
                consistent_preds_ilp = []
                for _concept in [level1, level2, level3, level4]:
                    label = child.getAttribute(_concept.name, 'label').item()
                    label_class = _concept.enum[label]
                    if label_class == "None":
                        break
                    pred = child.getAttribute(_concept.name, 'local/argmax').argmax().item()
                    pred_class = _concept.enum[pred]
                    pred_ilp = child.getAttribute(_concept.name, 'ILP').argmax().item()
                    pred_ilp_class = _concept.enum[pred_ilp]

                    consistent_labels.append(label_class)
                    consistent_preds.append(pred_class)
                    consistent_preds_ilp.append(pred_ilp_class)
                if consistent_labels == consistent_preds:
                    consistent_positive += len(consistent_labels)
                else:
                    i = len(consistent_labels) - 1
                    while i > 0:
                        if consistent_labels[:i] == consistent_preds[:i]:
                            consistent_positive += i
                        i = i - 1
                if consistent_labels == consistent_preds_ilp:
                    consistent_positive_ilp += len(consistent_labels)
                else:
                    i = len(consistent_labels) - 1
                    while i > 0:
                        if consistent_labels[:i] == consistent_preds_ilp[:i]:
                            consistent_positive_ilp += i
                        i = i - 1
        print(f"consistent accuracy: {consistent_positive/consistent_all}")
        print(f"consistent ILP accuracy: {consistent_positive_ilp/consistent_all}")


    # for datanode in tqdm(program.populate(list(iter(dataloader))[:20], device=device)):
    for datanode in tqdm(program.populate(dataloader, device=device)):
        # start = time.time()
        if run_metrics:
            for child in datanode.getChildDataNodes('image'):
                pred = child.getAttribute('level1', 'local/argmax').argmax().item()
                pred_class = level1.enum[pred]
                if calculate_ilp:
                    pred_ilp = child.getAttribute('level1', 'ILP').argmax().item()
                    pred_ilp_class = level1.enum[pred_ilp]
                label = child.getAttribute('level1', 'label').item()
                label_class = level1.enum[label]
                total_support_set['level1'] += 1
                if label_class not in support_set['level1']:
                    support_set['level1'][label_class] = 0
                if label_class not in per_class_tp['level1']:
                    per_class_tp['level1'][label_class] = 0
                    per_class_fp['level1'][label_class] = 0
                    per_class_fn['level1'][label_class] = 0
                    per_class_tp_ilp['level1'][label_class] = 0
                    per_class_fp_ilp['level1'][label_class] = 0
                    per_class_fn_ilp['level1'][label_class] = 0
                if pred_class not in per_class_tp['level1']:
                    per_class_tp['level1'][pred_class] = 0
                    per_class_fp['level1'][pred_class] = 0
                    per_class_fn['level1'][pred_class] = 0
                    per_class_tp_ilp['level1'][pred_class] = 0
                    per_class_fp_ilp['level1'][pred_class] = 0
                    per_class_fn_ilp['level1'][pred_class] = 0
                if calculate_ilp:
                    if pred_ilp_class not in per_class_tp['level1']:
                        per_class_tp['level1'][pred_ilp_class] = 0
                        per_class_fp['level1'][pred_ilp_class] = 0
                        per_class_fn['level1'][pred_ilp_class] = 0
                        per_class_tp_ilp['level1'][pred_ilp_class] = 0
                        per_class_fp_ilp['level1'][pred_ilp_class] = 0
                        per_class_fn_ilp['level1'][pred_ilp_class] = 0

                support_set['level1'][label_class] += 1

                if pred == label:
                    corrects['level1'] += 1
                    per_class_tp['level1'][label_class] += 1
                else:
                    per_class_fp['level1'][pred_class] += 1
                    per_class_fn['level1'][label_class] += 1
                if calculate_ilp:
                    if pred_ilp == label:
                        ilp_corrects['level1'] += 1
                        per_class_tp_ilp['level1'][label_class] += 1
                    else:
                        per_class_fp_ilp['level1'][pred_ilp_class] += 1
                        per_class_fn_ilp['level1'][label_class] += 1
                
                totals['level1'] += 1
                for _concept in [level2, level3, level4]:
                    none_index = _concept.enum.index('None')
                    label = child.getAttribute(_concept.name, 'label').item()
                    label_class = _concept.enum[label]
                    if label_class != "None":
                        total_support_set[_concept.name] += 1
                    pred = child.getAttribute(_concept.name, 'local/argmax').argmax().item()
                    pred_class = _concept.enum[pred]
                    if calculate_ilp:
                        pred_ilp = child.getAttribute(_concept.name, 'ILP').argmax().item()
                        pred_ilp_class = _concept.enum[pred_ilp]
                    if label_class not in support_set[_concept.name]:
                        support_set[_concept.name][label_class] = 0

                    if label_class not in per_class_tp[_concept.name]:
                        per_class_tp[_concept.name][label_class] = 0
                        per_class_fp[_concept.name][label_class] = 0
                        per_class_fn[_concept.name][label_class] = 0
                        per_class_tp_ilp[_concept.name][label_class] = 0
                        per_class_fp_ilp[_concept.name][label_class] = 0
                        per_class_fn_ilp[_concept.name][label_class] = 0
                    if pred_class not in per_class_tp[_concept.name]:
                        per_class_tp[_concept.name][pred_class] = 0
                        per_class_fp[_concept.name][pred_class] = 0
                        per_class_fn[_concept.name][pred_class] = 0
                        per_class_tp_ilp[_concept.name][pred_class] = 0
                        per_class_fp_ilp[_concept.name][pred_class] = 0
                        per_class_fn_ilp[_concept.name][pred_class] = 0
                    if calculate_ilp:
                        if pred_ilp_class not in per_class_tp[_concept.name]:
                            per_class_tp[_concept.name][pred_ilp_class] = 0
                            per_class_fp[_concept.name][pred_ilp_class] = 0
                            per_class_fn[_concept.name][pred_ilp_class] = 0
                            per_class_tp_ilp[_concept.name][pred_ilp_class] = 0
                            per_class_fp_ilp[_concept.name][pred_ilp_class] = 0
                            per_class_fn_ilp[_concept.name][pred_ilp_class] = 0

                    support_set[_concept.name][label_class] += 1
                    if len(hierarchy[_concept.name]) != label:
                        totals[_concept.name] += 1
                        if _concept.name in backsteps:
                            if prior != len(hierarchy[backsteps[_concept.name]]):
                                if label == pred:
                                    consistent_corrects[_concept.name] += 1
                        if label == pred:
                            corrects[_concept.name] += 1
                        if calculate_ilp:
                            if label == pred_ilp:
                                ilp_corrects[_concept.name] += 1
                    prior = pred

                    if pred_class == label_class:
                        if label != none_index:
                            per_class_tp[_concept.name][label_class] += 1
                    else:
                        if pred != none_index:
                            per_class_fp[_concept.name][pred_class] += 1
                        if label != none_index:
                            per_class_fn[_concept.name][label_class] += 1
                    if calculate_ilp:
                        if pred_ilp_class == label_class:
                            if label != none_index:
                                per_class_tp_ilp[_concept.name][label_class] += 1
                        else:
                            if pred_ilp != none_index: 
                                per_class_fp_ilp[_concept.name][pred_ilp_class] += 1
                            if label != none_index:
                                per_class_fn_ilp[_concept.name][label_class] += 1
            
            ### calculate sequential fix
            if calculate_sequential:
                for child in datanode.getChildDataNodes('image'):
                    original_preds = []
                    sequential_preds = []
                    for _concept in [level1, level2, level3, level4]:
                        pred = child.getAttribute(_concept.name, 'local/argmax').argmax().item()
                        pred_class = _concept.enum[pred]
                        original_preds.append(pred_class)
                    ### propagating None labels forward
                    sequential_preds.append(original_preds[0])
                    for i in range(1, len(original_preds)):
                        if sequential_preds[i-1] == 'None':
                            sequential_preds.append("None")
                        else:
                            sequential_preds.append(original_preds[i])
                    ### find the furthest level not None level
                    lowest_level = 0
                    for i in range(len(sequential_preds)):
                        if sequential_preds[i] != 'None':
                            lowest_level = i
                    ### propagate the lowest level label to all the previous levels based on the reverse_flat_hierarchy
                    if lowest_level != 0:
                        for i in range(lowest_level-1, -1, -1):
                            _map_key = {2: "level3", 1: "level2", 0: "level1"}
                            sequential_preds[i] = reverse_flat_hierarchy[_map_key[i]][sequential_preds[i+1]]
                    ### calculate metrics
                    for i, _concept in enumerate([level1, level2, level3, level4]):
                        if "None" in _concept.enum:
                            none_index = _concept.enum.index('None')
                        else:
                            none_index = -1
                        label = child.getAttribute(_concept.name, 'label').item()
                        label_class = _concept.enum[label]
                        pred_class = sequential_preds[i]

                        if label_class not in per_class_tp_sequential[_concept.name]:
                            per_class_tp_sequential[_concept.name][label_class] = 0
                            per_class_fp_sequential[_concept.name][label_class] = 0
                            per_class_fn_sequential[_concept.name][label_class] = 0
                        if pred_class not in per_class_tp_sequential[_concept.name]:
                            per_class_tp_sequential[_concept.name][pred_class] = 0
                            per_class_fp_sequential[_concept.name][pred_class] = 0
                            per_class_fn_sequential[_concept.name][pred_class] = 0

                        if len(hierarchy[_concept.name]) != label:
                            if label_class == pred_class:
                                sequential_corrects[_concept.name] += 1

                        if pred_class == label_class:
                            if label != none_index:
                                per_class_tp_sequential[_concept.name][label_class] += 1
                        else:
                            if pred_class != "None":
                                per_class_fp_sequential[_concept.name][pred_class] += 1
                            if label != none_index:
                                per_class_fn_sequential[_concept.name][label_class] += 1
                
        
        # end = time.time()
        # print(f"elapsed time for evaluating one datanode after having all results are: {end - start}")
    #### calculate the perc lass Precision, Recall, and F1 scores
    f1_res = {"level1":{}, "level2":{}, "level3":{}, "level4":{}}
    total_precision, total_recall, total_f1 = 0, 0, 0
    total_precision_ilp, total_recall_ilp, total_f1_ilp = 0, 0, 0
    total_precision_sequential, total_recall_sequential, total_f1_sequential = 0, 0, 0
    all_support = 0
    for _concept in [level1, level2, level3, level4]:
        level_precision = 0
        level_recall = 0
        level_f1 = 0

        level_precision_ilp = 0
        level_recall_ilp = 0
        level_f1_ilp = 0

        level_precision_sequential = 0
        level_recall_sequential = 0
        level_f1_sequential = 0

        for key in per_class_fn_ilp[_concept.name]:
            if key == "None":
                continue
            if per_class_tp_ilp[_concept.name][key] != 0:
                precision_ilp = per_class_tp_ilp[_concept.name][key] / (per_class_tp_ilp[_concept.name][key] + per_class_fp_ilp[_concept.name][key])
                recall_ilp = per_class_tp_ilp[_concept.name][key] / (per_class_tp_ilp[_concept.name][key] + per_class_fn_ilp[_concept.name][key])
                f1_ilp = 2 * precision_ilp * recall_ilp / (precision_ilp + recall_ilp)
            else:
                precision_ilp = 0
                recall_ilp = 0
                f1_ilp = 0

            if per_class_tp[_concept.name][key] != 0:
                precision_normal = per_class_tp[_concept.name][key] / (per_class_tp[_concept.name][key] + per_class_fp[_concept.name][key])
                recall_normal = per_class_tp[_concept.name][key] / (per_class_tp[_concept.name][key] + per_class_fn[_concept.name][key])
                f1_normal = 2 * precision_normal * recall_normal / (precision_normal + recall_normal)
            else:
                precision_normal = 0
                recall_normal = 0
                f1_normal = 0
            if calculate_sequential:
                if per_class_tp_sequential[_concept.name][key] != 0:
                    precision_sequential = per_class_tp_sequential[_concept.name][key] / (per_class_tp_sequential[_concept.name][key] + per_class_fp_sequential[_concept.name][key])
                    recall_sequential = per_class_tp_sequential[_concept.name][key] / (per_class_tp_sequential[_concept.name][key] + per_class_fn_sequential[_concept.name][key])
                    f1_sequential = 2 * precision_sequential * recall_sequential / (precision_sequential + recall_sequential)
                else:
                    precision_sequential = 0
                    recall_sequential = 0
                    f1_sequential = 0

            if key not in support_set[_concept.name]:
                support_set[_concept.name][key] = 0

            if total_support_set[_concept.name] != 0:
                level_precision += precision_normal * (support_set[_concept.name][key] / total_support_set[_concept.name])
                level_recall += recall_normal * (support_set[_concept.name][key] / total_support_set[_concept.name])
                level_f1 += f1_normal * (support_set[_concept.name][key] / total_support_set[_concept.name])

            if total_support_set[_concept.name] != 0: 
                level_precision_ilp += precision_ilp * (support_set[_concept.name][key] / total_support_set[_concept.name])
                level_recall_ilp += recall_ilp * (support_set[_concept.name][key] / total_support_set[_concept.name])
                level_f1_ilp += f1_ilp * (support_set[_concept.name][key] / total_support_set[_concept.name])

            if calculate_sequential:
                if total_support_set[_concept.name] != 0:
                    level_precision_sequential += precision_sequential * (support_set[_concept.name][key] / total_support_set[_concept.name])
                    level_recall_sequential += recall_sequential * (support_set[_concept.name][key] / total_support_set[_concept.name])
                    level_f1_sequential += f1_sequential * (support_set[_concept.name][key] / total_support_set[_concept.name])

            f1_res[_concept.name][key] = {
                "precision_ilp": precision_ilp,
                "recall_ilp": recall_ilp,
                "f1_ilp": f1_ilp,
                "precision_normal": precision_normal,
                "recall_normal": recall_normal,
                "f1_normal": f1_normal,
                # "precision_sequential": precision_sequential,
                # "recall_sequential": recall_sequential,
                # "f1_sequential": f1_sequential,
                "support": support_set[_concept.name][key],
            }
        f1_res[_concept.name]["total"] = {}
        f1_res[_concept.name]["total"]["level_precision"] = level_precision
        f1_res[_concept.name]["total"]["level_recall"] = level_recall
        f1_res[_concept.name]["total"]["level_f1"] = level_f1
        f1_res[_concept.name]["total"]["level_precision_ilp"] = level_precision_ilp
        f1_res[_concept.name]["total"]["level_recall_ilp"] = level_recall_ilp
        f1_res[_concept.name]["total"]["level_f1_ilp"] = level_f1_ilp
        # f1_res[_concept.name]["total"]["level_precision_sequential"] = level_precision_sequential
        # f1_res[_concept.name]["total"]["level_recall_sequential"] = level_recall_sequential
        # f1_res[_concept.name]["total"]["level_f1_sequential"] = level_f1_sequential
        f1_res[_concept.name]["total"]["support"] = total_support_set[_concept.name]
        all_support += total_support_set[_concept.name]

    for _concept in [level1, level2, level3, level4]:
        total_precision += f1_res[_concept.name]["total"]["level_precision"] * (total_support_set[_concept.name] / all_support)
        total_recall += f1_res[_concept.name]["total"]["level_recall"] * (total_support_set[_concept.name] / all_support)
        total_f1 += f1_res[_concept.name]["total"]["level_f1"] * (total_support_set[_concept.name] / all_support)
        total_precision_ilp += f1_res[_concept.name]["total"]["level_precision_ilp"] * (total_support_set[_concept.name] / all_support)
        total_recall_ilp += f1_res[_concept.name]["total"]["level_recall_ilp"] * (total_support_set[_concept.name] / all_support)
        total_f1_ilp += f1_res[_concept.name]["total"]["level_f1_ilp"] * (total_support_set[_concept.name] / all_support)
        if calculate_sequential:
            total_precision_sequential += f1_res[_concept.name]["total"]["level_precision_sequential"] * (total_support_set[_concept.name] / all_support)
            total_recall_sequential += f1_res[_concept.name]["total"]["level_recall_sequential"] * (total_support_set[_concept.name] / all_support)
            total_f1_sequential += f1_res[_concept.name]["total"]["level_f1_sequential"] * (total_support_set[_concept.name] / all_support)
    
    f1_res["total"] = {}
    f1_res["total"]["level_precision"] = total_precision
    f1_res["total"]["level_recall"] = total_recall
    f1_res["total"]["level_f1"] = total_f1
    f1_res["total"]["level_precision_ilp"] = total_precision_ilp
    f1_res["total"]["level_recall_ilp"] = total_recall_ilp
    f1_res["total"]["level_f1_ilp"] = total_f1_ilp
    # f1_res["total"]["level_precision_sequential"] = total_precision_sequential
    # f1_res["total"]["level_recall_sequential"] = total_recall_sequential
    # f1_res["total"]["level_f1_sequential"] = total_f1_sequential
    f1_res["total"]["support"] = all_support


    prefix = "Tasks/ImgHierarchy/"
    with open(f"{prefix}logger.json", "w") as file:
        json.dump(f1_res, file, indent=4)

    file2 = open(f"{prefix}logger_manual.txt", "w")
    for _c in corrects:
        if totals[_c] != 0:
            print(f"{_c} accuracy: {corrects[_c]/totals[_c]}", file=file2)
            print(f"{_c} ILP accuracy: {ilp_corrects[_c]/totals[_c]}", file=file2)  
            print(f"{_c} sequential accuracy: {sequential_corrects[_c]/totals[_c]}", file=file2)
            if _c in backsteps:
                print(f"{_c} consistent accuracy: {consistent_corrects[_c]/totals[_c]}", file=file2)
        else:
            print(f"no instances for {_c}", file=file2)                 
