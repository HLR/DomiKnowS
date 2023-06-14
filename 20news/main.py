import argparse
import os,sys
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append(".")
sys.path.append("../..")

from transformers import AutoTokenizer, RobertaModel

from domiknows.program.model_program import SolverPOIProgram
from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor, FunctionalReaderSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from domiknows.program.loss import NBCrossEntropyLoss


# Enable skeleton DataNode
def main(device):
    from graph import graph, news_group_contains, news, level1, level2, level3, news_group

    news_group['input_ids'] = FunctionalReaderSensor(keyword='input_ids', forward=lambda data: data.unsqueeze(0) ,device=device)
    
    news_group['attention_mask'] = FunctionalReaderSensor(keyword='attention_mask', forward=lambda data: data.unsqueeze(0) ,device=device)
    
    class NewsRepModule(nn.Module):
        def __init__(self, roberta):
            super().__init__()
            self.roberta = roberta

        def forward(self, input_ids, attention_mask):
            output = self.roberta(input_ids=input_ids.squeeze(0), attention_mask=attention_mask.squeeze(0))
            logits = output.pooler_output
            return logits.unsqueeze(0)
    
    rep_model = NewsRepModule(roberta=RobertaModel.from_pretrained("roberta-large"))

    # rep_model_path = ""
    # rep_model.load_state_dict(torch.load(rep_model_path))
    
    news_group['reps'] = ModuleLearner("input_ids", "attention_mask", module=rep_model)
    

    news[news_group_contains, "reps"] = JointSensor(news_group['reps'], forward=lambda x: (torch.ones(x.shape[1], 1), x.squeeze(0)))

    def get_label(*inputs, data):
        return data
    
    class Level1Calssification(nn.Module):
        def __init__(self,):
            super().__init__()
            self.classification = nn.Linear(1024, 8)

        def forward(self, logits):
            _out = self.classification(logits)
            return _out
        
    class Level2Calssification(nn.Module):
        def __init__(self,):
            super().__init__()
            self.classification = nn.Linear(1024, 15)

        def forward(self, logits):
            _out = self.classification(logits)
            return _out
        
    class Level3Calssification(nn.Module):
        def __init__(self):
            super().__init__()
            self.classification = nn.Linear(1024, 8)

        def forward(self, logits):
            _out = self.classification(logits)
            return _out
        
    level1_classification = Level1Calssification()
    level2_classification = Level2Calssification()
    level3_classification = Level3Calssification()

    # level1_classification_path = ""
    # level1_classification.load_state_dict(torch.load(level1_classification_path))
    # level2_classification_path = ""
    # level2_classification.load_state_dict(torch.load(level2_classification_path))
    # level3_classification_path = ""
    # level3_classification.load_state_dict(torch.load(level3_classification_path))

    news[level1] = ModuleLearner("reps", module=level1_classification)
    news[level1] = FunctionalReaderSensor(keyword='level1', forward=get_label, label=True)

    news[level2] = ModuleLearner("reps", module=level2_classification)
    news[level2] = FunctionalReaderSensor(keyword='level2', forward=get_label, label=True)

    news[level3] = ModuleLearner("reps", module=level3_classification)
    news[level3] = FunctionalReaderSensor(keyword='level3', forward=get_label, label=True)
    
    prefix = "Tasks/20news/"
    f = open(f"{prefix}logger.txt", "w")
    program = SolverPOIProgram(graph, inferTypes=[
        'ILP', 
        'local/argmax'],
        # probKey = ("local" , "meanNormalizedProbStd"),
                                poi = (news_group, news, level1, level2, level3),
                                loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                 metric={
                                    # 'ILP': PRF1Tracker(DatanodeCMMetric()),
                                            'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})
    return program

if __name__ == '__main__':
    from domiknows.utils import setProductionLogMode
    productionMode = True
    if productionMode:
        setProductionLogMode(no_UseTimeLog=True)
    from domiknows.utils import setDnSkeletonMode
    setDnSkeletonMode(True)
    import logging
    logging.basicConfig(level=logging.ERROR)

    from torch.utils.data import Dataset, DataLoader
    from datasets import load_dataset
    from reader import collate_label_set
    from graph import *

    dataset = load_dataset("rungalileo/20_Newsgroups_Fixed")
    dataloader = DataLoader(dataset['train'], batch_size=24, pin_memory=True, collate_fn=collate_label_set)
    test_loader = DataLoader(dataset['test'], batch_size=12, pin_memory=True, collate_fn=collate_label_set)

    device = 'cuda:1'
    program = main(device)
    # program.train(dataloader, train_epoch_num=10, Optim=lambda param: torch.optim.AdamW(param, lr=1e-5), device=device)

    # program.test(dataloader, device=device)
    # program.test(list(iter(dataloader))[:40], device=device)

    corrects = {
        'level1': 0,
        'level2': 0,
        'level3': 0,
    }

    consistent_corrects = {
        'level1': 0,
        'level2': 0,
        'level3': 0,
    }

    ilp_corrects = {
        'level1': 0,
        'level2': 0,
        'level3': 0,
    }

    totals = {
        'level1': 0,
        'level2': 0,
        'level3': 0,
    }

    backsteps = {'level3': 'level2'}
    per_class_tp = {"level1": {}, "level2": {}, "level3": {}}
    per_class_fp = {"level1": {}, "level2": {}, "level3": {}}
    per_class_fn = {"level1": {}, "level2": {}, "level3": {}}

    per_class_tp_ilp = {"level1": {}, "level2": {}, "level3": {}}
    per_class_fp_ilp = {"level1": {}, "level2": {}, "level3": {}}
    per_class_fn_ilp = {"level1": {}, "level2": {}, "level3": {}}

    support_set = {"level1": {}, "level2": {}, "level3": {}}
    total_support_set = {"level1": 0, "level2": 0, "level3": 0}
    from tqdm import tqdm
    # for datanode in tqdm(program.populate(list(iter(dataloader))[:20], device=device)):
    for datanode in tqdm(program.populate(dataloader, device=device)):
        for child in datanode.getChildDataNodes('news'):
            pred = child.getAttribute('level1', 'local/argmax').argmax().item()
            pred_class = level1.enum[pred]
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
            if pred_ilp == label:
                ilp_corrects['level1'] += 1
                per_class_tp_ilp['level1'][label_class] += 1
            else:
                per_class_fp_ilp['level1'][pred_ilp_class] += 1
                per_class_fn_ilp['level1'][label_class] += 1
            
            totals['level1'] += 1
            for _concept in [level2, level3]:
                none_index = _concept.enum.index('None')
                label = child.getAttribute(_concept.name, 'label').item()
                label_class = _concept.enum[label]
                if label_class != "None":
                    total_support_set[_concept.name] += 1
                pred = child.getAttribute(_concept.name, 'local/argmax').argmax().item()
                pred_class = _concept.enum[pred]
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
                if pred_ilp_class not in per_class_tp[_concept.name]:
                    per_class_tp[_concept.name][pred_ilp_class] = 0
                    per_class_fp[_concept.name][pred_ilp_class] = 0
                    per_class_fn[_concept.name][pred_ilp_class] = 0
                    per_class_tp_ilp[_concept.name][pred_ilp_class] = 0
                    per_class_fp_ilp[_concept.name][pred_ilp_class] = 0
                    per_class_fn_ilp[_concept.name][pred_ilp_class] = 0

                support_set[_concept.name][label_class] += 1
                if _concept == level2:
                    hierarchy = hierarchy_1
                    prior_limit = 7
                else:
                    hierarchy = hierarchy_2
                    prior_limit = 14

                if label != none_index:
                    totals[_concept.name] += 1
                    if _concept.name in backsteps:
                        if prior != prior_limit:
                            if label == pred:
                                consistent_corrects[_concept.name] += 1
                    if label == pred:
                        corrects[_concept.name] += 1
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
                
                if pred_ilp_class == label_class:
                    if label != none_index:
                        per_class_tp_ilp[_concept.name][label_class] += 1
                else:
                    if pred_ilp != none_index: 
                        per_class_fp_ilp[_concept.name][pred_ilp_class] += 1
                    if label != none_index:
                        per_class_fn_ilp[_concept.name][label_class] += 1

    #### calculate the perc lass Precision, Recall, and F1 scores
    f1_res = {"level1":{}, "level2":{}, "level3":{}}
    total_precision, total_recall, total_f1 = 0, 0, 0
    total_precision_ilp, total_recall_ilp, total_f1_ilp = 0, 0, 0
    all_support = 0
    for _concept in [level1, level2, level3]:
        level_precision = 0
        level_recall = 0
        level_f1 = 0

        level_precision_ilp = 0
        level_recall_ilp = 0
        level_f1_ilp = 0

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

            f1_res[_concept.name][key] = {
                "precision_ilp": precision_ilp,
                "recall_ilp": recall_ilp,
                "f1_ilp": f1_ilp,
                "precision_normal": precision_normal,
                "recall_normal": recall_normal,
                "f1_normal": f1_normal,
                "support": support_set[_concept.name][key],
            }
        f1_res[_concept.name]["total"] = {}
        f1_res[_concept.name]["total"]["level_precision"] = level_precision
        f1_res[_concept.name]["total"]["level_recall"] = level_recall
        f1_res[_concept.name]["total"]["level_f1"] = level_f1
        f1_res[_concept.name]["total"]["level_precision_ilp"] = level_precision_ilp
        f1_res[_concept.name]["total"]["level_recall_ilp"] = level_recall_ilp
        f1_res[_concept.name]["total"]["level_f1_ilp"] = level_f1_ilp
        f1_res[_concept.name]["total"]["support"] = total_support_set[_concept.name]
        all_support += total_support_set[_concept.name]

    for _concept in [level1, level2, level3]:
        total_precision += f1_res[_concept.name]["total"]["level_precision"] * (total_support_set[_concept.name] / all_support)
        total_recall += f1_res[_concept.name]["total"]["level_recall"] * (total_support_set[_concept.name] / all_support)
        total_f1 += f1_res[_concept.name]["total"]["level_f1"] * (total_support_set[_concept.name] / all_support)
        total_precision_ilp += f1_res[_concept.name]["total"]["level_precision_ilp"] * (total_support_set[_concept.name] / all_support)
        total_recall_ilp += f1_res[_concept.name]["total"]["level_recall_ilp"] * (total_support_set[_concept.name] / all_support)
        total_f1_ilp += f1_res[_concept.name]["total"]["level_f1_ilp"] * (total_support_set[_concept.name] / all_support)
    
    f1_res["total"] = {}
    f1_res["total"]["level_precision"] = total_precision
    f1_res["total"]["level_recall"] = total_recall
    f1_res["total"]["level_f1"] = total_f1
    f1_res["total"]["level_precision_ilp"] = total_precision_ilp
    f1_res["total"]["level_recall_ilp"] = total_recall_ilp
    f1_res["total"]["level_f1_ilp"] = total_f1_ilp
    f1_res["total"]["support"] = all_support

    import json

    prefix = "Tasks/20news/"
    with open(f"{prefix}logger.json", "w") as file:
        json.dump(f1_res, file, indent=4)

    file2 = open(f"{prefix}logger_manual.txt", "w")
    for _c in corrects:
        if totals[_c] != 0:
            print(f"{_c} accuracy: {corrects[_c]/totals[_c]}", file=file2)
            print(f"{_c} ILP accuracy: {ilp_corrects[_c]/totals[_c]}", file=file2)  
            if _c in backsteps:
                print(f"{_c} consistent accuracy: {consistent_corrects[_c]/totals[_c]}", file=file2)
        else:
            print(f"no instances for {_c}", file=file2)                 
