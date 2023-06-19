import argparse
import os,sys
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append(".")
sys.path.append("../..")

from transformers import AutoTokenizer, BertModel

from domiknows.program.model_program import SolverPOIProgram
from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor, FunctionalReaderSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from domiknows.program.loss import NBCrossEntropyLoss


# Enable skeleton DataNode
def main(device):
    from graph_2level import graph, news_group_contains, news, level1, level2, news_group

    news_group['input_ids'] = FunctionalReaderSensor(keyword='input_ids', forward=lambda data: data.unsqueeze(0) ,device=device)
    
    news_group['attention_mask'] = FunctionalReaderSensor(keyword='attention_mask', forward=lambda data: data.unsqueeze(0) ,device=device)
    
    class NewsRepModule(nn.Module):
        def __init__(self, roberta):
            super().__init__()
            self.roberta = roberta
            self.drop_layer = nn.Dropout(p=0.3)

        def forward(self, input_ids, attention_mask):
            output = self.roberta(input_ids=input_ids.squeeze(0), attention_mask=attention_mask.squeeze(0))
            logits = output.pooler_output
            logits = self.drop_layer(logits)
            return logits.unsqueeze(0)
        
    prefix = "Tasks/20news/models/"
    rep_model = NewsRepModule(roberta=BertModel.from_pretrained("bert-base-uncased"))

    rep_model_path = f"{prefix}rep_model2_base.pt"
    rep_model.load_state_dict(torch.load(rep_model_path, map_location=device))
    
    news_group['reps'] = ModuleLearner("input_ids", "attention_mask", module=rep_model)
    

    news[news_group_contains, "reps"] = JointSensor(news_group['reps'], forward=lambda x: (torch.ones(x.shape[1], 1), x.squeeze(0)))

    def get_label(*inputs, data):
        return data
    
    class Level2Calssification(nn.Module):
        def __init__(self,):
            super().__init__()
            self.classification = nn.Linear(768, 8)

        def forward(self, logits):
            multip_tensor = torch.tensor([
                0.7291, 0.7614, 0.6675, 0.7811, 
                0.9346, 0.9374, 0.8810, 0.9394
                ]).to(logits.device)
            _out = self.classification(logits)
            ### comment later
            _out = (torch.softmax(_out, -1) * pow(0.729, 4)) / torch.mean(torch.softmax(_out, -1), -1)[0]
            return _out
        
    class Level1Calssification(nn.Module):
        def __init__(self,):
            super().__init__()
            self.classification = nn.Linear(768, 50)
            self.relu = nn.LeakyReLU()
            self.classification2 = nn.Linear(50, 16)
            

        def forward(self, logits):
            multip_tensor = torch.tensor([
                0.6810, 0.8102, 0.8345, 0.6973, 0.7577, 0.9541, 
                0.7990, 0.0475, 0.6869,0.8835, 0.8266, 0.7986, 
                0.8609, 0.8114, 0.5521, 0.7321
                ]).to(logits.device)
            _out = self.classification(logits)
            _out = self.relu(_out)
            _out = self.classification2(_out)

            ### comment later
            _out = (torch.softmax(_out, -1) * pow(0.754, 4)) / torch.mean(torch.softmax(_out, -1), -1)[0]
            return _out
        
    level1_classification = Level1Calssification()
    level2_classification = Level2Calssification()

    level1_classification_path = f"{prefix}level1_classification2_base.pt"
    level1_classification.load_state_dict(torch.load(level1_classification_path, map_location=device))
    level2_classification_path = f"{prefix}level2_classification2_base.pt"
    level2_classification.load_state_dict(torch.load(level2_classification_path, map_location=device))
    news[level1] = ModuleLearner("reps", module=level1_classification)
    news[level1] = FunctionalReaderSensor(keyword='level1', forward=get_label, label=True)

    news[level2] = ModuleLearner("reps", module=level2_classification)
    news[level2] = FunctionalReaderSensor(keyword='level2', forward=get_label, label=True)

    prefix = "Tasks/20news/logs/"
    f = open(f"{prefix}logger_base.txt", "w")
    program = SolverPOIProgram(graph, inferTypes=[
        'ILP', 
        'local/argmax'],
        # probKey = ("local" , "meanNormalizedProb"),
                                poi = (news_group, news, level1, level2),
                                loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                 metric={
                                    'ILP': PRF1Tracker(DatanodeCMMetric()),
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
    from reader_base_2level import collate_label_set
    from graph_2level import *

    dataset = load_dataset("rungalileo/20_Newsgroups_Fixed")
    def trim_dataset(dataset, key):
        labels = {}
        for item in dataset[key]:
            if item['label'] not in labels:
                labels[item['label']] = 0
            labels[item['label']] += 1
        # print(labels)
        before_len = len(dataset[key])
        # print(before_len)
        dataset[key] = dataset[key].filter(lambda example: example['label'] and example['label'] != "None" and example['text'] and len(example["text"]) >= 10)
        after_len = len(dataset[key])
        # print(after_len)
        # print("the number of removed items : ", before_len - after_len)
        labels = {}
        for item in dataset[key]:
            if item['label'] not in labels:
                labels[item['label']] = 0
            labels[item['label']] += 1
        # print(labels)
        return dataset
    dataset = trim_dataset(dataset, "train")
    dataset = trim_dataset(dataset, "test")

    dataloader = DataLoader(dataset['train'], batch_size=24, collate_fn=collate_label_set)
    test_loader = DataLoader(dataset['test'], batch_size=200, collate_fn=collate_label_set)

    device = 'cuda:2'
    program = main(device)
    # program.train(dataloader, train_epoch_num=10, Optim=lambda param: torch.optim.AdamW(param, lr=1e-5), device=device)

    # program.test(test_loader, device=device)
    # program.test(list(iter(dataloader))[:40], device=device)

    corrects = {
        'level1': 0,
        'level2': 0,
    }

    consistent_corrects = {
        'level1': 0,
        'level2': 0,
    }

    ilp_corrects = {
        'level1': 0,
        'level2': 0,
    }

    totals = {
        'level1': 0,
        'level2': 0,
    }

    total_with_none = {
        'level1': 0,
        'level2': 0,
    }
    corrects_with_none = {
        'level1': 0,
        'level2': 0,
    }
    total_with_none_ilp = {
        'level1': 0,
        'level2': 0,
    }
    corrects_with_none_ilp = {
        'level1': 0,
        'level2': 0,
    }
    changed_total = {
        'level1': 0,
        'level2': 0,
    }
    changed_corrects = {
        'level1': 0,
        'level2': 0,
    }
    changed_wrong = {
        'level1': 0,
        'level2': 0,
    }

    backsteps = {"level2": "level1"}
    per_class_tp = {"level1": {}, "level2": {},}
    per_class_fp = {"level1": {}, "level2": {},}
    per_class_fn = {"level1": {}, "level2": {},}

    per_class_tp_ilp = {"level1": {}, "level2": {},}
    per_class_fp_ilp = {"level1": {}, "level2": {},}
    per_class_fn_ilp = {"level1": {}, "level2": {},}

    support_set = {"level1": {}, "level2": {},}
    total_support_set = {"level1": 0, "level2": 0,}
    from tqdm import tqdm
    # for datanode in tqdm(program.populate(list(iter(dataloader))[:20], device=device)):
    for datanode in tqdm(program.populate(test_loader, device=device)):
        for child in datanode.getChildDataNodes('news'):
            for _concept in [level1, level2]:
                total_with_none[_concept.name] += 1
                total_with_none_ilp[_concept.name] += 1
                if 'None' in _concept.enum:
                    none_index = _concept.enum.index('None')
                else:
                    none_index = -1
                label = child.getAttribute(_concept.name, 'label').item()
                label_class = _concept.enum[label]
                if label_class != "None":
                    total_support_set[_concept.name] += 1
                pred = child.getAttribute(_concept.name, 'local/argmax').argmax().item()
                pred_class = _concept.enum[pred]
                pred_ilp = child.getAttribute(_concept.name, 'ILP').argmax().item()
                pred_ilp_class = _concept.enum[pred_ilp]
                if pred_class == label_class:
                    corrects_with_none[_concept.name] += 1
                if pred_ilp_class == label_class:
                    corrects_with_none_ilp[_concept.name] += 1
                if label_class not in support_set[_concept.name]:
                    support_set[_concept.name][label_class] = 0
                if pred_class != pred_ilp_class and label_class != "None":
                    changed_total[_concept.name] += 1
                    if pred_ilp_class == label_class:
                        changed_corrects[_concept.name] += 1
                    elif pred_class == label_class:
                        changed_wrong[_concept.name] += 1
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
                if _concept.name == level1.name:
                    hierarchy = 0
                    prior_limit = -1
                elif _concept.name == level2.name:
                    hierarchy = hierarchy_1
                    prior_limit = 16
                
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
    f1_res = {"level1":{}, "level2":{}}
    total_precision, total_recall, total_f1 = 0, 0, 0
    total_precision_ilp, total_recall_ilp, total_f1_ilp = 0, 0, 0
    all_support = 0
    for _concept in [level1, level2]:
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

    for _concept in [level1, level2]:
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

    prefix = "Tasks/20news/logs/"
    with open(f"{prefix}logger2_base.json", "w") as file:
        json.dump(f1_res, file, indent=4)

    file2 = open(f"{prefix}logger2_manual_base.txt", "w")
    for _c in corrects:
        if totals[_c] != 0:
            print(f"{_c} accuracy: {corrects[_c]/totals[_c]}", file=file2)
            print(f"{_c} ILP accuracy: {ilp_corrects[_c]/totals[_c]}", file=file2)  
            print(f"{_c} accuracy with none: {corrects_with_none[_c]/total_with_none[_c]}", file=file2)
            print(f"{_c} ILP accuracy with none: {corrects_with_none_ilp[_c]/total_with_none_ilp[_c]}", file=file2)
            print(f"{_c} total changed for ILP is {changed_total[_c]}, correct changes are {changed_corrects[_c]}({changed_corrects[_c]/changed_total[_c]}%), wrong changes are {changed_wrong[_c]}({changed_wrong[_c]/changed_total[_c]}%)", file=file2)
            if _c in backsteps:
                print(f"{_c} consistent accuracy: {consistent_corrects[_c]/totals[_c]}", file=file2)
        else:
            print(f"no instances for {_c}", file=file2)                 
