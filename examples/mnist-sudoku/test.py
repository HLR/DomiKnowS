import sys

sys.path.append('../../')

import logging

logging.basicConfig(level=logging.INFO)

from regr.program import SolverPOIProgram
from regr.program.metric import MacroAverageTracker
from regr.program.loss import NBCrossEntropyLoss
from regr.program.lossprogram import SampleLossProgram, PrimalDualProgram
from regr.program.model.pytorch import SolverModel
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
import torch
import os

import config

from model import graph, sudoku, empty_entry, same_row, same_col, same_table
from data import trainloader, validloader

from regr import setProductionLogMode

setProductionLogMode()


def get_classification_report(dataloader, sample_num=10, limit=100, infer_suffixes=['/ILP', '/local/argmax']):
    pred_results = {}

    for suffix in infer_suffixes:
        pred_results[suffix] = []

    labels_all = []

    for d_idx, (loss, metric, node) in tqdm(enumerate(program.test_epoch(dataloader)),
                                            total=min(limit, len(dataloader)), position=0, leave=True):
        if d_idx >= limit:
            break

        node.inferILPResults()

        labels = node.getAttribute('labels')
        labels_all.extend(labels.tolist()[0])

        for suffix in infer_suffixes:
            preds = torch.empty((config.size ** 2))

            for i, entry in enumerate(node.getChildDataNodes(conceptName=empty_entry)):
                entry_pred = torch.argmax(entry.getAttribute('<empty_entry_label>' + suffix), dim=0)

                preds[i] = entry_pred

            pred_results[suffix].extend(preds.tolist())

            if d_idx < sample_num:
                print(suffix)

                print(preds.reshape(config.size, config.size))

                print(labels.reshape(config.size, config.size))

                print('\n')

        if d_idx < sample_num:
            print("===========================")

    for suffix, preds in pred_results.items():
        print("========= %s =========" % suffix)
        print(classification_report(labels_all, preds))
        print("===========================")


class CallbackProgram(SampleLossProgram):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.after_train_epoch = []

    def call_epoch(self, name, dataset, epoch_fn, **kwargs):
        if name == 'Testing':
            for fn in self.after_train_epoch:
                fn(kwargs)
        else:
            super().call_epoch(name, dataset, epoch_fn, **kwargs)


program = CallbackProgram(graph, SolverModel,
                          poi=(sudoku, empty_entry, same_row, same_col, same_table),
                          inferTypes=['local/argmax'],
                          sample=True,
                          sampleSize=100,
                          sampleGlobalLoss=True,
                          beta=1)

program.model.load_state_dict(torch.load('checkpoints/epoch1/model.pth'))

get_classification_report(validloader)
