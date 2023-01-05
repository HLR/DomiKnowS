import torch

torch.manual_seed(0)

import random

random.seed(0)

import numpy as np

np.random.seed(0)

import sys

sys.path.append('../../')

from torch import nn
import torch.nn.functional as F
from regr.sensor.pytorch.sensors import FunctionalSensor, ReaderSensor, ConstantSensor, JointSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.relation_sensors import EdgeSensor
from regr.program import SolverPOIProgram
from regr.program.lossprogram import PrimalDualProgram
from regr.program.model.pytorch import SolverModel
from regr.program.loss import NBCrossEntropyLoss
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from functools import partial
from tqdm import tqdm
from sklearn.metrics import classification_report

from graph import *
from data import train_dataset, valid_dataset, train_mini_dataset
from net import SimpleLSTM

from regr import setProductionLogMode

setProductionLogMode(no_UseTimeLog=True)

word['word'] = ReaderSensor(keyword='word')

word['predicate'] = ReaderSensor(keyword='predicate')

word[tag] = ModuleLearner(word['word'], word['predicate'], module=SimpleLSTM())
word[tag] = ReaderSensor(keyword='arg_label', label=True)


def make_batch(input_words):
    return input_words.flatten().unsqueeze(0), torch.ones((1, len(input_words)))


sentence['word', sentence_contains.reversed] = JointSensor(word['word'], forward=make_batch)

for i, sp in enumerate(spans):
    # dummy values
    word[sp] = FunctionalSensor(word['word'], forward=lambda x: torch.ones(len(x), 2) * 0.5)

    word[sp] = ReaderSensor(keyword='span_%d' % i, label=True)

word['spanFixed'] = FunctionalSensor(word['word'], forward=lambda x: torch.ones(len(x), 1))

sentence[span_num] = FunctionalSensor(forward=lambda: torch.ones(1, num_spans))


class CallbackProgram(SolverPOIProgram):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.after_train_epoch = []

    def call_epoch(self, name, dataset, epoch_fn, **kwargs):
        if name == 'Testing':
            for fn in self.after_train_epoch:
                fn(kwargs)
        else:
            super().call_epoch(name, dataset, epoch_fn, **kwargs)


program = CallbackProgram(graph,
                          poi=(sentence, word),
                          inferTypes=['local/argmax'],
                          metric={})


def print_scores(dataset):
    all_lbl = []
    all_pred = []

    for i, node in tqdm(enumerate(program.populate(dataset, device='auto')), total=len(dataset), position=0,
                        leave=True):
        word_nodes = node.findDatanodes(select='word')
        for wn in word_nodes:
            lbl = wn.getAttribute('<tag>/label')[0]
            pred = torch.argmax(wn.getAttribute('<tag>/local/argmax'))

            all_lbl.append(lbl)
            all_pred.append(pred)

    print(classification_report(all_lbl, all_pred))


def post_epoch_metrics(kwargs):
    print_scores(train_mini_dataset)
    print_scores(valid_dataset)


program.after_train_epoch = [post_epoch_metrics]

program.train(train_dataset,
              train_epoch_num=10,
              Optim=partial(torch.optim.Adam, lr=1e-5),
              test_every_epoch=True,
              device='auto')

'''program = PrimalDualProgram(graph, SolverModel,
                            poi=(sentence, word),
                            inferTypes=['local/argmax'],
                            loss=MacroAverageTracker(NBCrossEntropyLoss()),
                            metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})

program.train(train_dataset,
              train_epoch_num=10,
              Optim=partial(torch.optim.Adam, lr=1e-5),
              device='auto',
              c_warmup_iters=0)'''
