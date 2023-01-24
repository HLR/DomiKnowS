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
from regr.program import SolverPOIProgram, SolverPOIDictLossProgram
from regr.program.lossprogram import PrimalDualProgram, SampleLossProgram
from regr.program.model.pytorch import SolverModel, SolverModelDictLoss
from regr.program.loss import NBCrossEntropyLoss, NBCrossEntropyDictLoss
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from functools import partial
from tqdm import tqdm
from sklearn.metrics import classification_report

from graph import *
from data import train_dataset, valid_dataset, train_mini_dataset
from net import SimpleLSTM

from regr import setProductionLogMode

setProductionLogMode(no_UseTimeLog=True)

# reading words and labels
word['word'] = ReaderSensor(keyword='word')

word['predicate'] = ReaderSensor(keyword='predicate')

word[tag] = ModuleLearner(word['word'], word['predicate'], module=SimpleLSTM())
word[tag] = ReaderSensor(keyword='arg_label', label=True)


def make_batch(input_words):
    return input_words.flatten().unsqueeze(0), torch.ones((1, len(input_words)))


sentence['word', sentence_contains.reversed] = JointSensor(word['word'], forward=make_batch)

sentence['spans_all'] = ReaderSensor(keyword='spans_all')

# get distribution over valid spans based on model predicted tags
def make_spans_func(arg_num):
    def span_distribution(spans_all, tags):
        result = torch.empty((tags.shape[0], num_spans))
        for sp_idx in range(num_spans):
            curr_span = spans_all[0][sp_idx]
            mask = curr_span[:, 0]
            tags_masked = torch.masked_select(tags[:, arg_num], mask > 0)
            result[:, sp_idx] = torch.sum(tags_masked)
        return result
    return span_distribution


word[span_num_1] = FunctionalSensor(sentence['spans_all'], word[tag], forward=make_spans_func(1))
word[span_num_2] = FunctionalSensor(sentence['spans_all'], word[tag], forward=make_spans_func(2))

for i, sp in enumerate(spans):
    # dummy values
    #word[sp] = FunctionalSensor(word['word'], forward=lambda x: torch.ones(len(x), 2) * 0.5)

    word['sp_label_%d' % i] = ReaderSensor(keyword='span_%d' % i)

    def manual_fixedL(x, lbl):
        result = torch.ones(len(x), 2) * -100
        for j, l in enumerate(lbl):
            result[j, int(l[0])] = 100
        return result

    # each valid span is set to the ground truth mask
    word[sp] = FunctionalSensor(word['word'], word['sp_label_%d' % i], forward=manual_fixedL)

    word[sp] = ReaderSensor(keyword='span_%d' % i, label=True)

word['spanFixed'] = FunctionalSensor(word['word'], forward=lambda x: torch.ones(len(x), 1))

'''first = torch.ones(1, num_spans) * -100
first[:, 0] = 100

second = torch.ones(1, num_spans) * -100
second[:, 1] = 100

uniform = torch.ones(1, num_spans)'''

#sentence[span_num_1] = FunctionalSensor(forward=lambda: uniform)
#sentence[span_num_2] = FunctionalSensor(forward=lambda: uniform)

class CallbackProgram(SolverPOIDictLossProgram):
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
                          dictloss={'tag': MacroAverageTracker(NBCrossEntropyDictLoss(weight=torch.tensor([1.20999499, 22.3466872, 7.76391863]))),
                                    'default': MacroAverageTracker(NBCrossEntropyDictLoss())},
                          metric={})

'''program = CallbackProgram(graph, SolverModelDictLoss,
                            poi=(sentence, word),
                            inferTypes=['local/argmax'],
                            dictloss={'tag': MacroAverageTracker(NBCrossEntropyDictLoss(weight=torch.tensor([1.20999499, 22.3466872, 7.76391863]))),
                                      'default': MacroAverageTracker(NBCrossEntropyDictLoss())},
                            metric={})'''
                            #sample=True,
                            #sampleSize=100,
                            #sampleGlobalLoss=False,
                            #beta=1)

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

        #print(all_pred[-len(word_nodes):])

    print(classification_report(all_lbl, all_pred))
    print(all_pred[:100])


def post_epoch_metrics(kwargs):
    print_scores(train_mini_dataset)
    print_scores(valid_dataset)


program.after_train_epoch = [post_epoch_metrics]

program.train(train_dataset,
              train_epoch_num=40,
              Optim=partial(torch.optim.Adam, lr=1e-3),
              test_every_epoch=True,
              device='auto',
              #c_warmup_iters=0
              )

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
