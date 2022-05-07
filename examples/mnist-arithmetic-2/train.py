import sys
sys.path.append('../../')

import logging
logging.basicConfig(level=logging.INFO)

from data import get_readers
from functools import partial
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
from operator import itemgetter

from graph import digits_0, digits_1, summations, name_to_number

from model import build_program
import config

trainloader, validloader, testloader = get_readers()

#program = LearningBasedProgram(graph, Model)

def argmax(lst):
    index, _ = max(enumerate(lst), key=itemgetter(1))
    return index

def get_classification_report(program, reader, total=None, verbose=False):
    digit_pred_a = []
    sum_pred_a = []

    digit_label_a = []
    sum_label_a = []

    for i, node in tqdm(enumerate(program.populate(reader, device='auto')), total=total):
        node.inferILPResults()

        #print(node.getAttributes())

        digit0_pred = torch.argmax(node.getAttribute('<digits1>/ILP'))
        digit1_pred = torch.argmax(node.getAttribute('<digits0>/ILP'))
        summation_pred = torch.argmax(node.getAttribute('<summations>/ILP'))

        if verbose:
            print(f"PRED: {digit0_pred} + {digit1_pred} = {summation_pred}")

        digit0_label = node.getAttribute('<digits0>/label').item()
        digit1_label = node.getAttribute('<digits1>/label').item()
        summation_label = node.getAttribute('<summations>/label').item()

        if verbose:
            print(f"LABEL: {digit0_label} + {digit1_label} = {summation_label}")

        digit_pred_a.append(digit0_pred)
        digit_pred_a.append(digit1_pred)

        digit_label_a.append(digit0_label)
        digit_label_a.append(digit1_label)

        sum_pred_a.append(summation_pred)
        sum_label_a.append(summation_label)

    print(classification_report(digit_label_a, digit_pred_a))
    print(classification_report(sum_label_a, sum_pred_a))


program = build_program()

#get_classification_report(program, validloader, total=config.num_valid, verbose=True)

for i in range(10):
    print("EPOCH", i)

    program.train(trainloader,
              train_epoch_num=1,
              Optim=lambda p: torch.optim.Adam(p, lr=0.001),
              device='auto')

    # validation
    get_classification_report(program, validloader, total=config.num_valid, verbose=False)