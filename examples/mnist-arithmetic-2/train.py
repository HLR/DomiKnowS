import sys
sys.path.append('../../')

import logging
logging.basicConfig(level=logging.INFO)

from data import get_readers
from functools import partial
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report

from model import build_program, sum_func
import config

trainloader, validloader, testloader = get_readers()

#program = LearningBasedProgram(graph, Model)

def get_classification_report(program, reader, total=None, verbose=False):
    pred_all = []
    label_all = []

    for node in tqdm(program.populate(reader, device='auto'), total=total):
        node.inferILPResults()

        addition = node.getRelationLinks()['addition'][0]

        operands = addition.getRelationLinks()
        operand1 = operands['operand1'][0]
        operand2 = operands['operand2'][0]

        distr1 = operand1.getAttribute('<digit>/ILP')
        distr2 = operand2.getAttribute('<digit>/ILP')

        pred_digit_1 = torch.argmax(distr1)
        pred_digit_2 = torch.argmax(distr2)
        pred_sum = torch.argmax(sum_func(torch.unsqueeze(distr1, dim=0), torch.unsqueeze(distr2, dim=0)))

        label_sum = addition.getAttribute('<summation>/label')

        pred_all.append(pred_sum.item())
        label_all.append(label_sum)

        if verbose:
            print('pred:', pred_digit_1, pred_digit_2, pred_sum, 'label:', label_sum)

    return classification_report(label_all, pred_all)

program = build_program()

#print(get_classification_report(program, validloader, total=config.num_valid))

for i in range(10):
    print("EPOCH", i)

    program.train(trainloader,
              valid_set=validloader,
              test_set=testloader,
              train_epoch_num=1,
              Optim=partial(torch.optim.SGD,
                            lr=config.lr),
              device='auto')

    print(get_classification_report(program, validloader, total=config.num_valid))