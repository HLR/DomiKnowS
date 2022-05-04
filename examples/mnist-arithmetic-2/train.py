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

from model import build_program, sum_func
import config

trainloader, validloader, testloader = get_readers()

#program = LearningBasedProgram(graph, Model)

def argmax(lst):
    index, _ = max(enumerate(lst), key=itemgetter(1))
    return index

def get_classification_report(program, reader, total=None, verbose=False):
    pred_all = []
    label_all = []

    for node in tqdm(program.populate(reader, device='auto'), total=total):
        node.inferILPResults()

        suffix = "/ILP"
        idx = 0

        pred_digit_0_distr = list(map(lambda nm: node.getAttribute("<" + nm + f">{suffix}")[idx], digits_0))
        pred_digit_1_distr = list(map(lambda nm: node.getAttribute("<" + nm + f">{suffix}")[idx], digits_1))
        pred_sum_distr = list(map(lambda nm: node.getAttribute("<" + nm + f">{suffix}")[idx], summations))

        pred_digit_0 = argmax(pred_digit_0_distr)
        pred_digit_1 = argmax(pred_digit_1_distr)
        pred_sum = argmax(pred_sum_distr)

        label = node.getAttribute('label').item()

        if verbose:
            print('pred:', pred_digit_0, pred_digit_1, pred_sum, 'label:', label)

        label_all.append(label)
        pred_all.append(pred_sum)

    return classification_report(label_all, pred_all)


program = build_program()

# print(get_classification_report(program, validloader, total=config.num_valid, verbose=True))

for i in range(10):
    print("EPOCH", i)

    program.train(trainloader,
              train_epoch_num=1,
              Optim=partial(torch.optim.SGD,
                            lr=config.lr),
              device='auto')

    # validation
    print(get_classification_report(program, trainloader, total=config.num_train, verbose=True))