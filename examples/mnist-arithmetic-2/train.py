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


def get_pred_from_node(node, suffix):
    digit0_pred = torch.argmax(node.getAttribute(f'<digits0>{suffix}'))
    digit1_pred = torch.argmax(node.getAttribute(f'<digits1>{suffix}'))
    summation_pred = torch.argmax(node.getAttribute(f'<summations>{suffix}'))

    return digit0_pred, digit1_pred, summation_pred


def get_classification_report(program, reader, total=None):
    digits_results = {
        'label': []
    }

    summation_results = {
        'label': []
    }

    infer_suffixes = ['/ILP', '/local/argmax']

    for suffix in infer_suffixes:
        digits_results[suffix] = []
        summation_results[suffix] = []

    for i, node in tqdm(enumerate(program.populate(reader, device='auto')), total=total):
        node.inferILPResults()

        for suffix in infer_suffixes:
            digit0_pred, digit1_pred, summation_pred = get_pred_from_node(node, suffix)

            digits_results[suffix].append(digit0_pred)
            digits_results[suffix].append(digit1_pred)

            summation_results[suffix].append(summation_pred)

        digits_results['label'].append(node.getAttribute('digit0_label').item())
        digits_results['label'].append(node.getAttribute('digit1_label').item())
        summation_results['label'].append(node.getAttribute('<summations>/label').item())

    for suffix in infer_suffixes:
        print('============== RESULTS FOR:', suffix, '==============')

        print(classification_report(digits_results['label'], digits_results[suffix]))
        print(classification_report(summation_results['label'], summation_results[suffix]))

        print('==========================================')


program = build_program()

#get_classification_report(program, validloader, total=config.num_valid)

for i in range(1, 11):
    print("EPOCH", i)

    program.train(trainloader,
              train_epoch_num=1,
              Optim=lambda x: torch.optim.Adam(x, lr=0.001),
              device='auto')

    # validation
    get_classification_report(program, validloader, total=config.num_valid)
