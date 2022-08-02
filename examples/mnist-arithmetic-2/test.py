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
from regr.program import IMLProgram, SolverPOIProgram
from regr.program.callbackprogram import hook
from regr.program.lossprogram import PrimalDualProgram, SampleLossProgram
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from regr.program.model.pytorch import SolverModel
from regr.program.loss import NBCrossEntropyLoss, NBCrossEntropyIMLoss, BCEWithLogitsIMLoss
from regr import setProductionLogMode
import os

from model import build_program, NBSoftCrossEntropyIMLoss, NBSoftCrossEntropyLoss
import config

setProductionLogMode()

trainloader, trainloader_mini, validloader, testloader = get_readers()


def get_pred_from_node(node, suffix):
    digit0_pred = torch.argmax(node.getAttribute(f'<digits0>{suffix}'))
    digit1_pred = torch.argmax(node.getAttribute(f'<digits1>{suffix}'))
    summation_pred = torch.argmax(node.getAttribute(f'<summations>{suffix}'))

    return digit0_pred, digit1_pred, summation_pred


#program.populate(reader, device='auto')

def get_classification_report(program, reader, total=None, verbose=False, infer_suffixes=['/local/argmax']):
    digits_results = {
        'label': []
    }

    summation_results = {
        'label': []
    }

    for suffix in infer_suffixes:
        digits_results[suffix] = []
        summation_results[suffix] = []

    for i, (loss, metric, node) in tqdm(enumerate(program.test_epoch(reader)), total=total, position=0, leave=True):

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

        if verbose:
            for j, (digit_pred, digit_gt) in enumerate(zip(digits_results[suffix], digits_results['label'])):
                print(f'digit {j % 2}: pred {digit_pred}, gt {digit_gt}')

                if j % 2 == 1:
                    print(f'summation: pred {summation_results[suffix][j // 2]},'
                          f'gt {summation_results["label"][j // 2]}\n')

        print(classification_report(digits_results['label'], digits_results[suffix]))
        print(classification_report(summation_results['label'], summation_results[suffix]))

        print('==========================================')


graph, images = build_program()


program = SolverPOIProgram(graph,
                            poi=(images,),
                            inferTypes=['local/argmax'],
                            metric={})

# load model.pth
program.model.load_state_dict(torch.load('../../../results_new/direct_10k/epoch4/model.pth'))


# verify validation accuracy
print("validation evaluation")
get_classification_report(program, validloader, total=config.num_valid, verbose=False)

# get test accuracy
print("test evaluation")
get_classification_report(program, testloader, total=config.num_test, verbose=False)
