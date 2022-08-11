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
from itertools import chain

from model import build_program, NBSoftCrossEntropyIMLoss, NBSoftCrossEntropyLoss
import config

setProductionLogMode()

trainloader, trainloader_mini, validloader, testloader = get_readers()


def get_pred_from_node(node, suffix):
    digit0_pred = torch.argmax(node.getAttribute(f'<digits0>{suffix}'))
    digit1_pred = torch.argmax(node.getAttribute(f'<digits1>{suffix}'))
    #summation_pred = torch.argmax(node.getAttribute(f'<summations>{suffix}'))

    return digit0_pred, digit1_pred, 0


#program.populate(reader, device='auto')

def get_classification_report(program, reader, total=None, verbose=False, infer_suffixes=['/local/argmax']):
    digits_results = {
        'label': []
    }

    summation_results = {
        'label': []
    }

    satisfied = {}

    satisfied_overall = []

    for suffix in infer_suffixes:
        digits_results[suffix] = []
        summation_results[suffix] = []

    for i, node in tqdm(enumerate(program.populate(reader)), total=total, position=0, leave=True):

        for suffix in infer_suffixes:
            digit0_pred, digit1_pred, summation_pred = get_pred_from_node(node, suffix)

            digits_results[suffix].append(digit0_pred)
            digits_results[suffix].append(digit1_pred)

            summation_results[suffix].append(summation_pred)

        digits_results['label'].append(node.getAttribute('digit0_label').item())
        digits_results['label'].append(node.getAttribute('digit1_label').item())
        summation_results['label'].append(node.getAttribute('<summations>/label').item())

        curr_label = node.getAttribute('<summations>/label').item()

        verifyResult = node.verifyResultsLC()
        if verifyResult:
            satisfied_constraints = []
            for lc_idx, lc in enumerate(verifyResult):
                if lc not in satisfied:
                    satisfied[lc] = []
                satisfied[lc].append(verifyResult[lc]['satisfied'])
                satisfied_constraints.append(verifyResult[lc]['satisfied'])

                #print("constraint #%d" % (lc_idx), lc + ':', verifyResult[lc]['satisfied'], 'label = %d' % curr_label)

            num_constraints = len(verifyResult)
            satisfied_overall.append(1 if num_constraints * 100 == sum(satisfied_constraints) else 0)

    for suffix in infer_suffixes:
        print('============== RESULTS FOR:', suffix, '==============')

        if verbose:
            for j, (digit_pred, digit_gt) in enumerate(zip(digits_results[suffix], digits_results['label'])):
                print(f'digit {j % 2}: pred {digit_pred}, gt {digit_gt}')

                if j % 2 == 1:
                    print(f'summation: pred {summation_results[suffix][j // 2]},'
                          f'gt {summation_results["label"][j // 2]}\n')

        print(classification_report(digits_results['label'], digits_results[suffix], digits=5))
        #print(classification_report(summation_results['label'], summation_results[suffix], digits=5))

        print('==========================================')

    #sat_values = list(chain(*satisfied.values()))
    #print('Average constraint satisfactions: %f' % (sum(sat_values)/len(sat_values)))
    print('Average constraint satisfactions: %f' % (sum(satisfied_overall)/len(satisfied_overall)))


graph, images = build_program()


program = SolverPOIProgram(graph,
                            poi=(images,),
                            inferTypes=['local/argmax'],
                            metric={})

# load model.pth
model_path = '../../../results_new/primaldual_500/epoch14/model.pth'

state_dict = torch.load(model_path)

'''
# baseline - remove summation layer
del state_dict['global/images/<summations>/modulelearner-1.lin1.weight']
del state_dict['global/images/<summations>/modulelearner-1.lin1.bias']
del state_dict['global/images/<summations>/modulelearner-1.lin2.weight']
del state_dict['global/images/<summations>/modulelearner-1.lin2.bias']
'''
program.model.load_state_dict(state_dict)

print('loaded model from %s' % model_path)

# verify validation accuracy
print("validation evaluation")
get_classification_report(program, validloader, total=config.num_valid, verbose=False)

# get test accuracy
print("test evaluation")
get_classification_report(program, testloader, total=config.num_test, verbose=False)
