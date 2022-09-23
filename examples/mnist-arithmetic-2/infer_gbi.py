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
from regr.program.model.base import Mode
import os
from itertools import chain
from regr.utils import detuple
import torch.nn.functional as F

from model import build_program, NBSoftCrossEntropyIMLoss, NBSoftCrossEntropyLoss
import config

from gbi import get_lambda

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


graph, images, digit0, digit1 = build_program()


program = SolverPOIProgram(graph,
                            poi=(images,),
                            inferTypes=['local/argmax', 'local/softmax'],
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

def populate_forward(model, data_item):
    _, _, *output = model(data_item)
    node = detuple(*output[:1])
    return node


def get_constraints(node):
    verifyResult = node.verifyResultsLC()

    assert verifyResult

    satisfied_constraints = []
    for lc_idx, lc in enumerate(verifyResult):
        satisfied_constraints.append(verifyResult[lc]['satisfied'])

        # print("constraint #%d" % (lc_idx), lc + ':', verifyResult[lc]['satisfied'], 'label = %d' % curr_label)

    num_constraints = len(verifyResult)
    num_satisifed = sum(satisfied_constraints) // 100

    return num_satisifed, num_constraints


for data_iter, data_item in enumerate(validloader):
    model = program.model

    with torch.no_grad():
        node = populate_forward(model, data_item)

    # get label
    digit0_label = node.getAttribute('digit0_label').item()
    digit1_label = node.getAttribute('digit1_label').item()
    summation_label = node.getAttribute('summation_label').item()

    # get pred
    digit0_pred, digit1_pred, summation_pred = get_pred_from_node(node, '/local/argmax')

    # get constraint satisfaction
    num_satisfied, num_constraints = get_constraints(node)

    if num_satisfied == num_constraints:
        continue

    print('GT: %d + %d = %d' % (digit0_label, digit1_label, summation_label))
    print('PRED: %d + %d = %d' % (digit0_pred, digit1_pred, summation_pred))
    print('CONSTRAINTS SATISFACTION: %d/%d' % (num_satisfied, num_constraints))

    print('Starting GBI:')

    model_l, c_opt = get_lambda(model, lr=1e-1)
    model_l.mode(Mode.TRAIN)
    model_l.train()
    model_l.reset()

    for c_iter in range(100):
        c_opt.zero_grad()

        node_l = populate_forward(model_l, data_item)

        num_satisfied_l, num_constraints_l = get_constraints(node_l)

        logits = node_l.getAttribute('logits')
        log_probs = torch.sum(F.log_softmax(logits, dim=-1))

        c_loss = -1 * log_probs * num_satisfied_l

        print("iter=%d, c_loss=%d, satisfied=%d" % (c_iter, c_loss.item(), num_satisfied_l))

        if num_satisfied_l == num_constraints_l:
            print('SATISFIED')
            digit0_pred_l, digit1_pred_l, summation_pred_l = get_pred_from_node(node_l, '/local/argmax')
            print('GT: %d + %d = %d' % (digit0_label, digit1_label, summation_label))
            print('PRED: %d + %d = %d' % (digit0_pred_l, digit1_pred_l, summation_pred_l))
            break

        c_loss.backward()
        c_opt.step()

    break