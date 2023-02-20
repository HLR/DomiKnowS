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
from domiknows.program import IMLProgram, SolverPOIProgram
from domiknows.program.callbackprogram import hook
from domiknows.program.lossprogram import PrimalDualProgram, SampleLossProgram
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from domiknows.program.model.pytorch import SolverModel
from domiknows.program.loss import NBCrossEntropyLoss, NBCrossEntropyIMLoss, BCEWithLogitsIMLoss
from domiknows import setProductionLogMode
from domiknows.program.model.base import Mode
import os
from itertools import chain
from domiknows.utils import detuple
import torch.nn.functional as F

from model import build_program, NBSoftCrossEntropyIMLoss, NBSoftCrossEntropyLoss
import config

from gbi import get_lambda, reg_loss

# Enable skeleton DataNode
from domiknows.utils import setDnSkeletonMode
setDnSkeletonMode(True)

setProductionLogMode()

num_train = 1
trainloader, trainloader_mini, validloader, testloader = get_readers(num_train)

def get_pred_from_node(node, suffix):
    digit0_pred = torch.argmax(node.getAttribute(f'<digits0>{suffix}'))
    digit1_pred = torch.argmax(node.getAttribute(f'<digits1>{suffix}'))
    summation_pred = torch.argmax(node.getAttribute(f'<summations>{suffix}'))

    return digit0_pred, digit1_pred, summation_pred


# program.populate(reader, device='auto')

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

                # print("constraint #%d" % (lc_idx), lc + ':', verifyResult[lc]['satisfied'], 'label = %d' % curr_label)

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
        # print(classification_report(summation_results['label'], summation_results[suffix], digits=5))

        print('==========================================')

    # sat_values = list(chain(*satisfied.values()))
    # print('Average constraint satisfactions: %f' % (sum(sat_values)/len(sat_values)))
    print('Average constraint satisfactions: %f' % (sum(satisfied_overall) / len(satisfied_overall)))


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
    """
    Forward pass through torch model.
    Returns DataNode and DataNodeBuilder.
    """
    _, _, *output = model(data_item)
    node = detuple(*output[:1])
    return node, output[1]


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


def is_correct_digits(node):
    # get label
    digit0_label = node.getAttribute('digit0_label').item()
    digit1_label = node.getAttribute('digit1_label').item()
    summation_label = node.getAttribute('summation_label').item()

    # get pred
    digit0_pred, digit1_pred, summation_pred = get_pred_from_node(node, '/local/argmax')

    return digit0_label == digit0_pred and digit1_label == digit1_pred


def run_gbi(program, dataloader, data_iters, gbi_iters, label_names, is_correct):
    """
    Runs gradient-based inference on program. Prints pre- and post- accuracy/constraint violations.
    data_iters: number of datapoints to test in validloader
    gbi_iters: number of gradient based inference optimization steps
    label_names: names of concepts used to get log probabilities from
    is_correct: function with parameter datanode that returns whether or not the prediction is correct
    """

    total = 0
    incorrect_initial = 0
    unsatisfied_initial = 0
    incorrect_after = 0
    unsatisfied_after = 0

    for data_iter, data_item in enumerate(dataloader):
        total += 1

        # end early based on number of test samples
        if total > data_iters:
            break

        model = program.model

        # forward pass through model
        with torch.no_grad():
            node, _ = populate_forward(model, data_item)

        # get constraint satisfaction
        num_satisfied, num_constraints = get_constraints(node)

        if not is_correct(node):
            incorrect_initial += 1

        if num_satisfied == num_constraints:
            continue

        unsatisfied_initial += 1

        print('INDEX: %d' % data_iter)
        print('CONSTRAINTS SATISFACTION: %d/%d' % (num_satisfied, num_constraints))

        print('Starting GBI:')

        # make copy of original model
        # model_l is the model that gets optimized
        model_l, c_opt = get_lambda(model, lr=1e-1)
        model_l.mode(Mode.TRAIN)
        model_l.train()
        model_l.reset()

        satisfied = False

        for c_iter in range(gbi_iters):
            c_opt.zero_grad()

            # forward pass through model_l
            node_l, builder_l = populate_forward(model_l, data_item)

            num_satisfied_l, num_constraints_l = get_constraints(node_l)

            is_satisifed = 1 if num_satisfied_l == num_constraints_l else 0

            # logits = node_l.getAttribute('logits')
            # log_probs = torch.sum(F.log_softmax(logits, dim=-1))

            # calculate global log prob from all labels
            #for ln in label_names:
            #    log_probs += torch.sum(torch.log(node_l.getAttribute('<%s>/local/softmax' % ln)))

            # collect probs from all datanodes (regular)
            '''probs = {}
            # iter through datanodes
            for dn in builder_l['dataNode']:
                dn.inferLocal()
                # find concept names
                for c in dn.collectConceptsAndRelations():
                    c_prob = dn.getAttribute('<%s>/local/softmax' % c[0].name)
                    if c_prob.grad_fn is not None:
                        probs[c[0].name] = c_prob'''

            # collect probs from all datanodes (skeleton)
            probs = {}
            for var_name, var_val in node_l.getAttribute('variableSet').items():
                if not var_name.endswith('/label') and var_val.grad_fn is not None:
                    probs[var_name] = torch.sum(F.log_softmax(var_val, dim=-1))

            # get total log prob
            log_probs = 0.0
            for c_prob in probs.values():
                log_probs += torch.sum(torch.log(c_prob))

            # constraint loss: NLL * binary satisfaction + regularization loss
            # reg loss is calculated based on L2 distance of weights between optimized model and original weights
            c_loss = -1 * log_probs * is_satisifed + reg_loss(model_l, model)

            # print("iter=%d, c_loss=%d, satisfied=%d" % (c_iter, c_loss.item(), num_satisfied_l))

            if num_satisfied_l == num_constraints_l:
                satisfied = True
                print('SATISFIED')

                if is_correct(node_l):
                    print('CORRECT')
                else:
                    incorrect_after += 1

                break

            c_loss.backward()
            c_opt.step()

        if not satisfied:
            print('NOT SATISFIED')

            unsatisfied_after += 1
            incorrect_after += 1

        print('-------------------')

    print('num samples: %d' % total)
    print('initial incorrect: %.2f' % (incorrect_initial / total))
    print('initial unsatisfied: %.2f' % (unsatisfied_initial / total))
    print('after incorrect: %.2f' % (incorrect_after / total))
    print('after unsatisifed: %.2f' % (unsatisfied_after / total))


run_gbi(program, validloader, 1000, 100, ['digits0', 'digits1'], is_correct_digits)
