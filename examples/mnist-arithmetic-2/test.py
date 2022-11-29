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
import argparse

from model import build_program, NBSoftCrossEntropyIMLoss, NBSoftCrossEntropyLoss
import config

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, choices=['Sampling', 'Semantic', 'PrimalDual', 'Explicit', 'DigitLabel', 'Baseline'])
parser.add_argument('--checkpoint_path', type=str, required=True)
parser.add_argument('--log', default=False, action='store_true')
parser.add_argument('--cuda', default=False, action='store_true')

args = parser.parse_args()

print(args)

model_name = args.model_name
checkpoint_path = args.checkpoint_path
no_log = not args.log
device = 'cuda' if args.cuda else 'cpu'

if no_log:
    setProductionLogMode(no_UseTimeLog=True)

trainloader, trainloader_mini, validloader, testloader = get_readers(0)


def get_pred_from_node(node, suffix):
    digit0_node = node.findDatanodes(select='image')[0]
    digit1_node = node.findDatanodes(select='image')[1]

    #print(digit0_node.getAttributes())

    digit0_pred = torch.argmax(digit0_node.getAttribute(f'<digits>{suffix}'))
    digit1_pred = torch.argmax(digit1_node.getAttribute(f'<digits>{suffix}'))
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

        pair_node = node.findDatanodes(select='pair')[0]
        digit0_node = node.findDatanodes(select='image')[0]
        digit1_node = node.findDatanodes(select='image')[1]

        digits_results['label'].append(digit0_node.getAttribute('digit_label').item())
        digits_results['label'].append(digit1_node.getAttribute('digit_label').item())
        summation_results['label'].append(pair_node.getAttribute('summation_label'))

        verifyResult = node.verifyResultsLC()
        if verifyResult:
            satisfied_constraints = []
            for lc_idx, lc in enumerate(verifyResult):
                if lc not in satisfied:
                    satisfied[lc] = []
                satisfied[lc].append(verifyResult[lc]['satisfied'])
                satisfied_constraints.append(verifyResult[lc]['satisfied'])

                #print("constraint #%d" % (lc_idx), lc + ':', verifyResult[lc]['satisfied'], 'label = %d' % summation_results['label'][-1])

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


use_digit_labels = (model_name == 'DigitLabel')

sum_setting = None
if model_name == 'Explicit':
    sum_setting = 'explicit'
elif model_name == 'Baseline':
    sum_setting = 'baseline'

graph, image, image_pair, image_batch = build_program(device=device, sum_setting=sum_setting, digit_labels=use_digit_labels)


program = SolverPOIProgram(graph,
                            poi=(image_batch, image, image_pair),
                            inferTypes=['local/argmax', 'ILP'],
                            metric={})

# load model.pth
model_path = checkpoint_path

state_dict = torch.load(model_path)

'''if model_name == 'baseline':
    # remove summation layer
    del state_dict['global/images/<summations>/modulelearner-1.lin1.weight']
    del state_dict['global/images/<summations>/modulelearner-1.lin1.bias']
    del state_dict['global/images/<summations>/modulelearner-1.lin2.weight']
    del state_dict['global/images/<summations>/modulelearner-1.lin2.bias']'''

program.model.load_state_dict(state_dict)

print('loaded model from %s' % model_path)

# verify validation accuracy
print("validation evaluation")
get_classification_report(program, validloader, total=config.num_valid, verbose=False, infer_suffixes=['/local/argmax', '/ILP'])

# get test accuracy
print("test evaluation")
get_classification_report(program, testloader, total=config.num_test, verbose=False, infer_suffixes=['/local/argmax', '/ILP'])
