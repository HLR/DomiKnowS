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

# build configs from command line args
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, choices=['Sampling', 'Semantic', 'PrimalDual', 'Explicit', 'DigitLabel', 'Baseline'], default='PrimalDual')
parser.add_argument('--checkpoint_path', type=str, default='checkpoints/primaldual_10k.pth')
parser.add_argument('--log', type=str, default='None', choices=['None', 'TimeOnly', 'All'])
parser.add_argument('--cuda', default=False, action='store_true')
parser.add_argument('--ILP', default=False, action='store_true')
parser.add_argument('--no_fixedL', default=False, action='store_true')

args = parser.parse_args()

print(args)

model_name = args.model_name
checkpoint_path = args.checkpoint_path
device = 'cuda' if args.cuda else 'cpu'

if args.log == 'None':
    setProductionLogMode(no_UseTimeLog=True)
elif args.log == 'TimeOnly':
    setProductionLogMode(no_UseTimeLog=False)

# import data
trainloader, trainloader_mini, validloader, testloader = get_readers(0)


def get_pred_from_node(node, suffix):
    pair_node = node.findDatanodes(select='pair')[0]
    digit0_node = node.findDatanodes(select='image')[0]
    digit1_node = node.findDatanodes(select='image')[1]

    #print(digit0_node.getAttributes())

    if args.cuda:
        digit0_pred = torch.argmax(digit0_node.getAttribute(f'<digits>{suffix}')).cpu()
        digit1_pred = torch.argmax(digit1_node.getAttribute(f'<digits>{suffix}')).cpu()
        summation_pred = torch.argmax(pair_node.getAttribute(f'<summations>{suffix}')).cpu()
    else:
        digit0_pred = torch.argmax(digit0_node.getAttribute(f'<digits>{suffix}'))
        digit1_pred = torch.argmax(digit1_node.getAttribute(f'<digits>{suffix}'))
        summation_pred = torch.argmax(pair_node.getAttribute(f'<summations>{suffix}'))

    return digit0_pred, digit1_pred, summation_pred


#program.populate(reader, device='auto')

def get_classification_report(program, reader, total=None, verbose=False, infer_suffixes=['/local/argmax'], print_incorrect=False):
    digits_results = {
        'label': []
    }

    summation_results = {
        'label': []
    }

    satisfied = {}

    satisfied_overall = {}

    for suffix in infer_suffixes:
        digits_results[suffix] = []
        summation_results[suffix] = []

    # iter through test data
    for i, node in enumerate(program.populate(reader, device=device)):

        # e.g. /local/argmax or /ILP
        for suffix in infer_suffixes:
            # get predictions and add to list
            digit0_pred, digit1_pred, summation_pred = get_pred_from_node(node, suffix)

            digits_results[suffix].append(digit0_pred)
            digits_results[suffix].append(digit1_pred)

            summation_results[suffix].append(summation_pred)

        # get labels and add to list
        pair_node = node.findDatanodes(select='pair')[0]
        digit0_node = node.findDatanodes(select='image')[0]
        digit1_node = node.findDatanodes(select='image')[1]

        if args.cuda:
            digits_results['label'].append(digit0_node.getAttribute('digit_label').cpu().item())
            digits_results['label'].append(digit1_node.getAttribute('digit_label').cpu().item())
            summation_results['label'].append(pair_node.getAttribute('summation_label').cpu().item())
        else:
            digits_results['label'].append(digit0_node.getAttribute('digit_label').item())
            digits_results['label'].append(digit1_node.getAttribute('digit_label').item())
            summation_results['label'].append(pair_node.getAttribute('summation_label'))

        if print_incorrect and (digits_results['/local/argmax'][-1] != digits_results['label'][-1] or digits_results['/local/argmax'][-2] != digits_results['label'][-2]):
            for suffix in infer_suffixes:
                print("%s: %d + %d = %d" % (suffix,
                                        digits_results[suffix][-1],
                                        digits_results[suffix][-2],
                                        summation_results[suffix][-1]))

            print()

        # get constraint verification stats
        for suffix in infer_suffixes:
            verifyResult = node.verifyResultsLC(key=suffix)
            if verifyResult:
                satisfied_constraints = []
                ifSatisfied_avg = 0.0
                ifSatisfied_total = 0
                for lc_idx, lc in enumerate(verifyResult):
                    # add constraint satisfaction to total list (across all samples)
                    if lc not in satisfied:
                        satisfied[lc] = []
                    satisfied[lc].append(verifyResult[lc]['satisfied'])
                    satisfied_constraints.append(verifyResult[lc]['satisfied'])

                    # build average ifSatisfied value for this single sample
                    if 'ifSatisfied' in verifyResult[lc]:
                        ifSatisfied_avg += verifyResult[lc]['ifSatisfied']
                        ifSatisfied_total += 1

                # add average ifSatisifed value to overall stats
                if suffix not in satisfied_overall:
                    satisfied_overall[suffix] = []

                satisfied_overall[suffix].append(ifSatisfied_avg / ifSatisfied_total)

                #satisfied_overall[suffix].append(1 if num_constraints * 100 == sum(satisfied_constraints) else 0)
                #pred_digit_sum = digits_results[suffix][-1] + digits_results[suffix][-2]
                #label_sum = summation_results['label'][-1]
                #satisfied_overall[suffix].append(1 if pred_digit_sum == label_sum else 0)

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
    for suffix in infer_suffixes:
        print('Average constraint satisfactions: %s - %f' % (suffix, sum(satisfied_overall[suffix])/len(satisfied_overall[suffix])))


use_digit_labels = (model_name == 'DigitLabel')

sum_setting = None
if model_name == 'Explicit':
    sum_setting = 'explicit'
elif model_name == 'Baseline':
    sum_setting = 'baseline'

graph, image, image_pair, image_batch = build_program(device=device, sum_setting=sum_setting, digit_labels=use_digit_labels, use_fixedL=not args.no_fixedL, test=True)


inferTypes = ['local/argmax']
if args.ILP:
    inferTypes.append('ILP')

program = SolverPOIProgram(graph,
                            poi=(image_batch, image, image_pair),
                            inferTypes=inferTypes,
                            metric={})

# load model.pth
model_path = checkpoint_path

state_dict = torch.load(model_path)

# remove learned linear layers for baseline model if summation isn't predicted
if model_name == 'Baseline' and not args.no_fixedL:
    remove_keys = ["global/pair/<summations>/modulelearner-1.lin1.weight", "global/pair/<summations>/modulelearner-1.lin1.bias", "global/pair/<summations>/modulelearner-1.lin2.weight", "global/pair/<summations>/modulelearner-1.lin2.bias"]

    for k in remove_keys:
        del state_dict[k]

program.model.load_state_dict(state_dict)

print('loaded model from %s' % model_path)

classification_suffixes = ['/local/argmax']
if args.ILP:
    classification_suffixes.append('/ILP')

# verify validation accuracy
print("validation evaluation")
get_classification_report(program, validloader, total=config.num_valid, verbose=False, infer_suffixes=classification_suffixes, print_incorrect=False)

# get test accuracy
print("test evaluation")
get_classification_report(program, testloader, total=config.num_test, verbose=False, infer_suffixes=classification_suffixes, print_incorrect=False)
