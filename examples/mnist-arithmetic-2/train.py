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
from regr.program import IMLProgram, SolverPOIProgram, CallbackProgram
from regr.program.callbackprogram import hook
from regr.program.primaldualprogram import PrimalDualProgram
from regr.program.metric import MacroAverageTracker
from regr.program.model.pytorch import SolverModel
from regr.program.loss import NBCrossEntropyLoss, NBCrossEntropyIMLoss, BCEWithLogitsIMLoss

from model import build_program, NBSoftCrossEntropyIMLoss, NBSoftCrossEntropyLoss
import config


trainloader, trainloader_mini, validloader, testloader = get_readers()


def get_pred_from_node(node, suffix):
    digit0_pred = torch.argmax(node.getAttribute(f'<digits0>{suffix}'))
    digit1_pred = torch.argmax(node.getAttribute(f'<digits1>{suffix}'))
    summation_pred = torch.argmax(node.getAttribute(f'<summations>{suffix}'))

    return digit0_pred, digit1_pred, summation_pred


def get_classification_report(program, reader, total=None, verbose=False, infer_suffixes=['/ILP', '/local/argmax']):
    digits_results = {
        'label': []
    }

    summation_results = {
        'label': []
    }

    for suffix in infer_suffixes:
        digits_results[suffix] = []
        summation_results[suffix] = []

    for i, node in tqdm(enumerate(program.populate(reader, device='auto')), total=total, position=0, leave=True):
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


class PrimalDualCallbackProgram(PrimalDualProgram):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.after_train_epoch = []

    def call_epoch(self, *args, **kwargs):
        super().call_epoch(*args, **kwargs)
        self.after_train_epoch[0](kwargs['c_session']['iter']/config.num_train)


program = PrimalDualCallbackProgram(graph,
                    Model=SolverModel,
                    poi=(images,),
                    inferTypes=['local/argmax'],
                    loss=MacroAverageTracker(NBCrossEntropyLoss()))


'''class Program(CallbackProgram, IMLProgram):
    pass


program = Program(graph,
                   poi=(images,),
                   inferTypes=['local/argmax'],
                   loss=MacroAverageTracker(NBSoftCrossEntropyIMLoss(prior_weight=0.1, lmbd=0.5)))'''


epoch_num = 1


def post_epoch_metrics():
    global epoch_num

    if epoch_num % 5 == 0:
        print("train evaluation")
        get_classification_report(program, trainloader_mini, total=config.num_valid, verbose=False)

        print("validation evaluation")
        get_classification_report(program, validloader, total=config.num_valid, verbose=False)

    epoch_num += 1


def post_epoch_metrics_pd(epoch_num):
    if epoch_num % 5 == 0:
        print("train evaluation")
        get_classification_report(program, trainloader_mini, total=config.num_valid, verbose=False)

        print("validation evaluation")
        get_classification_report(program, validloader, total=config.num_valid, verbose=False)


program.after_train_epoch = [post_epoch_metrics_pd]


def test_adam(params):
    print('initializing optimizer')
    return torch.optim.Adam(params, lr=0.001)


program.train(trainloader,
              train_epoch_num=20,
              Optim=test_adam,
              device='auto')

'''for i in range(1, config.epochs + 1):
    print("EPOCH", i)

    program.train(trainloader,
              train_epoch_num=20,
              Optim=lambda x: torch.optim.Adam(x, lr=0.001),
              device='auto')


    if i == 0:
        program = IMLProgram(graph,
                             poi=(images,),
                             inferTypes=['local/argmax'],
                             loss=MacroAverageTracker(NBSoftCrossEntropyIMLoss(prior_weight=0, lmbd=0.5)))

    # validation'''

