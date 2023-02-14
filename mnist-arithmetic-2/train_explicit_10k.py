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
import os

from model import build_program, NBSoftCrossEntropyIMLoss, NBSoftCrossEntropyLoss, get_avg_time
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

    for i, node in tqdm(enumerate(program.populate(reader, device='auto')), total=total, position=0, leave=True):

        for suffix in infer_suffixes:
            digit0_pred, digit1_pred, summation_pred = get_pred_from_node(node, suffix)

            digits_results[suffix].append(digit0_pred.cpu().item())
            digits_results[suffix].append(digit1_pred.cpu().item())

            summation_results[suffix].append(summation_pred.cpu().item())

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


graph, images = build_program(sum_setting='explicit')


class CallbackProgram(SolverPOIProgram):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.after_train_epoch = []

    def call_epoch(self, name, dataset, epoch_fn, **kwargs):
        if name == 'Testing':
            for fn in self.after_train_epoch:
                fn(kwargs)
        else:
            super().call_epoch(name, dataset, epoch_fn, **kwargs)


program = CallbackProgram(graph,
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


def post_epoch_metrics(kwargs, interval=1, train=False, valid=True):
    global epoch_num

    print("classification layer time:", str(get_avg_time()) + 'ms')

    if epoch_num % interval == 0:
        if train:
            print("train evaluation")
            get_classification_report(program, trainloader_mini, total=config.num_valid, verbose=False)

        if valid:
            print("validation evaluation")
            get_classification_report(program, validloader, total=config.num_valid, verbose=False)

    epoch_num += 1


def save_model(kwargs, interval=1, directory='checkpoints'):
    save_dir = os.path.join(directory, f'epoch{epoch_num}')

    print('saving model to', save_dir)
    if epoch_num % interval == 0:
        if os.path.isdir(save_dir):
            print("WARNING: %s already exists. Overwriting contents." % save_dir)
        else:
            os.mkdir(save_dir)

        torch.save(program.model.state_dict(), os.path.join(save_dir, 'model.pth'))
        #torch.save(program.cmodel.state_dict(), os.path.join(save_dir, 'cmodel.pth'))
        torch.save(program.opt.state_dict(), os.path.join(save_dir, 'opt.pth'))
        #torch.save(program.copt.state_dict(), os.path.join(save_dir, 'copt.pth'))

        #other_params = {}
        #other_params['c_session'] = kwargs['c_session']
        #other_params['beta'] = program.beta

        #torch.save('other_params', os.path.join(save_dir, 'other.pth'))


def load_program_inference(program, save_dir):
    program.model.load_state_dict(torch.load(os.path.join(save_dir, 'model.pth')))


program.after_train_epoch = [save_model, post_epoch_metrics]


def test_adam(params):
    print('initializing optimizer')
    return torch.optim.Adam(params, lr=0.0005)


program.train(trainloader,
              train_epoch_num=config.epochs,
              Optim=test_adam,
              device='auto',
              test_every_epoch=True)

#optim = program.model.params()


'''for i in range(1, config.epochs + 1):
    print("EPOCH", i)

    program.train(trainloader,
              train_epoch_num=1,
              Optim=test_adam,
              device='auto')

    # validation
    post_epoch_metrics_pd(i, interval=1)'''

