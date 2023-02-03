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
import argparse

from model import build_program, NBSoftCrossEntropyIMLoss, NBSoftCrossEntropyLoss
import config

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, choices=['Sampling', 'Semantic', 'PrimalDual', 'Explicit', 'DigitLabel', 'Baseline'], help='Method of integrating constraints')
parser.add_argument('--num_train', type=int, default=10000, help='Number of training iterations per epoch')
parser.add_argument('--log', type=str, default='None', choices=['None', 'TimeOnly', 'All'], help='None: no logs, TimeOnly: only output timing logs, All: output all logs. Logs will be found in the logs directory.')
parser.add_argument('--cuda', default=False, action='store_true', help='Enable CUDA for training')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for')

args = parser.parse_args()

print(args)

model_name = args.model_name
num_train = args.num_train
device = 'cuda' if args.cuda else 'cpu'

if args.log == 'None':
    setProductionLogMode(no_UseTimeLog=True)
elif args.log == 'TimeOnly':
    setProductionLogMode(no_UseTimeLog=False)

trainloader, trainloader_mini, validloader, testloader = get_readers(num_train)


def get_pred_from_node(node, suffix):
    digit0_node = node.findDatanodes(select='image')[0]
    digit1_node = node.findDatanodes(select='image')[1]

    if args.cuda:
        digit0_pred = torch.argmax(digit0_node.getAttribute(f'<digits>{suffix}')).cpu()
        digit1_pred = torch.argmax(digit1_node.getAttribute(f'<digits>{suffix}')).cpu()
    else:
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

    for suffix in infer_suffixes:
        digits_results[suffix] = []
        summation_results[suffix] = []

    for i, node in tqdm(enumerate(program.populate(reader, device=device)), total=total, position=0, leave=True):

        for suffix in infer_suffixes:
            digit0_pred, digit1_pred, summation_pred = get_pred_from_node(node, suffix)

            digits_results[suffix].append(digit0_pred.cpu().item())
            digits_results[suffix].append(digit1_pred.cpu().item())

            summation_results[suffix].append(summation_pred)

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

    for suffix in infer_suffixes:
        print('============== RESULTS FOR:', suffix, '==============')

        if verbose:
            for j, (digit_pred, digit_gt) in enumerate(zip(digits_results[suffix], digits_results['label'])):
                print(f'digit {j % 2}: pred {digit_pred}, gt {digit_gt}')

                if j % 2 == 1:
                    print(f'summation: pred {summation_results[suffix][j // 2]},'
                          f'gt {summation_results["label"][j // 2]}\n')

        print(classification_report(digits_results['label'], digits_results[suffix]))
        #print(classification_report(summation_results['label'], summation_results[suffix]))

        print('==========================================')


use_digit_labels = (model_name == 'DigitLabel')

sum_setting = None
if model_name == 'Explicit':
    sum_setting = 'explicit'
elif model_name == 'Baseline':
    sum_setting = 'baseline'

graph, image, image_pair, image_batch = build_program(device=device, sum_setting=sum_setting, digit_labels=use_digit_labels)

if model_name == 'PrimalDual':
    class PrimalDualCallbackProgram(PrimalDualProgram):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.after_train_epoch = []

        def call_epoch(self, name, dataset, epoch_fn, **kwargs):
            if name == 'Testing':
                for fn in self.after_train_epoch:
                    fn(kwargs)
            else:
                super().call_epoch(name, dataset, epoch_fn, **kwargs)


    program = PrimalDualCallbackProgram(graph, SolverModel,
                        poi=(image_batch, image, image_pair),
                        inferTypes=['local/argmax'],
                        metric={})

elif model_name == 'Sampling':
    class CallbackProgram(SampleLossProgram):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.after_train_epoch = []

        def call_epoch(self, name, dataset, epoch_fn, **kwargs):
            if name == 'Testing':
                for fn in self.after_train_epoch:
                    fn(kwargs)
            else:
                super().call_epoch(name, dataset, epoch_fn, **kwargs)


    program = CallbackProgram(graph, SolverModel,
                              poi=(image_batch, image, image_pair),
                              inferTypes=['local/argmax'],
                              metric={},
                              sample=True,
                              sampleSize=100,
                              sampleGlobalLoss=True,
                              beta=1)

elif model_name == 'Semantic':
    class CallbackProgram(SampleLossProgram):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.after_train_epoch = []

        def call_epoch(self, name, dataset, epoch_fn, **kwargs):
            if name == 'Testing':
                for fn in self.after_train_epoch:
                    fn(kwargs)
            else:
                super().call_epoch(name, dataset, epoch_fn, **kwargs)


    program = CallbackProgram(graph, SolverModel,
                              poi=(image_batch, image, image_pair),
                              inferTypes=['local/argmax'],
                              metric={},
                              sample=True,
                              sampleSize=-1,  # Semantic Sample when -1
                              sampleGlobalLoss=True,
                              beta=1)

elif model_name in ['DigitLabel', 'Explicit', 'Baseline']:
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
                              poi=(image_batch, image, image_pair),
                              inferTypes=['local/argmax'],
                              loss=MacroAverageTracker(NBCrossEntropyLoss()),
                              metric={})

'''class Program(CallbackProgram, IMLProgram):
    pass


program = Program(graph,
                   poi=(image,),
                   inferTypes=['local/argmax'],
                   loss=MacroAverageTracker(NBSoftCrossEntropyIMLoss(prior_weight=0.1, lmbd=0.5)))'''


epoch_num = 1


def post_epoch_metrics(kwargs, interval=1, train=True, valid=True):
    global epoch_num

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


if model_name == 'Semantic':
    def test_adam(params):
        print('initializing optimizer')
        return torch.optim.Adam(params, lr=0.0005)


    program.train(trainloader,
                  train_epoch_num=args.epochs,
                  Optim=test_adam,
                  device=device,
                  test_every_epoch=True,
                  c_warmup_iters=0)

elif model_name == 'Sampling':
    def test_adam(params):
        print('initializing optimizer')
        return torch.optim.Adam(params, lr=0.0005)


    program.train(trainloader,
                  train_epoch_num=args.epochs,
                  Optim=test_adam,
                  device=device,
                  test_every_epoch=True,
                  c_warmup_iters=0)

elif model_name == 'PrimalDual':
    def test_adam(params):
        print('initializing optimizer')
        return torch.optim.Adam(params, lr=0.0001)


    program.train(trainloader,
                  train_epoch_num=args.epochs,
                  Optim=test_adam,
                  device=device,
                  test_every_epoch=True,
                  c_warmup_iters=0)

elif model_name == 'DigitLabel':
    def test_adam(params):
        print('initializing optimizer')
        return torch.optim.Adam(params, lr=0.0001)


    program.train(trainloader,
                  train_epoch_num=args.epochs,
                  Optim=test_adam,
                  device=device,
                  test_every_epoch=True)

elif model_name == 'Explicit':
    def test_adam(params):
        print('initializing optimizer')
        return torch.optim.Adam(params, lr=0.0005)


    program.train(trainloader,
                  train_epoch_num=args.epochs,
                  Optim=test_adam,
                  device=device,
                  test_every_epoch=True)

elif model_name == 'Baseline' and num_train == 10000:
    def test_adam(params):
        print('initializing optimizer')
        return torch.optim.Adam(params, lr=0.001)


    program.train(trainloader,
                  train_epoch_num=args.epochs,
                  Optim=test_adam,
                  device=device,
                  test_every_epoch=True)

elif model_name == 'Baseline' and num_train == 500:
    def test_adam(params):
        print('initializing optimizer')
        return torch.optim.Adam(params, lr=0.01)


    program.train(trainloader,
                  train_epoch_num=args.epochs,
                  Optim=test_adam,
                  device=device,
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

