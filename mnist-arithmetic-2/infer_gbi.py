import sys

sys.path.append('../../')

import logging

logging.basicConfig(level=logging.INFO)

from data import get_readers
import torch
from domiknows.program import SolverPOIProgram, GBIProgram
from domiknows import setProductionLogMode
from domiknows.program.model.base import Mode
from domiknows.program.model_program import SolverModel
from domiknows.program.lossprogram import LossProgram
from domiknows.utils import detuple
import torch.nn.functional as F
import argparse
from tqdm import tqdm

from model import build_program
from graph import graph, image_batch, image, image_pair
import config

from gbi import get_lambda, reg_loss

# Enable skeleton DataNode
from domiknows.utils import setDnSkeletonMode
setDnSkeletonMode(True)

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=False, action='store_true', help='Enable CUDA')
parser.add_argument('--pytorch-gbi', default=False, action='store_true', help='Use the Pytorch implementation of GBI')
parser.add_argument('--num-samples', default=1000, type=int, help='Number of samples to do inference on.')

# gbi hyperparameters
parser.add_argument('--gbi-iters', default=100, type=int, help='Number of GBI iterations')
parser.add_argument('--lr', default=1e-1, type=float, help='Learning rate for GBI')
parser.add_argument('--reg-weight', default=1, type=float, help='Weight for regularization loss')

# enable training mode
parser.add_argument('--training', default=False, action='store_true', help='Use GBI to update model parameters. Will output a trained checkpoint file.')

args = parser.parse_args()

device = 'cuda' if args.cuda else 'cpu'

setProductionLogMode()

trainloader, trainloader_mini, validloader, testloader = get_readers(args.num_samples)


def get_pred_from_node(node, suffix):
    """
    Get digit and summation value prediction from datanodes
    If the model doesn't predict a summation value, then a dummy value is returned
    """
    # pair_node = node.findDatanodes(select='pair')[0]
    # digit0_node = node.findDatanodes(select='image')[0]
    # digit1_node = node.findDatanodes(select='image')[1]

    # if args.cuda:
    #     digit0_pred = torch.argmax(digit0_node.getAttribute(f'<digits>{suffix}')).cpu()
    #     digit1_pred = torch.argmax(digit1_node.getAttribute(f'<digits>{suffix}')).cpu()
    #     summation_pred = torch.argmax(pair_node.getAttribute(f'<summations>{suffix}')).cpu()
    # else:
    #     print(pair_node.getAttributes().keys())
    #     print(f'<summations>{suffix}')

    #     digit0_pred = torch.argmax(digit0_node.getAttribute(f'<digits>{suffix}'))
    #     digit1_pred = torch.argmax(digit1_node.getAttribute(f'<digits>{suffix}'))
    #     summation_pred = torch.argmax(pair_node.getAttribute(f'<summations>{suffix}'))

    digit_id = f'image/<digits>{suffix}'
    summation_id = f'pair/<summations>{suffix}'

    # get predictions from node
    digits = node.getAttribute(digit_id)
    sum_vals = node.getAttribute(summation_id)

    # if no GBI results, fallback on argmax prediction
    if suffix == '/GBI' and digits is None:
        digits = node.getAttribute(f'image/<digits>/local/argmax')
        sum_vals = node.getAttribute(f'pair/<summations>/local/argmax')
    
    digit0_pred, digit1_pred = digits
    summation_pred = sum_vals

    digit0_pred = torch.argmax(digit0_pred)
    digit1_pred = torch.argmax(digit1_pred)
    summation_pred = torch.argmax(summation_pred)

    if args.cuda:
        digit0_pred = digit0_pred.cpu()
        digit1_pred = digit1_pred.cpu()
        summation_pred = summation_pred.cpu()

    return digit0_pred, digit1_pred, summation_pred


graph, image, image_pair, image_batch = build_program(device=device, test=True)

if args.pytorch_gbi:
    program = SolverPOIProgram(
        graph,
        poi=(image_batch, image, image_pair),
        inferTypes=['local/argmax', 'local/softmax'],
        metric={}
    )

else:
    program = GBIProgram(
        graph,
        SolverModel,
        poi=(image_batch, image, image_pair),
        inferTypes=['local/argmax', 'local/softmax', 'GBI'],
        metric={},
        gbi_iters=args.gbi_iters,
        lr=args.lr,
        reg_weight=args.reg_weight,
        reset_params=(not args.training)
    )

# load model.pth
model_path = 'checkpoints/primaldual_500.pth'
# model_path = 'gbi_model_100.pth'

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


def get_constraints_satisfaction(node, suffix='/local/argmax'):
    """
    Get constraint satisfaction from datanode
    Returns number of satisfied constraints and total number of constraints
    """
    verifyResult = node.verifyResultsLC(key=suffix)

    assert verifyResult

    satisfied_constraints = []
    for lc_idx, lc in enumerate(verifyResult):
        satisfied_constraints.append(verifyResult[lc]['satisfied'])

        # print("constraint #%d" % (lc_idx), lc + ':', verifyResult[lc]['satisfied'], 'label = %d' % curr_label)

    num_constraints = len(verifyResult)
    num_satisifed = sum(satisfied_constraints) // 100

    return num_satisifed, num_constraints


def are_both_digits_correct(node, suffix='/local/argmax'):
    """
    Returns boolean indicating whether datanode contains a correct prediction for both digits or not
    """

    # get label
    # pair_node = node.findDatanodes(select='pair')[0]
    # digit0_node = node.findDatanodes(select='image')[0]
    # digit1_node = node.findDatanodes(select='image')[1]
    
    # if args.cuda:
    #     digit0_label = digit0_node.getAttribute('digit_label').cpu().item()
    #     digit1_label = digit1_node.getAttribute('digit_label').cpu().item()
    # else:
    #     digit0_label = digit0_node.getAttribute('digit_label').item()
    #     digit1_label = digit1_node.getAttribute('digit_label').item()

    digit0_label, digit1_label = node.getAttribute('image/digit_label')

    if args.cuda:
        digit0_label = digit0_label.cpu().item()
        digit1_label = digit1_label.cpu().item()

    # get pred
    digit0_pred, digit1_pred, _ = get_pred_from_node(node, suffix)

    return digit0_label == digit0_pred and digit1_label == digit1_pred


def run_gbi(program, dataloader, data_iters, gbi_iters, label_names, is_correct):
    """
    Runs gradient-based inference on program. Prints pre- and post- accuracy/constraint violations.
    data_iters: number of datapoints to test in validloader
    gbi_iters: number of gradient based inference optimization steps
    label_names: names of concepts used to get log probabilities from
    is_correct: function with parameter datanode that returns whether or not the prediction is correct

    returns: list of sample indices that initially did not satisfy constraints
    """

    unsat_initial = []

    total = 0
    incorrect_initial = 0
    unsatisfied_initial = 0
    incorrect_after = 0
    unsatisfied_after = 0

    for data_iter, data_item in enumerate(dataloader, total=min(len(dataloader), data_iters)):
        if data_iter >= data_iters:
            break

        total += 1

        # end early based on number of test samples
        if total > data_iters:
            break

        model = program.model
        
        # --- Forward pass through model
        with torch.no_grad():
            node, _ = populate_forward(model, data_item)
        
        # Get constraint satisfaction for the current DataNode
        num_satisfied, num_constraints = get_constraints_satisfaction(node)

        # --- Test if to to start GBI for this data_item
        if num_satisfied == num_constraints:
            continue
        else:
            unsatisfied_initial += 1
            unsat_initial.append(data_iter)

        # --- Continue with GBI
        
        # test if prediction is correct - for debugging only
        if not is_correct(node):
            incorrect_initial += 1
        # ------
        
        print('data item INDEX: %d' % data_iter)
        print('CONSTRAINTS SATISFACTION: %d/%d' % (num_satisfied, num_constraints))

        print('Starting GBI:')

        # -- Make copy of original model
        model_l, c_opt = get_lambda(model, lr=args.lr)
        
        # -- model_l is the model that gets optimized by GBI
        model_l.mode(Mode.TRAIN)
        model_l.train()
        # model_l.reset()

        satisfied = False

        print('GBI START')

        for c_iter in range(gbi_iters):
            # -- forward pass through model_l
            node_l, builder_l = populate_forward(model_l, data_item)

            num_satisfied_l, num_constraints_l = get_constraints_satisfaction(node_l)

            is_satisifed = 1 if num_satisfied_l == num_constraints_l else 0

            # logits = node_l.getAttribute('logits')
            # log_probs = torch.sum(F.log_softmax(logits, dim=-1))

            # calculate global log prob from all labels
            #for ln in label_names:
            #    log_probs += torch.sum(torch.log(node_l.getAttribute('<%s>/local/softmax' % ln)))

            #  collect probs from all datanodes (regular)
            # probs = {}
            # iter through datanodes
            '''for dn in builder_l['dataNode']:
                dn.inferLocal()
                # find concept names
                for c in dn.collectConceptsAndRelations():
                    c_prob = dn.getAttribute('<%s>/local/softmax' % c[0].name)
                    if c_prob is not None and c_prob.grad_fn is not None:
                        probs[c[0].name] = c_prob'''

            # -- collect probs from datanode (in skeleton mode) 
            probs = []
            for var_name, var_val in node_l.getAttribute('variableSet').items():
                if var_name.endswith('>'):# and var_val.requires_grad:
                    probs.append(F.log_softmax(var_val, dim=-1).flatten())

            log_probs_cat = torch.cat(probs, dim=0)
            log_probs = log_probs_cat.mean()

            # get total log prob
            # log_probs = sum(probs.values) / len(probs)
            # print(log_probs)

            #  -- Constraint loss: NLL * binary satisfaction + regularization loss
            # reg loss is calculated based on L2 distance of weights between optimized model and original weights
            c_loss = log_probs * (1 - is_satisifed) + args.reg_weight * reg_loss(model_l, model)

            print("iter=%d, c_loss=%.4f, satisfied=%d" % (c_iter, c_loss.item(), num_satisfied_l))

            # --- Check if constraints are satisfied
            if num_satisfied_l == num_constraints_l:
                satisfied = True
                print(f'GBI iteration {c_iter} constraints SATISFIED')

                if is_correct(node_l):
                    print(f'Prediction for GBI iteration {c_iter} CORRECT')
                else:
                    incorrect_after += 1

                # --- End early if constraints are satisfied
                break

            # --- Backward pass on model_l
            c_loss.backward()
            
            # print("Step after backward")
            # for name, x in model_l.named_parameters():
            #     if x.grad is None:
            #         print(name, 'no grad')
            #         continue
                
            #     print(name, 'grad: ', torch.sum(torch.abs(x.grad)))
                
            #  -- Update model_l
            c_opt.step()

        # --- Test if GBI was successful - for debugging only
        if not satisfied:
            print('Constraints NOT SATISFIED')

            unsatisfied_after += 1
            incorrect_after += 1

        print('-------------------')

    print('num samples: %d' % total)
    print('initial incorrect: %.2f' % (incorrect_initial / total))
    print('initial unsatisfied: %.2f' % (unsatisfied_initial / total))
    print('after incorrect: %.2f' % (incorrect_after / total))
    print('after unsatisifed: %.2f' % (unsatisfied_after / total))

    return unsat_initial


def run_gbi_domiknows(program, dataloader, data_iters, is_correct, save_interval=100, output_folder='gbi_checkpoints'):
    """
    Runs the domiknows implementation of gradient-based inference on program. Prints pre- and post- accuracy/constraint violations.
    data_iters: number of datapoints to test in validloader
    gbi_iters: number of gradient based inference optimization steps
    label_names: names of concepts used to get log probabilities from
    is_correct: function with parameter datanode that returns whether or not the prediction is correct

    returns: list of sample indices that initially did not satisfy constraints
    """

    unsat_initial = []

    total = 0
    incorrect_initial = 0
    unsatisfied_initial = 0
    incorrect_after = 0
    unsatisfied_after = 0
    
    for i, dataitem in enumerate(tqdm(dataloader, total=min(len(dataloader), data_iters))):
        if i >= data_iters:
            break

        total += 1

        node = program.populate_one(dataitem, grad = True)
        node.inferLocal()

        # pre-gbi stats (argmax)
        num_satisfied_argmax, num_constraints_argmax = get_constraints_satisfaction(node)
        if num_satisfied_argmax != num_constraints_argmax:
            unsatisfied_initial += 1
            unsat_initial.append(i)
        
        if not is_correct(node):
            incorrect_initial += 1
        
        # post-gbi stats
        num_satisfied_gbi, num_constraints_gbi = get_constraints_satisfaction(node, suffix='/GBI')
        if num_satisfied_gbi != num_constraints_gbi:
            print('UNSATISFIED')
            unsatisfied_after += 1
        else:
            print('SATISFIED')

        # print('GBI', torch.argmax(node.getAttribute('image/<digits>/GBI'), dim=-1).tolist())
        # print('LABEL', node.getAttribute('image/digit_label').tolist())

        if not is_correct(node, suffix='/GBI'):
            print('INCORRECT')
            incorrect_after += 1
        else:
            print('CORRECT')
        
        # save at interval
        if args.training and i % save_interval == 0:
            params_filename = f'n{i}_gbiiters{args.gbi_iters}_lr{args.lr}_reg{args.reg_weight}.pth'
            torch.save(program.model.state_dict(), f'{output_folder}/{params_filename}')

    params_filename = f'n{i}_gbiiters{args.gbi_iters}_lr{args.lr}_reg{args.reg_weight}.pth'
    torch.save(program.model.state_dict(), f'{output_folder}/{params_filename}')

    print('num samples: %d' % total)
    print('initial incorrect: %.2f' % (incorrect_initial / total))
    print('initial unsatisfied: %.2f' % (unsatisfied_initial / total))
    print('after incorrect: %.2f' % (incorrect_after / total))
    print('after unsatisifed: %.2f' % (unsatisfied_after / total))

    return unsat_initial

if args.pytorch_gbi:
    run_gbi(program, validloader, args.num_samples, args.gbi_iters, ['digits0', 'digits1'], are_both_digits_correct)
else:
    run_gbi_domiknows(program, validloader, args.num_samples, are_both_digits_correct)
