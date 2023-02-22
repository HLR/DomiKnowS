import torch
torch.manual_seed(0)

import random
random.seed(0)

import numpy as np
np.random.seed(0)

from domiknows.sensor.pytorch.sensors import FunctionalSensor, ReaderSensor, JointSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.program import SolverPOIProgram
from domiknows.utils import detuple
from sklearn.metrics import classification_report, accuracy_score
import copy
from torch.optim import SGD
import json

from graph import *
from data import valid_dataset
from net import SimpleLSTM, HighwayLSTM
from gbi import reg_loss

from domiknows import setProductionLogMode
setProductionLogMode(no_UseTimeLog=True)

# reading words and labels
sentence['words'] = ReaderSensor(keyword='words')

# (1, seq_length, 1)
sentence['predicate'] = ReaderSensor(keyword='predicate')

lstm = HighwayLSTM(
    label_space = [
        'B-ARG0',
        'I-ARG0',
        'B-ARG1',
        'I-ARG1',
        'O'
    ]
)
lstm.load_state_dict(torch.load('srl/data/bio_srl_lstm_epoch2.pth', map_location=torch.device('cpu')))

def print_grads(model):
    for name, x in model.named_parameters():
        if x.grad is None:
            print(name, 'None')
        else:
            print(name, torch.norm(x.grad))

#lstm = SimpleLSTM()
#lstm.load_state_dict(torch.load('examples/srl/data/simple_lstm_50.pth', map_location=torch.device('cpu')))

lstm_copy = copy.deepcopy(lstm)

for param in lstm_copy.parameters():
    param.requires_grad = False

# (1, seq_length, 3)
sentence['predictions_tmp'] = ModuleLearner(sentence['words'], sentence['predicate'], module=lstm)

sentence['predictions', 'likelihood'] = JointSensor(sentence['predictions_tmp'], forward=lambda x: (x[0][0], [x[0][1]]))

for j, (pred, tag_names) in enumerate(predictions):
    def _logits_to_pred(select_idx):
        def logits_to_pred(logits):
            logits = logits[0]
            if select_idx < len(logits):
                return logits[select_idx].unsqueeze(0)
            return torch.tensor([100, -100, -100]).float().unsqueeze(0)
        return logits_to_pred

    sentence[pred] = FunctionalSensor(sentence['predictions'], forward=_logits_to_pred(j))

for i, single_span in enumerate(spans):
    sentence['single_span_%d' % i] = ReaderSensor(keyword='span_%d' % i)
    for j, span_tkn in enumerate(single_span):
        def span_to_tkn(span_vec):
            return [span_vec[0][j]]

        def _span_to_tkn_fixed(select_idx):
            def span_to_tkn_fixed(span_vec):
                result = torch.ones(1, 2) * -100
                if select_idx < len(span_vec[0]):
                    idx = span_vec[0][select_idx].long()
                else:
                    idx = 0
                result[0, idx] = 100
                return result
            return span_to_tkn_fixed

        sentence[span_tkn] = FunctionalSensor(sentence['single_span_%d' % i], forward=_span_to_tkn_fixed(j))


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
                          poi=(sentence,),
                          inferTypes=['local/argmax'],
                          metric={})

model = program.model


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

        #print("constraint #%d" % (lc_idx), lc + ':', verifyResult[lc]['satisfied'])

    num_constraints = len(verifyResult)
    num_satisifed = sum(satisfied_constraints) // 100

    return num_satisifed, num_constraints


def print_node_predictions(node, templ='%s'):
    attributes = node.getAttributes()
    sentence_length = attributes['predictions'].shape[0]

    tags = []
    for i in range(num_words):
        tags.append(torch.argmax(attributes['<pred_%d>' % i]).item())
    tags = tags[:sentence_length]

    print(templ % tags) 

def is_correct(node, data_item, verbose=True):
    attributes = node.getAttributes()
    sentence_length = attributes['predictions'].shape[0]

    tags = []
    for i in range(num_words):
        tags.append(torch.argmax(attributes['<pred_%d>' % i]).item())
    tags = tags[:sentence_length]

    gt = data_item['arg_label'].squeeze().tolist()

    if verbose:
        print('PRED:\t%s' % tags)
        print('GT:\t%s' % gt)

    return gt == tags

def get_metrics(node, data_item):
    attributes = node.getAttributes()
    sentence_length = attributes['predictions'].shape[0]

    tags = []
    for i in range(num_words):
        tags.append(torch.argmax(attributes['<pred_%d>' % i]).item())
    tags = tags[:sentence_length]

    gt = data_item['arg_label'].squeeze().tolist()

    acc = accuracy_score(gt, tags)

    print('token-level tag accuracy: %.2f' % acc)

    return gt, tags

def manual_constraints_check(node, data_item):
    attributes = node.getAttributes()
    sentence_length = attributes['predictions'].shape[0]

    tags = []
    for i in range(num_words):
        tags.append(torch.argmax(attributes['<pred_%d>' % i]).item())
    tags = tags[:sentence_length]
    
    tags = [1 if t != 0 else 0 for t in tags]

    #print('valid spans set: %s' % [s.long().tolist() for s in data_item['spans_all'][0]])

    manual_span_check = any([
        tags == s.long().tolist() for s in data_item['spans_all'][0]
    ])

    print('manual constraints check: %s' % ('SATISIFED' if manual_span_check else 'NOT SATISIFIED'))
    
    return manual_span_check

def get_node_likelihood(builder_l):
    probs = {}
    # iter through datanodes
    for dn in builder_l['dataNode']:
        dn.inferLocal()
        # find concept names
        for c in dn.collectConceptsAndRelations():
            c_prob = dn.getAttribute('<%s>/local/softmax' % c[0].name)
            if c_prob.grad_fn is not None:
                probs[c[0].name] = c_prob

    # get total log prob
    log_probs = 0.0
    for c_prob in probs.values():
        log_probs += torch.sum(torch.log(c_prob))
    
    return log_probs


pre_metrics = {
    'correct': 0,
    'satisfied': 0,
    'total': 0,
    'token_preds': [],
    'token_gts': []
}
post_metrics = {
    'correct': 0,
    'satisfied': 0,
    'total': 0,
    'token_preds': [],
    'token_gts': []
}

only_not_satisfied = False

# dataset only containing values indices for examples that aren't initially satisfied by the model
# i.e., examples that we have to run gbi over
not_satisfied = [4]
not_satisfied_iter = filter(lambda x: x[0] in not_satisfied, enumerate(valid_dataset))

use_dset = not_satisfied_iter if only_not_satisfied else enumerate(valid_dataset)

for data_iter, data_item in use_dset:
    data_item = valid_dataset[data_iter]

    print('-' * 10 + ' idx = %d / %d ' % (data_iter, len(valid_dataset) - 1) + '-' * 10)

    # make initial prediction
    with torch.no_grad():
        node, _ = populate_forward(model, data_item)

    # evaluate initial prediction
    num_satisfied, num_constraints = get_constraints(node)

    is_correct(node, data_item)
    token_gts, token_preds = get_metrics(node, data_item)

    # save metrics
    pre_metrics['token_preds'].extend(token_preds)
    pre_metrics['token_gts'].extend(token_gts)
    pre_metrics['total'] += 1

    # if constraints are all satisfied, skip
    if num_satisfied == num_constraints:
        pre_metrics['satisfied'] += 1
        print('SATISFIED')
        if is_correct(node, data_item, verbose=False):
            print('CORRECT')
            pre_metrics['correct'] += 1
        else:
            print('INCORRECT')
        continue

    # if constraints aren't all satisfied, do gbi
    not_satisfied.append(data_iter)
    post_metrics['total'] += 1
 
    print('CONSTRAINTS SATISFACTION: %d/%d' % (num_satisfied, num_constraints))

    print('\nStarting GBI:')

    # reset model parameters
    # map keys from pure pytorch model to domiknows model
    mapped_state_dict = {'global/Sentence/predictions_tmp/modulelearner.' + k: v for k, v in lstm_copy.state_dict().items()}
    # load mapped model
    model.load_state_dict(copy.deepcopy(mapped_state_dict))

    # optimizer for gbi model
    c_opt = SGD(model.parameters(), lr=1e-3)

    satisfied = False
    last_token_preds = None

    # gbi iterations
    for c_iter in range(100):
        c_opt.zero_grad()

        # forward pass through model
        node_l, builder_l = populate_forward(model, data_item)

        # get constraint satisfaction
        num_satisfied_l, num_constraints_l = get_constraints(node_l)

        # get model likelihood for current prediction
        log_probs = node_l.getAttribute('likelihood')

        # calculate gbi loss:
        # constraint term: (violations/total_constraints) * likelihood
        rel_constraint_violation = (num_constraints_l - num_satisfied_l) / num_constraints_l
        #c_loss = -1 * log_probs * num_satisfied_l + reg_loss(model_l, model)
        
        # regularization term: l2 distance between weights current optimized model and original model
        reg_term = reg_loss(model, lstm_copy)
        cons_term = log_probs * rel_constraint_violation

        # sum with weighting term
        reg_weight = 1.0
        c_loss = cons_term + reg_weight * reg_term

        # backpropagate  
        c_loss.backward()

        # make sure all the gradients are defined
        for name, x in model.named_parameters():
            assert x.grad is not None

        c_opt.step()

        # save metrics and print
        print("iter=%d, c_loss=%.4f (cons=%.4f, reg=%.4f), num_satisfied=%d" % (c_iter, c_loss.item(), cons_term.item(), reg_term.item(), num_satisfied_l))
        #print_node_predictions(node_l, templ='PRED:\t')
        is_correct(node_l, data_item)
        manual_constraints_check(node_l, data_item)
        _, last_token_preds = get_metrics(node_l, data_item)

        if num_satisfied_l == num_constraints_l:
            satisfied = True
            print('SATISFIED')
            post_metrics['satisfied'] += 1

            if is_correct(node_l, data_item, verbose=False):
                print('CORRECT')
                post_metrics['correct'] += 1
            else:
                print('INCORRECT')

            break

        print()

    post_metrics['token_preds'].extend(last_token_preds)
    post_metrics['token_gts'].extend(token_gts)

    if not satisfied:
        print('NOT SATISFIED')

# save non-satisfied examples
with open('srl/not_satisfied_dev.json', 'w') as file_out:
    json.dump(not_satisfied, file_out)

# print metrics
print('\n\n')

print('-' * 10, 'METRICS', '-' * 10)

print('Pre accuracy: %.2f, Pre satsifaction: %.2f, n=%d' %
    (
        pre_metrics['correct'] / pre_metrics['total'],
        pre_metrics['satisfied'] / pre_metrics['total'],
        post_metrics['total']
    )
)

print(classification_report(pre_metrics['token_gts'], pre_metrics['token_preds'], zero_division=0))

print('\n\n')

print('Post accuracy: %.2f, Post satsifaction: %.2f, n=%d' %
    (
        post_metrics['correct'] / post_metrics['total'],
        post_metrics['satisfied'] / post_metrics['total'],
        post_metrics['total']
    )
)

print(classification_report(post_metrics['token_gts'], post_metrics['token_preds'], zero_division=0))
