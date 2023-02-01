import sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")


######################################################################
# run the code:
# python train_domi_bert_no_batch.py -ilp True -cuda 5
# python train_domi_bert_no_batch.py -pd True -cuda 0
# python train_domi_bert_no_batch.py -sample True -cuda 6
# python train_domi_bert_no_batch.py -pdilp True -cuda 0
# python train_domi_bert_no_batch.py -sampleilp True -cuda 7
######################################################################

import torch
from torch import nn
import numpy as np

from regr.sensor.pytorch.sensors import ReaderSensor, ConcatSensor, FunctionalSensor, JointSensor, JointReaderSensor
from regr.sensor.pytorch.learners import ModuleLearner
import graph

from regr.program import POIProgram, IMLProgram, SolverPOIProgram
from regr.program.metric import MacroAverageTracker, PRF1Tracker, PRF1Tracker, DatanodeCMMetric
from regr.program.loss import NBCrossEntropyLoss

from model_domi import BIO_Model

import torch.nn.functional as F
from torch.utils import data
from tqdm import trange, tqdm
from transformers import BertTokenizer, AdamW

from data_set import BIOProcessor, BIODataSet

from regr.program.lossprogram import PrimalDualProgram
from regr.program.lossprogram import SampleLossProgram
from regr.program.model.pytorch import SolverModel

import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-ilp', dest='ilp', default=False, help='use ILP or not', type=bool)
parser.add_argument('-pd', dest='pd', default=False,help='use primaldual or not', type=bool)
parser.add_argument('-sample', dest='sample', default=False, help='use sampling loss or not', type=bool)
parser.add_argument('-pdilp', dest='pdilp', default=False, help='use sampling loss or not', type=bool)
parser.add_argument('-sampleilp', dest='sampleilp', default=False, help='use sampling loss or not', type=bool)
parser.add_argument('-cuda', dest='cuda', default=0, help='cuda number', type=int)
args = parser.parse_args()


device = "cuda:"+str(args.cuda)
# device = "cpu"

######################################################################
# Data Reader
######################################################################
num_epochs = 8
batch_size=1
# data_dir='domi_bert_bio/data'
data_dir='data'
max_len=32

tokenizer = BertTokenizer.from_pretrained('./local_model_directory')
# tokenizer = BertTokenizer.from_pretrained('domi_bert_bio/local_model_directory')
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


bio_tagging_processor = BIOProcessor()

train_examples = bio_tagging_processor.get_train_examples(data_dir)
val_examples = bio_tagging_processor.get_dev_examples(data_dir)
# test_examples = bio_tagging_processor.get_test_examples(data_dir)

tags_vals = bio_tagging_processor.get_labels()
label_map = {}

for (i, label) in enumerate(tags_vals):
    if '-' in label:
        label = '_'.join(label.split('-'))
    label_map[label] = i

# print(label_map) ### {'O': 0, 'B_MISC': 1, 'I_MISC': 2, 'B_PER': 3, 'I_PER': 4, 'B_ORG': 5, 'I_ORG': 6, 'B_LOC': 7, 'I_LOC': 8, '[CLS]': 9, '[SEP]': 10, 'X': 11}
# {'O': 0, 'B_MISC': 1, 'I_MISC': 2, 'B_PER': 3, 'I_PER': 4, 'B_ORG': 5, 'I_ORG': 6, 'B_LOC': 7, 'I_LOC': 8, 'X': 9}
# print(label_map.keys())

train_dataset = BIODataSet(data_list=train_examples[:100], tokenizer=tokenizer, label_map=label_map, max_len=max_len)
eval_dataset = BIODataSet(data_list=val_examples[:100], tokenizer=tokenizer, label_map=label_map, max_len=max_len)
# test_dataset = BIODataSet(data_list=test_examples[:100], tokenizer=tokenizer, label_map=label_map, max_len=max_len)

train_iter = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
eval_iter = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
# test_iter = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

######################################################################
# Data Processing
######################################################################

def generate_data(data_iteration):
    data = []
    for batch in data_iteration:
    # for batch in test_iter:
        b_input_ids, b_labels, b_input_mask, b_token_type_ids, b_label_masks = batch
        # one_hot = F.one_hot(b_labels, num_classes=len(label_map))
        data.append({
            'text': b_input_ids.view(1, max_len),
            'input_mask': b_input_mask.view(1, max_len),
            'token_type_ids': b_token_type_ids.view(1, max_len),
            'labels': b_labels.view(max_len),
            'label_masks': b_label_masks.view(1, max_len),
        })
    return data

train_examples = generate_data(train_iter)
valid_examples = generate_data(eval_iter)
# test_examples = generate_data(test_iter)


######################################################################
# Graph Declaration
######################################################################
from graph import graph, sentence, word, labels, sen_word_rel
graph.detach()

def forward_tensor(x,x2,x3,x4):
    words = []
    input_mask = []
    token_type_ids = []
    label_masks = []

    rels = []
    total = 0
    for i in range(len(x)):
        words.extend(x[i])
        input_mask.extend(x2[i])
        token_type_ids.extend(x3[i])
        label_masks.extend(x4[i])
        rels.append((total, total + len(x[i])))
        total += len(x[i])

    connection = torch.zeros(total, len(x))
    for sid, rel in enumerate(rels):
        connection[rel[0]: rel[1]][:] = 1

    words = torch.LongTensor(words)
    input_mask = torch.LongTensor(input_mask)
    token_type_ids = torch.LongTensor(token_type_ids)
    label_masks = torch.LongTensor(label_masks)
    return connection, words, input_mask, token_type_ids, label_masks


print('start the ReaderSensor!')

sentence['text'] = ReaderSensor(keyword='text', device=device)
sentence['input_mask'] = ReaderSensor(keyword='input_mask', device=device)
sentence['token_type_ids'] = ReaderSensor(keyword='token_type_ids', device=device)
sentence['label_masks'] = ReaderSensor(keyword='label_masks', device=device)

word[sen_word_rel[0], 'text', 'input_mask', 'token_type_ids', 'label_masks'] = JointSensor(sentence['text'], sentence['input_mask'], sentence['token_type_ids'], sentence['label_masks'], forward=forward_tensor, device=device)

word[labels] = ReaderSensor(keyword='labels',label=True, device=device)
print('start the ModuleLearner!')

model = BIO_Model.from_pretrained('./local_model_directory', num_labels=len(label_map)).to(device)
# model = BIO_Model.from_pretrained('bert-base-cased', num_labels=len(label_map)).to(device)
word[labels] = ModuleLearner('text', 'input_mask', 'token_type_ids', 'label_masks', module=model, device=device)

######################################################################
# ILP or PD or SAMPLELOSS or PD+ILP or SAMPLELOSS+ILP
######################################################################

if args.ilp:
    print('run ilp program')
    program = SolverPOIProgram(graph, poi=(sentence, word), inferTypes=['ILP', 'local/argmax'],
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                            'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                    device=device)

if args.pd:
    print('run PrimalDual program')
    program = PrimalDualProgram(graph, SolverModel, poi=(sentence, word), inferTypes=['local/argmax'],
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                    beta=1.0, device=device)

if args.sample:
    print('run sampling loss program')
    program = SampleLossProgram(graph, SolverModel, poi=(sentence, word), inferTypes=['local/argmax'],
                                    sample=True, sampleSize=100, sampleGlobalLoss = False,
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                    device=device)

if args.pdilp:
    print('run PrimalDual + ILP program')
    program = PrimalDualProgram(graph, SolverModel, poi=(sentence, word), inferTypes=['ILP', 'local/argmax'],
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                            'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                    beta=1.0, device=device)

if args.sampleilp:
    print('run sampling loss + ILP program')
    program = SampleLossProgram(graph, SolverModel, poi=(sentence, word), inferTypes=['ILP', 'local/argmax'],
                                    sample=True, sampleSize=100, sampleGlobalLoss=False,
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()), 
                                    metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                            'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                    beta=1.0, device=device)


print('finish Graph Declaration')

######################################################################
### Train the model
######################################################################

# for i in range(num_epochs):
#     program.train(train_examples, train_epoch_num=1, Optim=lambda param: torch.optim.Adam(param, lr=0.01, weight_decay=1e-5), device=device)
#     program.save("domi_"+str(i))


train_time_start = time.time()
program.train(train_examples, train_epoch_num=num_epochs, Optim=lambda param: torch.optim.Adam(param, lr=0.01, weight_decay=1e-5), device=device)
train_time_end = time.time()  
print('training time execution time: ', (train_time_end - train_time_start)*1000, ' milliseconds')

if args.ilp:
    program.save("saved_models/domi_ilp_epoch_"+str(num_epochs)+'.pt')
if args.pd:
    program.save("saved_models/domi_pd_epoch_"+str(num_epochs)+'.pt')
if args.sample:
    program.save("saved_models/domi_sampleloss_epoch_"+str(num_epochs)+'.pt')
if args.pdilp:
    program.save("saved_models/domi_pd+ilp_epoch_"+str(num_epochs)+'.pt')
if args.sampleilp:
    program.save("saved_models/domi_sampleloss+ilp_epoch_"+str(num_epochs)+'.pt')

#####################################################################
### Evaluate the model
#####################################################################

if args.ilp:
    program.load("saved_models/domi_ilp_epoch_"+str(num_epochs)+'.pt')
if args.pd:
    program.load("saved_models/domi_pd_epoch_"+str(num_epochs)+'.pt')
if args.sample:
    program.load("saved_models/domi_sampleloss_epoch_"+str(num_epochs)+'.pt')
if args.pdilp:
    program.load("saved_models/domi_pd+ilp_epoch_"+str(num_epochs)+'.pt')
if args.sampleilp:
    program.load("saved_models/domi_sampleloss+ilp_epoch_"+str(num_epochs)+'.pt')


from regr.utils import setProductionLogMode
productionMode = False
import logging
logging.basicConfig(level=logging.INFO)


test_time_start = time.time()
program.test(valid_examples, device=device)
test_time_end= time.time()
print('test time execution time: ', (test_time_end - test_time_start)*1000, ' milliseconds')

#####################################################################
### Compute Violation Rate
#####################################################################

violation_rate = 0
count_constraints = 0
for node in program.populate(valid_examples, device=device):
        verifyResult = node.verifyResultsLC()
        node_average = 0
        if verifyResult:
            for lc in verifyResult:
                if "ifSatisfied" in verifyResult[lc]:
                    node_average += verifyResult[lc]["ifSatisfied"]
                else:
                    node_average += verifyResult[lc]["satisfied"]
            node_average = node_average / len(verifyResult)
            violation_rate += node_average
            count_constraints += 1
print(f"Average satisfaction is : {violation_rate/count_constraints}")

                # print("lc %s is %i%% satisfied by learned results"%(lc, verifyResult[lc]['satisfied']))
# print('test time execution time: ', (test_time_end - test_time_start)*1000, ' milliseconds')

