### Chen Zheng 05/19/2022

import sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

######################################################################
# run the code:
# python train_domi_rnn.py -ilp True -cuda 1
# python train_domi_rnn.py -sample True -cuda 2
# python train_domi_rnn.py -sampleilp True -cuda 3
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

from model_domi import RNNTagger
from data_reader_no_torchtext import load_examples, word_mapping, char_mapping, tag_mapping, lower_case

from regr.program.primaldualprogram import PrimalDualProgram
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
# train_sentences = load_examples('bio_data/train.txt', True) ### True: Replace every digit in a string by a zero.
# dev_sentences = load_examples('bio_data/testa.txt', True)
# test_sentences = load_examples('bio_data/testb.txt', True)
train_sentences = load_examples('../bio_data/train.txt', True) ### True: Replace every digit in a string by a zero.
dev_sentences = load_examples('../bio_data/testa.txt', True)
test_sentences = load_examples('../bio_data/testb.txt', True)

### load glove vector and glove vocab
from torchtext.vocab import GloVe
glove = GloVe()
glove_vocab = glove.stoi

### tagging labels vocab and id
dico_tags, tag_vocab, id_to_tag = tag_mapping(train_sentences) ### tag_vocab: tag_to_id

######################################################################
# Data Processing
######################################################################

def generate_data(sentences, word_to_id, tag_to_id, lower=False):
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[lower_case(w,lower) if lower_case(w,lower) in word_to_id else 'unk']
                 for w in str_words]
        labels = [tag_to_id[w[-1]] for w in s]
        data.append({
            'fullsentencestr': str_words, ## string 
            # 'text': [words],
            # 'labels': labels, ## label
            'text': torch.LongTensor([words]),
            'labels': torch.LongTensor(labels), ## label
        })
    return data

train_examples = generate_data(train_sentences, glove_vocab, tag_vocab, lower=True)
valid_examples = generate_data(dev_sentences, glove_vocab, tag_vocab, lower=True)
test_examples = generate_data(test_sentences, glove_vocab, tag_vocab, lower=True)

print("{} / {} / {} sentences in train / dev / test.".format(len(train_examples), len(valid_examples), len(test_examples))) ## 14041 / 3250 / 3453 sentences in train / dev / test.14041 / 3250 / 3453 sentences in train / dev / test.

print(test_examples[0])


######################################################################
# Graph Declaration
######################################################################
# from graph import graph, sentence, word, b_loc, i_loc, b_per, i_per, b_org, i_org, b_misc, i_misc, o, pad, bos
from graph import graph, sentence, word, labels, sen_word_rel
graph.detach()

# def forward_tensor(x):
#     words = []
#     rels = []
#     total = 0
#     for sentence in x:
#         words.extend(sentence)
#         rels.append((total, total + len(sentence)))
#         total += len(sentence)
#         # print(words.extend(sentence))
#         # print(rels)
#         # print(total)
#         # print('<><><>'*50)

#     connection = torch.zeros(len(x), total)
#     for sid, rel in enumerate(rels):
#         connection[sid][rel[0]: rel[1]] = 1

#     words = torch.LongTensor(words)
#     return connection, words

def forward_tensor(x):
    words = []
    rels = []
    total = 0
    for sentence in x:
        words.extend(sentence)
        rels.append((total, total + len(sentence)))
        total += len(sentence)

    connection = torch.zeros(total, len(x))
    for sid, rel in enumerate(rels):
        connection[rel[0]: rel[1]][:] = 1

    words = torch.LongTensor(words)
    return connection, words

print('start the ReaderSensor!')

sentence['text'] = ReaderSensor(keyword='text', device=device)

word[sen_word_rel[0], 'text'] = JointSensor(sentence['text'], forward=forward_tensor, device=device) ## what is the meaning

word[labels] = ReaderSensor(keyword='labels',label=True, device=device)

print('start the ModuleLearner!')
word[labels] = ModuleLearner('text', module=RNNTagger(glove, tag_vocab, emb_dim=300, rnn_size=128, update_pretrained=False), device=device)


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
                                    beta=1.0,device=device)

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
# Train the model
######################################################################
n_epochs = 20

# for i in range(num_epochs):
#     program.train(train_examples, train_epoch_num=1, Optim=lambda param: torch.optim.Adam(param, lr=0.01, weight_decay=1e-5), device=device)
#     program.save("domi_"+str(i))

train_time_start = time.time()
program.train(train_examples, train_epoch_num=n_epochs, Optim=lambda param: torch.optim.Adam(param, lr=0.01, weight_decay=1e-5), device=device)
train_time_end = time.time()  
print('training time execution time: ', (train_time_end - train_time_start)*1000, ' milliseconds')

if args.ilp:
    program.save("saved_models/domi_ilp_epoch_"+str(n_epochs)+'.pt')
if args.pd:
    program.save("saved_models/domi_pd_epoch_"+str(n_epochs)+'.pt')
if args.sample:
    program.save("saved_models/domi_sampleloss_epoch_"+str(n_epochs)+'.pt')
if args.pdilp:
    program.save("saved_models/domi_pd+ilp_epoch_"+str(n_epochs)+'.pt')
if args.sampleilp:
    program.save("saved_models/domi_sampleloss+ilp_epoch_"+str(n_epochs)+'.pt')

######################################################################
# Evaluate the model
######################################################################

if args.ilp:
    program.load("saved_models/domi_ilp_epoch_"+str(n_epochs)+'.pt')
if args.pd:
    program.load("saved_models/domi_pd_epoch_"+str(n_epochs)+'.pt')
if args.sample:
    program.load("saved_models/domi_sampleloss_epoch_"+str(n_epochs)+'.pt')
if args.pdilp:
    program.load("saved_models/domi_pd+ilp_epoch_"+str(n_epochs)+'.pt')
if args.sampleilp:
    program.load("saved_models/domi_sampleloss+ilp_epoch_"+str(n_epochs)+'.pt')


from regr.utils import setProductionLogMode
productionMode = False
# if productionMode:
#     setProductionLogMode(no_UseTimeLog=False)
import logging
logging.basicConfig(level=logging.INFO)

test_time_start = time.time()
program.test(valid_examples, device=device)
test_time_end= time.time()  
print('test time execution time: ', (test_time_end - test_time_start)*1000, ' milliseconds')


# def compute_scores(item, criteria="P"):
#         entities = ["location", "people", "organization", "other"]
#         instances = {"location": 937, "people": 774, "organization": 512, "other": 610, "work_for": 71, "located_in": 75, "live_in": 103, 
#                      "orgbase_on": 97, "kill": 55} ### ???
#         sum_entity = 0
#         sum_relations = 0
#         precision_entity = 0
#         precision_relations = 0
#         normal_precision_entity = 0
#         normal_precision_relations = 0
#         sum_all = 0
#         precision_all = 0
#         normal_precision_all = 0
#         for key in entities:
#             sum_entity += float(instances[key])
#             precision_entity += float(instances[key]) * float(item[key][criteria])
#             normal_precision_entity += float(item[key][criteria])

#         sum_all = sum_relations + sum_entity
#         precision_all = precision_entity + precision_relations
#         normal_precision_all = normal_precision_relations + normal_precision_entity

#         outputs = {}
        
#         if criteria == "P":
#             outputs["micro_" + str(criteria) + "_entities"] = precision_entity / sum_entity
#             outputs["micro_" + str(criteria) + "_relations"] = precision_relations / sum_relations
#             outputs["micro_" + str(criteria) + "_all"] = precision_all / sum_all

#         outputs["macro_" + str(criteria) + "_entities"] = normal_precision_entity / len(entities)
        
#         return outputs




# metrics = program.model.metric['argmax'].value()
# results = compute_scores(metrics, criteria="F1")
# score = results["macro_F1_all"]







######################################################################
# save model
######################################################################

# def compute_scores(item, criteria="P"):
#         entities = ["location", "people", "organization", "other"]
#         instances = {"location": 937, "people": 774, "organization": 512, "other": 610, "work_for": 71, "located_in": 75, "live_in": 103, 
#                      "orgbase_on": 97, "kill": 55} ### ???
#         sum_entity = 0
#         sum_relations = 0
#         precision_entity = 0
#         precision_relations = 0
#         normal_precision_entity = 0
#         normal_precision_relations = 0
#         sum_all = 0
#         precision_all = 0
#         normal_precision_all = 0
#         for key in entities:
#             sum_entity += float(instances[key])
#             precision_entity += float(instances[key]) * float(item[key][criteria])
#             normal_precision_entity += float(item[key][criteria])

#         sum_all = sum_relations + sum_entity
#         precision_all = precision_entity + precision_relations
#         normal_precision_all = normal_precision_relations + normal_precision_entity

#         outputs = {}
        
#         if criteria == "P":
#             outputs["micro_" + str(criteria) + "_entities"] = precision_entity / sum_entity
#             outputs["micro_" + str(criteria) + "_relations"] = precision_relations / sum_relations
#             outputs["micro_" + str(criteria) + "_all"] = precision_all / sum_all

#         outputs["macro_" + str(criteria) + "_entities"] = normal_precision_entity / len(entities)
        
#         return outputs

# def save_best(program, epoch=1, best_epoch=-1, best_macro_f1=0):
#         import logging
#         logger = logging.getLogger(__name__)
#         metrics = program.model.metric['argmax'].value()
#         results = compute_scores(metrics, criteria="F1")
#         score = results["macro_F1_all"]
#         if score > best_macro_f1:
#             logger.info(f'New Best Score {score} achieved at Epoch {epoch}.')
#             best_epoch = epoch
#             best_macro_f1 = score
#             if args.number == 1:
#                 program.save(f'saves/conll04-bert-{split_id}-best-macro-f1.pt')
#             else:
#                 program.save(f'saves/conll04-bert-{split_id}-size-{args.number}-best_macro-f1.pt')
#         return epoch + 1, best_epoch, best_macro_f1