### Chen Zheng 05/19/2022

import sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

######################################################################
# run the code:
# python test.py -ilp True -cuda 0
# python test.py -pd True -cuda 0
# python test.py -sample True -cuda 0
# python test.py -pdilp True -cuda 0
# python test.py -sampleilp True -cuda 0
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


if args.pd or args.pdilp:
    device = "cpu"
else:
    device = "cuda:"+str(args.cuda)

# device = "cpu"

num_epochs = 5

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
from graph import graph, sentence, word, labels, sen_word_rel
graph.detach()

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
# Test the model
######################################################################

if args.ilp:
    program.load("2022_saved_model/ilp/ilp.pt")
if args.pd:
    program.load("2022_saved_model/pd/pd.pt")
if args.sample:
    program.load("2022_saved_model/sample/sample.pt")
if args.pdilp:
    program.load("2022_saved_model/pdilp/pdilp.pt")
if args.sampleilp:
    program.load("2022_saved_model/sampleilp/sampleilp.pt")

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

