import sys
sys.path.append("../../")

import logging
logging.basicConfig(level=logging.INFO)

from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.relation import disjoint
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.program import SolverPOIProgram
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from domiknows.program.loss import NBCrossEntropyLoss
from torch import nn
import torch
import glob
import random
import os

from modules.lstm_module import LSTMModule
from sensors.embedding_sensor import EmbeddingSensor

# params
EMBED_SIZE = 300
HIDDEN_SIZE = 100
NUM_CLASSES = 2
DROP_RATE = 0.5
TRAIN_SPLIT = 0.8

# build graph
Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
  review = Concept(name='review')

  positive = review(name='positive')
  negative = review(name='negative')

  disjoint(positive, negative)

# read text and labels
review['text'] = ReaderSensor(keyword='text')

review[positive] = ReaderSensor(keyword='positive', label=True)
review[negative] = ReaderSensor(keyword='negative', label=True)

# get embedding vectors from raw text
review['text_embed'] = EmbeddingSensor('text', embed_size=EMBED_SIZE)

# create learners
review['rnn_embed'] = ModuleLearner('text_embed', module=LSTMModule(EMBED_SIZE, HIDDEN_SIZE, DROP_RATE))

review[positive] = ModuleLearner('rnn_embed', module=nn.Linear(HIDDEN_SIZE * 2, NUM_CLASSES))
review[negative] = ModuleLearner('rnn_embed', module=nn.Linear(HIDDEN_SIZE * 2, NUM_CLASSES))

# create program
program = SolverPOIProgram(graph, inferTypes=['ILP', 'local/argmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()), metric={'ILP':PRF1Tracker(DatanodeCMMetric()),'argmax':PRF1Tracker(DatanodeCMMetric('local/argmax'))})

# import and shuffle data
def get_data(directory, label):
  data_all = []
  for path in glob.glob(os.path.join(directory, label + '/*.txt')):
    data_dict = {}
    with open(path, 'r') as f:
      data_dict['text'] = f.read()
      data_dict['positive'] = [1 if label == 'pos' else 0]
      data_dict['negative'] = [1 if label == 'neg' else 0]
    data_all.append(data_dict)
  return data_all

train_data = get_data('data/aclImdb/train', 'pos')
train_data.extend(get_data('data/aclImdb/train', 'neg'))
random.shuffle(train_data)

test_data = get_data('data/aclImdb/test', 'pos')
test_data.extend(get_data('data/aclImdb/test', 'neg'))
random.shuffle(test_data)

# train
split_idx = int(len(train_data)*TRAIN_SPLIT)
program.train(train_data[:split_idx],
              valid_set=train_data[split_idx:],
              test_set=test_data, train_epoch_num=10, Optim=torch.optim.Adam, device='cpu')