import sys
sys.path.append('../../')

import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import h5py
import os
import joblib
import pickle
import torch
import argparse
from sklearn.metrics import accuracy_score, f1_score

from regr.sensor.pytorch.sensors import FunctionalSensor, ReaderSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.program import SolverPOIProgram, IMLProgram, POIProgram
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric, ValueTracker
from regr.program.loss import NBCrossEntropyLoss, BCEWithLogitsLoss, BCEWithLogitsIMLoss

from TypenetGraphDepth import app_graph
import TypenetGraphDepth as TypenetGraph

from sensors.MLPEncoder import MLPEncoder
from sensors.TypeComparison import TypeComparison
from readers.TypenetReader import WikiReader

import config

print('current device: ', config.device)

# args
parser = argparse.ArgumentParser()
parser.add_argument('--limit', dest='limit', type=int, default=None)
parser.add_argument('--epochs', dest='epochs', type=int, default=10)

args = parser.parse_args()

# load data
file_data = {}

file_data['type_dict'] = joblib.load(os.path.join('resources/MIL_data/TypeNet_type2idx.joblib'))

file_data['train_bags'] = h5py.File(os.path.join("resources/MIL_data/entity_bags.hdf5"), "r")
#file_data['embeddings'] = np.zeros(shape=(2196018, 300))
file_data['embeddings'] = np.load(os.path.join('resources/data/pretrained_embeddings.npz'))["embeddings"]

file_data['typenet_matrix_orig'] = joblib.load(os.path.join('resources/MIL_data/TypeNet_transitive_closure.joblib'))

with open(os.path.join('resources/data/vocab.joblib'), "rb") as file:
    file_data['vocab_dict'] = pickle.load(file, fix_imports=True, encoding="latin1")

with open(os.path.join('resources/MIL_data/entity_dict.joblib'), "rb") as file:
    file_data['entity_dict'] = pickle.load(file, fix_imports=True, encoding="latin1")

with open(os.path.join('resources/MIL_data/entity_type_dict_orig.joblib'), "rb") as file:
    file_data['entity_type_dict'] = pickle.load(file, fix_imports=True, encoding="latin1")

wiki_train = WikiReader(file='resources/MIL_data/train.entities', type='file', file_data=file_data, bag_size=20, limit_size=args.limit)
wiki_dev = WikiReader(file='resources/MIL_data/dev.entities', type='file', file_data=file_data, bag_size=20, limit_size=args.limit)

'''i = 0
for w in wiki_train:
    if i == 10:
        break
    print('num labels:', w['GoldTypes'][0])'''

for w in wiki_train:
    print(w['tag_0'])
    break

print('building graph')
# get graph attributes
app_graph.detach()
mention = app_graph['mention']

# text data sensors
mention['MentionRepresentation'] = ReaderSensor(keyword='MentionRepresentation')
mention['Context'] = ReaderSensor(keyword='Context')

# label data sensors
for i, l_depth in enumerate(TypenetGraph.labels):
    mention[l_depth] = ReaderSensor(keyword='tag_%d' % i, label=True)

mention[TypenetGraph.label_other] = ReaderSensor(keyword='tag_other', label=True)

# module learners
mention['encoded'] = ModuleLearner(
    'Context',
    'MentionRepresentation', 
    module=MLPEncoder(
            pretrained_embeddings=file_data['embeddings'],
            mention_dim=file_data['embeddings'].shape[-1],
            hidden_dim=128
        )
    )

for i, l_depth in enumerate(TypenetGraph.labels):
    mention[l_depth] = ModuleLearner('encoded', module=TypeComparison(128, len(l_depth.attributes)))

mention[TypenetGraph.label_other] = ModuleLearner('encoded', module=TypeComparison(128, len(TypenetGraph.label_other.attributes)))

# create program
def multilabel_metric(pr, gt, data_item, prop):
    #print(data_item)

    return 0

program = POIProgram(
    app_graph,
    loss=MacroAverageTracker(NBCrossEntropyLoss())
    )

print('training')
# train
program.train(wiki_train, valid_set=wiki_dev, train_epoch_num=args.epochs, Optim=torch.optim.Adam, device=config.device)

