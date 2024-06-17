import sys
sys.path.append('../../../')

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

from domiknows.sensor.pytorch.sensors import FunctionalSensor, ReaderSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.program import SolverPOIProgram, IMLProgram, POIProgram
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric, ValueTracker
from domiknows.program.loss import NBCrossEntropyLoss, BCEWithLogitsLoss, BCEWithLogitsIMLoss

from TypenetGraph import app_graph

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

file_data['train_bags'] = h5py.File(os.path.join("../resources/MIL_data/entity_bags.hdf5"), "r")

#file_data['embeddings'] = np.zeros(shape=(2196018, 300))
file_data['embeddings'] = np.load(os.path.join('../resources/data/pretrained_embeddings.npz'))["embeddings"]

file_data['typenet_matrix_orig'] = joblib.load(os.path.join('../resources/MIL_data/TypeNet_transitive_closure.joblib'))

with open(os.path.join('../resources/data/vocab.joblib'), "rb") as file:
    file_data['vocab_dict'] = pickle.load(file, fix_imports=True, encoding="latin1")

with open(os.path.join('../resources/MIL_data/entity_dict.joblib'), "rb") as file:
    file_data['entity_dict'] = pickle.load(file, fix_imports=True, encoding="latin1")

file_data['type_dict'] = joblib.load(os.path.join('../resources/MIL_data/TypeNet_type2idx.joblib'))

with open(os.path.join('../resources/MIL_data/entity_type_dict_orig.joblib'), "rb") as file:
    file_data['entity_type_dict'] = pickle.load(file, fix_imports=True, encoding="latin1")

wiki_train = WikiReader(file='../resources/MIL_data/train.entities', type='file', file_data=file_data, bag_size=20, limit_size=args.limit)
wiki_dev = WikiReader(file='../resources/MIL_data/dev.entities', type='file', file_data=file_data, bag_size=20, limit_size=args.limit)

'''i = 0
for w in wiki_train:
    if i == 10:
        break
    print('num labels:', sum(w['GoldTypes'][0]))'''

print('building graph')
# get graph attributes
app_graph.detach()
mention = app_graph['mention']
label = app_graph['tag']

# text data sensors
mention['MentionRepresentation'] = ReaderSensor(keyword='MentionRepresentation')
mention['Context'] = ReaderSensor(keyword='Context')

# label data sensors
mention[label] = ReaderSensor(keyword='GoldTypes', label=True)

# module learners
mention['encoded'] = ModuleLearner('Context', 'MentionRepresentation', module=MLPEncoder(pretrained_embeddings=file_data['embeddings'], mention_dim=file_data['embeddings'].shape[-1]))
mention[label] = ModuleLearner('encoded', module=TypeComparison(config.num_types, 128))

# create program

def multilabel_metric(pr, gt, data_item, prop):
    rounded_pr = np.round(torch.sigmoid(pr.data).numpy())
    return f1_score(gt, rounded_pr, average='samples')

program = POIProgram(
    app_graph,
    loss=MacroAverageTracker(BCEWithLogitsLoss()),
    metric=MacroAverageTracker(multilabel_metric)
    )

print('training')
# train
program.train(wiki_train, valid_set=wiki_dev, train_epoch_num=args.epochs, Optim=torch.optim.Adam, device=config.device)