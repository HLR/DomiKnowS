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
from regr.program import SolverPOIProgram, IMLProgram, POIProgram, CallbackProgram
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric, ValueTracker
from regr.program.loss import NBCrossEntropyLoss, BCEWithLogitsLoss, BCEWithLogitsIMLoss

import TypenetGraph
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
parser.add_argument('--limit_classes', dest='limit_classes', type=int, default=None)
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

first_iter = list(wiki_train)[0]

print('building graph')

# get graph attributes
app_graph.detach()
mention = app_graph['mention']
mention_group = app_graph['mention_group']

# Enable skeleton DataNode
from regr.utils import setDnSkeletonMode
setDnSkeletonMode(True)

# text data sensors
mention['MentionRepresentation'] = ReaderSensor(keyword='MentionRepresentation')
mention['Context'] = ReaderSensor(keyword='Context')

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

# module learner predictions
for i, (type_name, type_concept) in enumerate(TypenetGraph.concepts.items()):
    if not args.limit_classes == None and i >= args.limit_classes:
        print('stopped after adding %d classe(s)' % args.limit_classes)
        break
    mention[type_concept] = ModuleLearner('encoded', module=TypeComparison(128, 2))

def test(input, target, data_item, prop, weight=None):
    print(prop)

class LossCallback():
    def __init__(self, program):
        self.program = program

    def __call__(self):
        vals = self.program.model.loss.value()

        print("averaged loss:", torch.tensor(list(vals.values())).mean())

class Program(CallbackProgram, POIProgram):
        pass

# create program
program = Program(
    app_graph,
    loss=MacroAverageTracker(NBCrossEntropyLoss()),
    metric=PRF1Tracker(DatanodeCMMetric())
    )

program.after_train_epoch = [LossCallback(program)]

print('training')
# train
program.train(wiki_train, train_epoch_num=args.epochs, Optim=torch.optim.Adam, device=config.device)


print(program.model.loss)
