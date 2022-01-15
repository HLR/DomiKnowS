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
import gc
from sklearn.metrics import accuracy_score, f1_score

from regr.sensor.pytorch.sensors import FunctionalSensor, ReaderSensor, JointSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.program import SolverPOIProgram, IMLProgram, POIProgram, CallbackProgram, SolverPOIDictLossProgram
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric, ValueTracker, CMWithLogitsMetric
from regr.program.loss import NBCrossEntropyLoss, BCEWithLogitsLoss, BCEWithLogitsIMLoss, NBCrossEntropyIMLoss
from regr.utils import setProductionLogMode

import TypenetGraph
from TypenetGraph import app_graph
from TypenetGraph import mention_group_contains
from TypenetGraph import concepts

from sensors.CNNEncoder import CNNEncoder
from sensors.TypeComparison import TypeComparison
from readers.TypenetReader import WikiReader, CachedWikiReader
from TotalTracker import PRF1SepTracker

import config


print('current device: ', config.device)

# args
parser = argparse.ArgumentParser()
parser.add_argument('--limit', dest='limit', type=int, default=None)
parser.add_argument('--epochs', dest='epochs', type=int, default=10)
parser.add_argument('--limit_classes', dest='limit_classes', type=int, default=None)

parser.add_argument('--use_cache', action='store_true')
parser.add_argument('--use_cached_embed', action='store_true')
parser.set_defaults(use_cache=False, use_cached_embed=False)

args = parser.parse_args()

print('use cache:', args.use_cache)
print('use cached embeddings:', args.use_cached_embed)

# load data
file_data = {}

#file_data['embeddings'] = np.zeros(shape=(2196018, 300))

if args.use_cached_embed:
    file_data['embeddings'] = np.load('resources/pruned_embed.npy')
else:
    file_data['embeddings'] = np.load('resources/data/pretrained_embeddings.npz')["embeddings"]

if not args.use_cache:
    file_data['type_dict'] = joblib.load(os.path.join('resources/MIL_data/TypeNet_type2idx.joblib'))

    file_data['train_bags'] = h5py.File(os.path.join("resources/MIL_data/entity_bags.hdf5"), "r")

    file_data['typenet_matrix_orig'] = joblib.load(os.path.join('resources/MIL_data/TypeNet_transitive_closure.joblib'))

    with open(os.path.join('resources/data/vocab.joblib'), "rb") as file:
        file_data['vocab_dict'] = pickle.load(file, fix_imports=True, encoding="latin1")

    with open(os.path.join('resources/MIL_data/entity_dict.joblib'), "rb") as file:
        file_data['entity_dict'] = pickle.load(file, fix_imports=True, encoding="latin1")

    with open(os.path.join('resources/MIL_data/entity_type_dict_orig.joblib'), "rb") as file:
        file_data['entity_type_dict'] = pickle.load(file, fix_imports=True, encoding="latin1")

    prune_map = {}

    wiki_train = WikiReader(file='resources/MIL_data/train.entities', type='file', file_data=file_data, bag_size=10, limit_size=args.limit, vocab_map=prune_map, prune=True)

    wiki_dev = WikiReader(file='resources/MIL_data/dev.entities', type='file', file_data=file_data, bag_size=10, limit_size=args.limit, vocab_map=prune_map, prune=True)

    wiki_train.save_cache('resources/wiki_train_cache.pkl')
    wiki_dev.save_cache('resources/wiki_dev_cache.pkl')

    file_data['embeddings'] = wiki_dev.get_pruned_embeddings()

    np.save('resources/pruned_embed', file_data['embeddings'])
    print('Saved pruned embeddings')
else:
    wiki_train = CachedWikiReader(file='resources/wiki_train_cache.pkl', type='file')
    wiki_dev = CachedWikiReader(file='resources/wiki_train_cache.pkl', type='file')

train_class_weights = wiki_train.get_class_weights()

#print(wiki_train.class_counts)

first_iter = list(wiki_train)[0]

#print(first_iter)

print('building graph')

# get graph attributes
mention = app_graph['mention']
mention_group = app_graph['mention_group']

# text data sensors
mention_group['MentionRepresentation_group'] = ReaderSensor(keyword='MentionRepresentation')
mention_group['Context_group'] = ReaderSensor(keyword='Context')

for type_name in concepts:
    if not type_name[:6] == 'Synset' or not config.freebase_only:
        mention_group[type_name + '_group'] = ReaderSensor(keyword=type_name)

#mention[mention_group_contains] = FunctionalSensor(mention_group['MentionRepresentation_group'], forward=lambda x:torch.ones((32, 1)))

#mention['MentionRepresentation'] = FunctionalSensor(mention_group['MentionRepresentation_group'], forward=make_batch_list)
#mention['Context'] = FunctionalSensor(mention_group['Context_group'], forward=make_batch_list)

def make_mention_props(mentionrep_group, context_group, *labels):
    result = [torch.ones((config.batch_size, 1), device=config.device), mentionrep_group[0], context_group[0]]
    for l in labels:
        if not torch.is_tensor(l):
            l = torch.tensor(l, device=config.device)
        result.append(l)
    return result

mention_type_names = []
mention_group_concepts = []

for type_name in concepts:
    if not type_name[:6] == 'Synset' or not config.freebase_only:
        mention_type_names.append(type_name + '_')
        mention_group_concepts.append(mention_group[type_name + '_group'])

mention[tuple([mention_group_contains, 'MentionRepresentation', 'Context'] + mention_type_names)] = JointSensor(mention_group['MentionRepresentation_group'], mention_group['Context_group'], *mention_group_concepts, forward=make_mention_props)

# module learners
mention['encoded'] = ModuleLearner(
    'Context',
    'MentionRepresentation',
    module=CNNEncoder(
            pretrained_embeddings=file_data['embeddings']
        )
    )

class WeightedNBCrossEntropyDictLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, builder, prop, input, target, *args, **kwargs):
        input = input.view(-1, input.shape[-1])
        target = target.view(-1).to(dtype=torch.long, device=input.device)
        #self.weight = torch.tensor([1.0, self.class_weights[prop]])
        return super().forward(input, target, *args, **kwargs) * self.class_weights[prop]

# module learner predictions
loss_dict = {}
for i, (type_name, type_concept) in enumerate(TypenetGraph.concepts.items()):
    if not args.limit_classes == None and i >= args.limit_classes:
        print('stopped after adding %d classe(s)' % args.limit_classes)
        break
    if not type_name[:6] == 'Synset' or not config.freebase_only:
        mention[type_concept] = FunctionalSensor(mention_group_contains, type_name + '_', forward=lambda x, y: y, label=True)
        mention[type_concept] = ModuleLearner('encoded', module=TypeComparison(300, 2))

        loss_dict[type_name] = WeightedNBCrossEntropyDictLoss(train_class_weights)

'''program = Program(app_graph,
    loss=MacroAverageTracker(NBCrossEntropyLoss()),
    metric=PRF1Tracker(DatanodeCMMetric()))'''

# create program
'''program = SolverPOIProgram(app_graph,
    inferTypes=['ILP', 'local/argmax'],
    loss=MacroAverageTracker(NBCrossEntropyLoss()),
    metric={'ILP':PRF1Tracker(DatanodeCMMetric()), 'argmax':PRF1Tracker(DatanodeCMMetric('local/argmax'))})
'''

program = SolverPOIDictLossProgram(app_graph,
        inferTypes=['local/argmax'],
        dictloss=loss_dict)

'''program = POIProgram(app_graph,
                    inferTypes=['local/argmax'],
                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                    metric=PRF1Tracker(DatanodeCMMetric('local/argmax')))'''

'''program = IMLProgram(app_graph,
                     inferTypes=['ILP', 'local/argmax'],
                     loss=MacroAverageTracker(NBCrossEntropyIMLoss(lmbd=0.5)),
                     metric={'ILP':PRF1Tracker(DatanodeCMMetric()),
                             'argmax':PRF1Tracker(DatanodeCMMetric('local/argmax'))})'''
#program.after_train_epoch = [LossCallback(program)]

print('training')
# train
for epoch in range(1, args.epochs + 1):
    print("Epoch %d" % epoch)

    program.train(wiki_train, Optim=torch.optim.Adam, device=config.device)

    if epoch % config.test_interval == 0:
        program_test = SolverPOIDictLossProgram(app_graph,
                                                inferTypes=['local/argmax'],
                                                dictloss=loss_dict,
                                                metric=PRF1SepTracker(DatanodeCMMetric('local/argmax')))

        program_test.test(wiki_dev)

        metric_all = program_test.model.metric[''].value()

        total_f1 = 0.0
        total_acc = 0.0
        num_class = len(metric_all)

        tp_total = 0.0
        fp_total = 0.0

        for type_name, metric in metric_all.items():
            tp_total += metric['tp']
            fp_total += metric['fp']

            total_f1 += metric['F1']
            total_acc += metric['accuracy']

        macro_f1 = total_f1/num_class
        micro_f1 = tp_total/(tp_total + fp_total)
        avg_acc = total_acc/num_class

        print("Macro-F1:", macro_f1)
        print("Micro-F1:", micro_f1)
        print("Average Accuracy:", avg_acc)