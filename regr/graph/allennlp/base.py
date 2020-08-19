import os
import logging
import pickle
from glob import glob
from collections import defaultdict
from itertools import chain
from typing import Dict, List, Tuple, Union
import torch
import numpy as np
import pandas as pd
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.common.params import Params
from allennlp.training import Trainer
from allennlp.training.optimizers import Optimizer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.nn.util import device_mapping
from ..trial import Trial
from ...utils import WrapperMetaClass
from ...solver import ilpOntSolverFactory
from ...solver.ilpOntSolverFactory import ilpOntSolverFactory
from ...solver.ilpOntSolver import ilpOntSolver
from ...solver.allennlpInferenceSolver import AllennlpInferenceSolver
from ...solver.allennlplogInferenceSolver import AllennlplogInferenceSolver
from ...sensor.allennlp.base import ReaderSensor
from ...sensor.allennlp.learner import SentenceEmbedderLearner
from ...sensor.allennlp.sensor import SentenceEmbedderSensor
from .. import Graph, Property
from ..dataNode import DataNode
from .model import GraphModel
from .utils import evaluate


DEBUG_TRAINING = 'REGR_DEBUG' in os.environ and os.environ['REGR_DEBUG']


class AllenNlpGraph(Graph, metaclass=WrapperMetaClass):
    __metaclass__ = WrapperMetaClass

    def __init__(
        self,
        *args,
        log_solver=False,
        **kwargs
    ):
        vocab = None # Vocabulary()
        self.model = GraphModel(self, vocab, *args, **kwargs)
        if log_solver:
            self.solver = ilpOntSolverFactory.getOntSolverInstance(self, AllennlplogInferenceSolver)
        else:
            self.solver = ilpOntSolverFactory.getOntSolverInstance(self, AllennlpInferenceSolver)
        self.solver_log_to(None)
        # do not invoke super().__init__() here

    def get_multiassign(self):
        def func(node):
            if isinstance(node, Property) and len(node) > 1:
                return node
            return None
        return list(self.traversal_apply(func))

    @property
    def poi(self):
        return self.get_multiassign()

    def get_sensors(self, *tests):
        def func(node):
            if isinstance(node, Property):
                yield from node.find(*tests)
        return list(self.traversal_apply(func))

    def solver_log_to(self, log_path:str=None):
        #solver_logger = logging.getLogger(ilpOntSolver.__module__)
        solver_logger = self.solver.myLogger
        solver_logger.propagate = False
        if DEBUG_TRAINING:
            solver_logger.setLevel(logging.DEBUG)
            param = {'precision': 4,
                     'threshold': np.inf,
                     'edgeitems': 3,
                     'linewidth': np.inf
                    }
            torch.set_printoptions(**param)
            np.set_printoptions(**param)
            pd.options.display.max_rows = None
            pd.options.display.max_columns = None
        else:
            solver_logger.setLevel(logging.INFO)
        solver_logger.handlers = []
        if log_path is not None:
            if DEBUG_TRAINING:
                handler = logging.FileHandler(log_path)
            else:
                logFilesize = 5 * 1024 * 1024 * 1024 # 5G
                logBackupCount = 4
                logFileMode = 'a'
                handler = logging.handlers.RotatingFileHandler(log_path,
                                                               mode=logFileMode,
                                                               maxBytes=logFilesize,
                                                               backupCount=logBackupCount,
                                                               encoding=None,
                                                               delay=0)
            solver_logger.addHandler(handler)

    def save(self, path, **objects):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, 'model.th'), 'wb') as fout:
            torch.save(self.model.state_dict(), fout)
        self.model.vocab.save_to_files(os.path.join(path, 'vocab'))
        for k, v in objects.items():
            with open(os.path.join(path, k + '.pkl'), 'wb') as fout:
                pickle.dump(v, fout)

    def load(self, path, model:Union[str, int]='default', vocab=None):
        model_file = None
        if isinstance(model, int):
            model_file = 'model_state_epoch_{}'.format(model)
        elif isinstance(model, str):
            if model.endswith('.th'):
                model_file = model
            elif model == 'best':
                model_file = 'best.th'
            elif model == 'last':
                model_file = sorted(glob(os.path.join(path, "model_state_epoch_*.th")), key=lambda p: os.path.getmtime(p)).pop()
            elif model == 'default':
                model_file = 'model.th'
        if model_file is None:
            raise ValueError(('`model` in `load()` must be one of the following: '
                              'an integer for epoch number, '
                              'a string name ends with ".th", '
                              'the string "best" for best model, '
                              'the string "last" for last saved model, '
                              'or the string "default" for just "model.th".'))
        vocab = vocab or os.path.join(path, 'vocab')

        if torch.cuda.is_available() and not DEBUG_TRAINING:
            device = 0
            self.model.cuda()
        else:
            device = -1

        #print('Loading vocab from {}'.format(vocab))
        self.model.vocab = Vocabulary.from_files(vocab)
        self.model.extend_embedder_vocab()
        #print('Loading model from {}'.format(model_file))
        with open(model_file, 'rb') as fin:
            self.model.load_state_dict(torch.load(fin, map_location=device_mapping(device)))

    @property
    def reader(self):
        readers = set(self.get_sensors(ReaderSensor))
        assert len(readers) == 1 # consider only 1 reader now
        return readers.pop()

    def train(self, data_config, train_config):
        reader = self.reader
        train_dataset = reader.read(os.path.join(data_config.relative_path, data_config.train_path), metas={'dataset_type':'train'})
        valid_dataset = reader.read(os.path.join(data_config.relative_path, data_config.valid_path), metas={'dataset_type':'valid'})
        self.update_vocab_from_instances(train_dataset + valid_dataset, train_config.pretrained_files)

        # prepare optimizer
        #print({n: p.size() for n, p in self.model.named_parameters()})
        optimizer = Optimizer.from_params(self.model.named_parameters(), Params(train_config.optimizer))

        # prepare scheduler
        if 'scheduler' in train_config:
            scheduler = LearningRateScheduler.from_params(optimizer, Params(train_config.scheduler))
        else:
            scheduler = None

        # prepare iterator
        sorting_keys = [(sensor.fullname, 'num_tokens') for sensor in self.get_sensors(SentenceEmbedderSensor)]
        iterator = BucketIterator(sorting_keys=sorting_keys,
                                  track_epoch=True,
                                  **train_config.iterator)
        iterator.index_with(self.model.vocab)

        # prepare model
        training_state = self.model.training
        self.model.train()

        trainer = self.get_trainer(train_dataset, valid_dataset,
                                   optimizer=optimizer,
                                   learning_rate_scheduler=scheduler,
                                   iterator=iterator,
                                   **train_config.trainer)

        if train_config.trainer.serialization_dir is not None:
            self.solver_log_to(os.path.join(train_config.trainer.serialization_dir, 'solver.log'))

        metrics = trainer.train()

        # restore model
        self.model.train(training_state)

        return metrics

    def get_trainer(
        self,
        train_dataset,
        valid_dataset,
        summary_interval=100,
        histogram_interval=100,
        should_log_parameter_statistics=True,
        should_log_learning_rate=True,
        **kwargs
    ) -> Trainer:
        # prepare GPU
        if torch.cuda.is_available() and not DEBUG_TRAINING:
            device = 0
            self.model.cuda()
        else:
            device = -1

        trainer = Trainer(model=self.model,
                          train_dataset=train_dataset,
                          validation_dataset=valid_dataset,
                          shuffle=not DEBUG_TRAINING,
                          cuda_device=device,
                          summary_interval=summary_interval,
                          histogram_interval=histogram_interval,
                          should_log_parameter_statistics=should_log_parameter_statistics,
                          should_log_learning_rate=should_log_learning_rate,
                          **kwargs)

        return trainer

    def update_vocab_from_instances(self, instances, pretrained_files=None):
        #import pdb; pdb.set_trace()
        vocab = Vocabulary.from_instances(instances, pretrained_files=pretrained_files)
        self.model.vocab = vocab
        self.model.extend_embedder_vocab()

    def test(self, data_path, log_path=None, batch=1, log_violation=False):
        # prepare GPU
        if torch.cuda.is_available() and not DEBUG_TRAINING:
            device = 0
            self.model.cuda()
        else:
            device = -1

        reader = self.reader
        dataset = reader.read(data_path, metas={'dataset_type':'test'})

        # prepare iterator
        sentence_sensors = self.get_sensors(SentenceEmbedderLearner)
        sorting_keys = [(sensor.fullname, 'num_tokens') for sensor in sentence_sensors]
        iterator = BucketIterator(batch_size=batch,
                                  sorting_keys=sorting_keys,
                                  track_epoch=True)
        iterator.index_with(self.model.vocab)

        # prepare model
        training_state = self.model.training
        self.model.eval()

        self.solver_log_to(log_path)
        metrics = evaluate(model=self.model,
                           instances=dataset,
                           data_iterator=iterator,
                           cuda_device=device,
                           batch_weight_key=None,
                           log_violation=log_violation)

        # restore model
        self.model.train(training_state)

        return metrics

    def populate(self, data_item, root, *concepts, query=None, label_fn=None, data_dict=None):
        trial = Trial()
        if data_dict is None:
            data_dict = defaultdict(list)

        def peel(node_value, repeat_unpeelable=False):
            # get a iterator for each instance in the fist level
            # {a:[1,2,3], b:[[4.1,4.2],[5.1,5.2],[6.1,6.2]], c:{d:[7,8,9]}}
            # ->
            # [{a:1, b:[4.1,4.2], c{d:7}}, {a:2, b:[5.1,5.2], c{d:8}}, {a:3, b:[6.1,6.2], c{d:9}}]
            if isinstance(node_value, (torch.Tensor, np.ndarray, List, Tuple)):
                return iter(node_value)
            elif isinstance(node_value, (Dict,)):
                pvs = []
                keys = []
                for k, v in node_value.items():
                    pv = peel(v, repeat_unpeelable)
                    if pv:
                        pvs.append(pv)
                        keys.append(k)
                return (dict(zip(keys, zpv)) for zpv in zip(*pvs))
            # TODO: fields like "loss" fron allennlp might not be possible to be divided
            # according to dim(/batch). Return None or return the identical one for all?
            elif repeat_unpeelable:
                return repeat(node_value)
            return None

        def contain(node, k):
            return k.startswith(node.fullname)

        def contained_by(node):
            return lambda item: contain(node, item[0])

        root_data = []
        sub_data_item = {}
        for k, v in filter(contained_by(self), data_item.items()):
            node = self[k[len(self.fullname)+1:]]
            if not isinstance(node, Property): # skip sensors and learners
                continue
            if contain(root, k):
                if not root_data:
                    for index, _ in enumerate(peel(v)):
                        dataNode = DataNode(instanceID=len(root_data),
                                            instanceValue=None,
                                            ontologyNode=root)
                        dataNode.index = index
                        root_data.append(dataNode)
                #import pdb; pdb.set_trace()
                for dataNode, pv in zip(root_data, peel(v)):
                    dataNode.__dict__[node.prop_name] = pv
            else:
                sub_data_item[k] = v
        if label_fn:
            for dataNode in root_data:
                prob = label_fn(dataNode)
                if prob is not None:
                    trial[root, dataNode] = prob
        data_dict[root].extend(root_data)
        for rel in root.contains():
            pass
            #for dataNode in data_dict[concept]:
            #    dataNode
#             for rel in concept.is_a():
#                 pass
#             for rel in concept.has_a():
#                 pass
        return trial, data_dict

#     @staticmethod
#     def peel(data_item, index):
#         # get a new data_item by selecting index for first dim
#         def peel_one(node_value):
#             # closure variable - index
#             if isinstance(node_value, (torch.Tensor, List, Tuple)):
#                 return node_value[index]
#             elif isinstance(node_value, (Dict,)):
#                 return {k, peel_one(v, index): for k, v: node_value.items()}
#             # TODO: fields like "loss" fron allennlp might not be possible to be divided
#             # according to dim(/batch). Return None or return the identical one for all?
#             return None
#         peeled = {}
#         for k, v in data_item:
#             v = peel_one(v)
#             if v is not None:
#                 peeled[k] = v
#         return peeled

#     @staticmethod
#     def peeled(data_item):
#         keys, values = zip(*data_item.items())
#         zip(values)
#         # get a new data_item by selecting index for first dim
#         def peel_one(node_value):
#             # closure variable - index
#             if isinstance(node_value, (torch.Tensor, List, Tuple)):
#                 return node_value[index]
#             elif isinstance(node_value, (Dict,)):
#                 return {k, peel_one(v, index): for k, v: node_value.items()}
#             # TODO: fields like "loss" fron allennlp might not be possible to be divided
#             # according to dim(/batch). Return None or return the identical one for all?
#             return None
#         peeled = {}
#         for k, v in data_item:
#             v = peel_one(v)
#             if v is not None:
#                 peeled[k] = v
#         return peeled

#     def populate(self, data_item, *concepts, batch=None, query=None, key='label'):
#         from ..concept import Concept
#         from ..property import Property
#         from ..dataNode import DataNode

#         trial = Trial()
#         data_dict = defaultdict()

#         def contained(item):
#             k = item[0]
#             return k.startswith(self.fullname)

#         def extract_batch_value(node_value):
#             if batch is None:
#                 return node_value
#             # closure variables - batch
#             if isinstance(node_value, (torch.Tensor, List, Tuple)):
#                 return node_value[batch]
#             elif isinstance(node_value, (Dict,)):
#                 return {k, extract_batch(v): for k, v: node_value.items()}
#             return None # fields like "loss" fron allennlp might not be divided according to batch, or return the identical one for all?

#         for node_key, node_value in filter(contained, data_item.items()):
#             node = self[node_key]
#             if isinstance(node, Property):
#                 concept_node = node.sup
#                 if concept_node not in data_dict: # TODO: limit to *concepts?
#                     # create the nodes
#                     # get rid of batch
#                     node_value = extract_batch_value(node_value)
#                     if node_value is None:
#                         # skip if get none for batch
#                         continue
#                     data_dict[concept_node] = []
#                     # TODO: estimate the number of argument
#                     # FIXME: suppose 1 here, that means a single loop
#                     for _ in node_value:
#                         dataNode = DataNode(instanceID=len(data_dict[concept_node]),
#                                             instanceValue=None,
#                                             ontologyNode=concept_node)
#                         data_dict[concept_node].append(dataNode)
#                 # add a property
#                 # FIXME: same problem as above `for _ in node_value`
#                 for dataNode, value in zip(data_dict[concept_node], node_value):
#                     dataNode.__dict__[node.name] = value
#                     if node.name == key:
#                         # TODO: how to inteprete the value here then?
#                         # logistic output have two dim where [1] is the logit for being True
#                         trial[concept_node, dataNode] = value[-1]

#         # wire up
#         for concept in concepts:
#             for rel in concept.contains():
#                 for dataNode in data_dict[concept]:
#                     dataNode
#             for rel in concept.is_a():
#                 pass
#             for rel in concept.has_a():
#                 pass

#         return trial, data_dict
