import os
import logging
import pickle
import torch
import pandas as pd
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.training import Trainer
from allennlp.training.util import evaluate
from allennlp.nn.util import device_mapping
from ...utils import WrapperMetaClass
from ...solver import ilpSelectClassification
from ...solver.ilpSelectClassification import ilpOntSolver
from ...sensor.allennlp.base import ReaderSensor
from ...sensor.allennlp.learner import SentenceEmbedderLearner
from .. import Graph, Property
from .model import GraphModel


DEBUG_TRAINING = 'REGR_DEBUG' in os.environ and os.environ['REGR_DEBUG']


class AllenNlpGraph(Graph, metaclass=WrapperMetaClass):
    __metaclass__ = WrapperMetaClass

    def __init__(
        self,
        *args,
        **kwargs
    ):
        vocab = None # Vocabulary()
        self.model = GraphModel(self, vocab, *args, **kwargs)
        self.solver = ilpOntSolver.getInstance(self)
        self.solver_log_to(None)
        # do not invoke super().__init__() here

    def get_multiassign(self):
        ma = []

        def func(node):
            # use a closure to collect multi-assignments
            if isinstance(node, Property) and len(node) > 1:
                ma.append(node)
        self.traversal_apply(func)
        return ma

    @property
    def poi(self):
        return self.get_multiassign()

    def get_sensors(self, *tests):
        sensors = []

        def func(node):
            # use a closure to collect multi-assignments
            if isinstance(node, Property):
                sensors.extend(node.find(*tests))
        self.traversal_apply(func)
        return sensors

    def solver_log_to(self, log_path:str=None):
        solver_logger = logging.getLogger(ilpSelectClassification.__name__)
        solver_logger.propagate = False
        if DEBUG_TRAINING or True:
            solver_logger.setLevel(logging.DEBUG)
            pd.options.display.max_rows = None
            pd.options.display.max_columns = None
        else:
            solver_logger.setLevel(logging.INFO)
        solver_logger.handlers = []
        if log_path is not None:
            handler = logging.FileHandler(log_path)
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

    def load(self, path):
        if torch.cuda.is_available() and not DEBUG_TRAINING:
            device = 0
            self.model.cuda()
        else:
            device = -1

        self.model.vocab = Vocabulary.from_files(os.path.join(path, 'vocab'))
        self.model.extend_embedder_vocab()
        with open(os.path.join(path, 'model.th'), 'rb') as fin:
            self.model.load_state_dict(torch.load(fin, map_location=device_mapping(device)))

    @property
    def reader(self):
        sentence_sensors = self.get_sensors(ReaderSensor)
        readers = {sensor.reader for name, sensor in sentence_sensors}
        assert len(readers) == 1 # consider only 1 reader now
        return readers.pop()

    def train(self, data_config, train_config):
        reader = self.reader
        train_dataset = reader.read(os.path.join(data_config.relative_path, data_config.train_path), metas={'dataset_type':'train'})
        valid_dataset = reader.read(os.path.join(data_config.relative_path, data_config.valid_path), metas={'dataset_type':'valid'})
        self.update_vocab_from_instances(train_dataset + valid_dataset, train_config.pretrained_files)
        trainer = self.get_trainer(train_dataset, valid_dataset, **train_config.trainer)

        if train_config.trainer.serialization_dir is not None:
            self.solver_log_to(os.path.join(train_config.trainer.serialization_dir, 'solver.log'))

        return trainer.train()

    def get_trainer(
        self,
        train_dataset,
        valid_dataset,
        lr=1., wd=0.003, batch=64, epoch=1000, patience=50,
        serialization_dir='log/',
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

        # prepare optimizer
        #print([p.size() for p in model.parameters()])
        # options for optimizor: SGD, Adam, Adadelta, Adagrad, Adamax, RMSprop
        from torch.optim import Adam
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        sentence_sensors = self.get_sensors(SentenceEmbedderLearner)
        sorting_keys = [(sensor.fullname, 'num_tokens') for name, sensor in sentence_sensors]

        iterator = BucketIterator(batch_size=batch,
                                  sorting_keys=sorting_keys,
                                  track_epoch=True)
        iterator.index_with(self.model.vocab)
        from allennlp.training.learning_rate_schedulers import LearningRateScheduler
        from allennlp.common.params import Params
        scheduler = LearningRateScheduler.from_params(optimizer, Params({'type':'reduce_on_plateau'}))
        trainer = Trainer(model=self.model,
                          optimizer=optimizer,
                          iterator=iterator,
                          train_dataset=train_dataset,
                          validation_dataset=valid_dataset,
                          shuffle=not DEBUG_TRAINING,
                          patience=patience,
                          num_epochs=epoch,
                          cuda_device=device,
                          learning_rate_scheduler = scheduler,
                          serialization_dir=serialization_dir,
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

    def test(self, data_path, log_path=None, batch=1):
        # prepare GPU
        if torch.cuda.is_available() and not DEBUG_TRAINING:
            device = 0
            self.model.cuda()
        else:
            device = -1

        reader = self.reader
        dataset = reader.read(data_path, metas={'dataset_type':'test'})
        self.solver_log_to(log_path)
        sentence_sensors = self.get_sensors(SentenceEmbedderLearner)
        sorting_keys = [(sensor.fullname, 'num_tokens') for name, sensor in sentence_sensors]
        iterator = BucketIterator(batch_size=batch,
                                  sorting_keys=sorting_keys,
                                  track_epoch=True)
        iterator.index_with(self.model.vocab)

        final_metrics = evaluate(model=self.model,
                                 instances=dataset,
                                 data_iterator=iterator,
                                 cuda_device=device,
                                 batch_weight_key=None)

        return final_metrics
