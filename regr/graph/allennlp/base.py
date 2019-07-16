import os
import logging
import logging.config
import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.training import Trainer
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
        balance_factor: float = 0.5,
        label_smoothing: float = 0.1,
        focal_gamma: float = 2.,
        inference_interval: int = 10
    ):
        vocab = None # Vocabulary()
        self.model = GraphModel(self, vocab,
                                balance_factor=balance_factor,
                                label_smoothing=label_smoothing,
                                focal_gamma=focal_gamma,
                                inference_interval=inference_interval)
        self.solver = ilpOntSolver.getInstance(self)
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

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, 'model.th'), 'wb') as fout:
            torch.save(self.model.state_dict(), fout)
        self.model.vocab.save_to_files(os.path.join(path, 'vocab'))

    def train(self, data_config, model_config, train_config):
        sentence_sensors = self.get_sensors(ReaderSensor)
        readers = {sensor.reader for name, sensor in sentence_sensors}
        assert len(readers) == 1 # consider only 1 reader now
        reader = readers.pop()
        train_dataset = reader.read(os.path.join(data_config.relative_path, data_config.train_path), metas={'dataset_type':'train'})
        valid_dataset = reader.read(os.path.join(data_config.relative_path, data_config.valid_path), metas={'dataset_type':'valid'})
        self.update_vocab_from_instances(train_dataset + valid_dataset, model_config.pretrained_files)
        trainer = self.get_trainer(train_dataset, valid_dataset, **train_config)

        solver_logger = logging.getLogger(ilpSelectClassification.__name__)
        solver_logger.propagate = False
        if DEBUG_TRAINING:
            solver_logger.setLevel(logging.DEBUG)
        else:
            solver_logger.setLevel(logging.INFO)
        solver_logger.handlers = []
        if train_config.serialization_dir is not None:
            handler = logging.FileHandler(os.path.join(train_config.serialization_dir, 'solver.log'))
            solver_logger.addHandler(handler)

        return trainer.train()

    def get_trainer(
        self,
        train_dataset,
        valid_dataset,
        lr=1., wd=0.003, batch=64, epoch=1000, patience=50,
        serialization_dir='tensorboard/',
        summary_interval=100,
        histogram_interval=100,
        should_log_parameter_statistics=True,
        should_log_learning_rate=True
    ) -> Trainer:
        # prepare GPU
        if torch.cuda.is_available() and not DEBUG_TRAINING:
            device = 0
            model = self.model.cuda()
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
                          should_log_learning_rate=should_log_learning_rate)

        return trainer

    def update_vocab_from_instances(self, instances, pretrained_files=None):
        #import pdb; pdb.set_trace()
        vocab = Vocabulary.from_instances(instances, pretrained_files=pretrained_files)
        self.model.vocab = vocab
        self.model.extend_embedder_vocab()
