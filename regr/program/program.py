import logging
import torch
from tqdm import tqdm

from ..utils import consume
from .model.base import Mode
from ..sensor.pytorch.sensors import TorchSensor


def get_len(dataset, default=None):
    try:
        return len(dataset)
    except TypeError:  # `generator` does not have __len__
        return default

class LearningBasedProgram():
    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, **kwargs):
        self.graph = graph
        self.model = Model(graph, **kwargs)
        self.opt = None

    def update_nominals(self, dataset):
        pass

    def to(self, device='auto'):
        if device == 'auto':
            is_cuda = torch.cuda.is_available()
            if is_cuda:
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        if self.device is not None:
            for sensor in self.graph.get_sensors(TorchSensor):
                sensor.device = self.device

    def train(
        self,
        training_set,
        valid_set=None,
        test_set=None,
        device=None,
        train_epoch_num=1,
        Optim=None):
        if device is not None:
            self.to(device)
        if Optim is not None and list(self.model.parameters()):
            self.opt = Optim(self.model.parameters())
        else:
            self.opt = None
        for epoch in range(train_epoch_num):
            self.logger.info('Epoch: %d', epoch)

            if training_set is not None:
                self.logger.info('Training:')
                consume(tqdm(self.train_epoch(training_set), total=get_len(training_set), desc='Epoch {} Training'.format(epoch)))
                self.logger.info(' - loss:')
                self.logger.info(self.model.loss)
                self.logger.info(' - metric:')
                self.logger.info(self.model.metric)

            if valid_set is not None:
                self.logger.info('Validation:')
                consume(tqdm(self.test_epoch(valid_set), total=get_len(valid_set), desc='Epoch {} Validation'.format(epoch)))
                self.logger.info(' - loss:')
                self.logger.info(self.model.loss)
                self.logger.info(' - metric:')
                self.logger.info(self.model.metric)

        if test_set is not None:
            self.logger.info('Testing:')
            consume(tqdm(self.test_epoch(test_set), total=get_len(test_set), desc='Epoch {} Testing'.format(epoch)))
            self.logger.info(' - loss:')
            self.logger.info(self.model.loss)
            self.logger.info(' - metric:')
            self.logger.info(self.model.metric)

    def train_epoch(self, dataset):
        self.model.mode(Mode.TRAIN)
        self.model.reset()
        for data_item in dataset:
            if self.opt is not None:
                self.opt.zero_grad()
            loss, metric, output = self.model(data_item)
            if self.opt and loss:
                loss.backward()
                self.opt.step()
            yield loss, metric, output

    def test(self, dataset, device=None):
        if device is not None:
            self.to(device)
        self.logger.info('Testing:')
        consume(tqdm(self.test_epoch(dataset), total=get_len(dataset), desc='Testing'))
        self.logger.info(' - loss:')
        self.logger.info(self.model.loss)
        self.logger.info(' - metric:')
        self.logger.info(self.model.metric)

    def test_epoch(self, dataset, device=None):
        self.model.mode(Mode.TEST)
        self.model.reset()
        with torch.no_grad():
            for data_item in dataset:
                loss, metric, output = self.model(data_item)
                yield loss, metric, output

    def populate(self, dataset, device=None):
        if device is not None:
            self.to(device)
        yield from self.populate_epoch(dataset, device)

    def populate_one(self, data_item, device=None):
        return next(self.populate([data_item], device))

    def populate_epoch(self, dataset, device=None):
        self.model.mode(Mode.POPULATE)
        self.model.reset()
        with torch.no_grad():
            for data_item in dataset:
                _, _, output = self.model(data_item)
                yield output
