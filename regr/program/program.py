import logging
import torch
from tqdm import tqdm

from ..utils import consume, entuple, detuple
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
        test_every_epoch=False,
        Optim=None,
        train_callbacks={},
        valid_callbacks={},
        test_callbacks={},
        **kwargs):
        if device is not None:
            self.to(device)
        if Optim is not None and list(self.model.parameters()):
            self.opt = Optim(self.model.parameters())
        else:
            self.opt = None
        train_callback_storage = {}
        valid_callback_storage = {}
        test_callback_storage = {}
        self.stop = False
        self.train_epoch_num = train_epoch_num
        self.epoch = 0
        while self.epoch < self.train_epoch_num:
            self.epoch += 1
            self.logger.info('Epoch: %d', self.epoch)

            def epoch(name, dataset, epoch_fn, callbacks, callback_storage):
                if dataset is not None:
                    self.logger.info(f'{name}:')
                    if callbacks:
                        def callback():
                            for key, callback in callbacks.items():
                                storage = callback_storage.setdefault(key, tuple())
                                callback_storage[key] = callback(self, *entuple(storage))
                    else:
                        callback = None
                    consume(tqdm(epoch_fn(dataset, callback, **kwargs), total=get_len(dataset), desc=f'Epoch {self.epoch} {name}'))
                    if self.model.loss:
                        self.logger.info(' - loss:')
                        self.logger.info(self.model.loss)
                    if self.model.metric:
                        self.logger.info(' - metric:')
                        for key, metric in self.model.metric.items():
                            self.logger.info(f' - - {key}')
                            self.logger.info(metric)

            epoch('Training', training_set, self.train_epoch, train_callbacks, train_callback_storage)
            epoch('Validation', valid_set, self.test_epoch, valid_callbacks, valid_callback_storage)
            if test_every_epoch:
                epoch('Testing', test_set, self.test_epoch, test_callbacks, test_callback_storage)
        if not test_every_epoch:
            epoch('Testing', test_set, self.test_epoch, test_callbacks, test_callback_storage)

            
    def train_epoch(self, dataset, callback=None):
        self.model.mode(Mode.TRAIN)
        self.model.reset()
        for data_item in dataset:
            if self.opt is not None:
                self.opt.zero_grad()
            loss, metric, *output = self.model(data_item)
            if self.opt and loss:
                loss.backward()
                self.opt.step()
            yield (loss, metric, *output[:1])
        if callable(callback):
            callback()

    def test(self, dataset, device=None):
        if device is not None:
            self.to(device)
        self.logger.info('Testing:')
        consume(tqdm(self.test_epoch(dataset), total=get_len(dataset), desc='Testing'))
        self.logger.info(' - loss:')
        self.logger.info(self.model.loss)
        self.logger.info(' - metric:')
        for key, metric in self.model.metric.items():
                    self.logger.info(f' - - {key}')
                    self.logger.info(metric)

    def test_epoch(self, dataset, callback=None):
        self.model.mode(Mode.TEST)
        self.model.reset()
        with torch.no_grad():
            for data_item in dataset:
                loss, metric, *output = self.model(data_item)
                yield (loss, metric, *output[:1])
        if callable(callback):
            callback()

    def populate(self, dataset, device=None):
        if device is not None:
            self.to(device)
        yield from self.populate_epoch(dataset, device)

    def populate_epoch(self, dataset, callback=None):
        self.model.mode(Mode.POPULATE)
        self.model.reset()
        with torch.no_grad():
            for data_item in dataset:
                _, _, *output = self.model(data_item)
                yield detuple(*output[:1])
        if callable(callback):
            callback()

    def populate_one(self, data_item):
        return next(self.populate_epoch([data_item]))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
