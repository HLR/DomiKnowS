import logging
import torch
from tqdm import tqdm

from ..utils import consume, entuple, detuple
from .model.base import Mode
from ..sensor.pytorch.sensors import TorchSensor


class ProgramStorageCallback():
    def __init__(self, program, fn) -> None:
        self.program = program
        self.fn = fn
        self.storage = tuple()

    def __call__(self):
        self.storage = self.fn(self.program, *entuple(self.storage))


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
        self.epoch = None
        self.stop = None

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

    def call_epoch(self, name, dataset, epoch_fn, epoch_callbacks=None, step_callbacks=None, **kwargs):
        if dataset is not None:
            self.logger.info(f'{name}:')
            desc = name if self.epoch is None else f'Epoch {self.epoch} {name}'
            for _ in tqdm(epoch_fn(dataset, **kwargs), total=get_len(dataset), desc=desc):
                if step_callbacks: consume(callback() for callback in step_callbacks)
            if epoch_callbacks: consume(callback() for callback in epoch_callbacks)
            if self.model.loss:
                self.logger.info(' - loss:')
                self.logger.info(self.model.loss)
            if self.model.metric:
                self.logger.info(' - metric:')
                for key, metric in self.model.metric.items():
                    self.logger.info(f' - - {key}')
                    self.logger.info(metric)

    def train(
        self,
        training_set,
        valid_set=None,
        test_set=None,
        device=None,
        train_epoch_num=1,
        test_every_epoch=False,
        Optim=None,
        train_epoch_callbacks=None,
        valid_epoch_callbacks=None,
        test_epoch_callbacks=None,
        train_step_callbacks=None,
        valid_step_callbacks=None,
        test_step_callbacks=None,
        **kwargs):
        if device is not None:
            self.to(device)
        if Optim is not None and list(self.model.parameters()):
            self.opt = Optim(self.model.parameters())
        else:
            self.opt = None
        self.train_epoch_num = train_epoch_num
        self.epoch = 0
        self.stop = False
        while self.epoch < self.train_epoch_num and not self.stop:
            self.epoch += 1
            self.logger.info('Epoch: %d', self.epoch)
            self.call_epoch('Training', training_set, self.train_epoch, train_epoch_callbacks, train_step_callbacks, **kwargs)
            self.call_epoch('Validation', valid_set, self.test_epoch, valid_epoch_callbacks, valid_step_callbacks, **kwargs)
            if test_every_epoch:
                self.call_epoch('Testing', test_set, self.test_epoch, test_epoch_callbacks, test_step_callbacks, **kwargs)
        if not test_every_epoch:
            self.call_epoch('Testing', test_set, self.test_epoch, test_epoch_callbacks, test_step_callbacks, **kwargs)
        # reset epoch after everything
        self.epoch = None
        self.stop = None

    def train_epoch(self, dataset):
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

    def test(self, dataset, device=None, callbacks={}, **kwargs):
        if device is not None:
            self.to(device)
        callback_storage = {}
        self.call_epoch('Testing', dataset, self.test_epoch, callbacks, callback_storage, **kwargs)

    def test_epoch(self, dataset):
        self.model.mode(Mode.TEST)
        self.model.reset()
        with torch.no_grad():
            for data_item in dataset:
                loss, metric, *output = self.model(data_item)
                yield (loss, metric, *output[:1])

    def populate(self, dataset, device=None):
        if device is not None:
            self.to(device)
        yield from self.populate_epoch(dataset)

    def populate_epoch(self, dataset):
        self.model.mode(Mode.POPULATE)
        self.model.reset()
        with torch.no_grad():
            for data_item in dataset:
                _, _, *output = self.model(data_item)
                yield detuple(*output[:1])


    def populate_one(self, data_item):
        return next(self.populate_epoch([data_item]))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
