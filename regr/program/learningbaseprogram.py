import logging
import torch
from tqdm import tqdm

from ..utils import consume


class LearningBasedProgram():
    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, **kwargs):
        self.graph = graph
        self.model = Model(graph)
        self.opt = None

    def update_nominals(self, dataset):
        pass

    def train(
        self,
        training_set,
        valid_set=None,
        test_set=None,
        train_inference=False,
        valid_inference=False,
        device=None,
        train_epoch_num=1,
        Optim=None):
        if device is not None:
            self.model.to(device)
        if Optim is not None and list(self.model.parameters()):
            self.opt = Optim(self.model.parameters())
        else:
            self.opt = None
        for epoch in range(train_epoch_num):
            self.logger.info('Epoch: %d', epoch)

            if training_set is not None:
                self.logger.info('Training:')
                consume(tqdm(self.train_epoch(training_set, train_inference), total=len(training_set), desc='Epoch {} Training'.format(epoch)))
                self.logger.info(' - loss:')
                self.logger.info(self.model.loss)
                self.logger.info(' - metric:')
                self.logger.info(self.model.metric)

            if valid_set is not None:
                self.logger.info('Validation:')
                consume(tqdm(self.test(valid_set, valid_inference), total=len(valid_set), desc='Epoch {} Validation'.format(epoch)))
                self.logger.info(' - loss:')
                self.logger.info(self.model.loss)
                self.logger.info(' - metric:')
                self.logger.info(self.model.metric)

        if test_set is not None:
            self.logger.info('Testing:')
            consume(tqdm(self.test(test_set, valid_inference), total=len(test_set), desc='Epoch {} Testing'.format(epoch)))
            self.logger.info(' - loss:')
            self.logger.info(self.model.loss)
            self.logger.info(' - metric:')
            self.logger.info(self.model.metric)

    def train_epoch(self, dataset, inference=False):
        self.model.train()
        self.model.reset()
        for data in dataset:
            if self.opt is not None:
                self.opt.zero_grad()
            loss, metric, output = self.model(data, inference=inference)
            if self.opt is not None:
                loss.backward()
                self.opt.step()
            yield loss, metric, output

    def test(self, dataset, inference=True):
        self.model.eval()
        self.model.reset()
        with torch.no_grad():
            for data in dataset:
                loss, metric, output = self.model(data, inference=inference)
                yield loss, metric, output

    def eval(self, dataset, inference=True):
        self.model.eval()
        self.model.reset()
        with torch.no_grad():
            for data in dataset:
                _, _, output = self.model(data, inference=inference)
                yield output

    def eval_one(self, data, inference=True):
        # TODO: extend one sample data to 1-batch data
        self.model.eval()
        self.model.reset()
        with torch.no_grad():
            _, _, output = self.model(data, inference=inference)
            return output
