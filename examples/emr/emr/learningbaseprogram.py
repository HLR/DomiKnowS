import logging
import torch
from tqdm import tqdm

from .utils import consume, print_reformat


class LearningBasedProgram():
    logger = logging.getLogger(__name__)

    def __init__(self, graph, config):
        self.graph = graph
        self.model = config.model(graph)

    def train(self, training_set=None, valid_set=None, test_set=None, config=None):
        if config.device is not None:
            self.model.to(config.device)
        if list(self.model.parameters()):
            opt = config.opt(self.model.parameters())
        else:
            opt = None
        for epoch in range(config.epoch):
            self.logger.info('Epoch: %d', epoch)

            if training_set is not None:
                self.logger.info('Training:')
                consume(tqdm(self.train_epoch(training_set, opt, config.train_inference), total=len(training_set)))
                self.logger.info(' - loss:')
                self.print_metric(self.model.loss)
                self.logger.info(' - metric:')
                self.print_metric(self.model.metric)

            if valid_set is not None:
                self.logger.info('Validation:')
                consume(tqdm(self.test(valid_set, config.valid_inference), total=len(valid_set)))
                self.logger.info(' - loss:')
                self.print_metric(self.model.loss)
                self.logger.info(' - metric:')
                self.print_metric(self.model.metric)

        if test_set is not None:
            self.logger.info('Testing:')
            consume(tqdm(self.test(test_set, config.valid_inference), total=len(test_set)))
            self.logger.info(' - loss:')
            self.print_metric(self.model.loss)
            self.logger.info(' - metric:')
            self.print_metric(self.model.metric)

    def print_metric(self, metric):
        for (pred, _), value in metric.value().items():
            self.logger.info('   - %s: %s', pred.sup.prop_name.name, print_reformat(value))

    def train_epoch(self, dataset, opt=None, inference=False):
        self.model.train()
        for data in dataset:
            if opt is not None:
                opt.zero_grad()
            loss, metric, output = self.model(data, inference=inference)
            if opt is not None:
                loss.backward()
                opt.step()
            yield loss, metric, output

    def test(self, dataset, inference=True):
        self.model.eval()
        self.model.loss.reset()
        self.model.metric.reset()
        with torch.no_grad():
            for data in dataset:
                loss, metric, output = self.model(data, inference=inference)
                yield loss, metric, output

    def eval(self, dataset, inference=True):
        self.model.eval()
        self.model.loss.reset()
        self.model.metric.reset()
        with torch.no_grad():
            for data in dataset:
                _, _, output = self.model(data, inference=inference)
                yield output

    def eval_one(self, data, inference=True):
        # TODO: extend one sample data to 1-batch data
        self.model.eval()
        self.model.loss.reset()
        self.model.metric.reset()
        with torch.no_grad():
            _, _, output = self.model(data, inference=inference)
            return output
