import logging
import torch
from tqdm import tqdm

from .utils import consume, print_reformat
from .graph.primal_dual_model import PrimalDualModel


class LearningBasedProgram():
    logger = logging.getLogger(__name__)

    def __init__(self, graph, config):
        self.graph = graph
        self.model = config.model(graph)
        self.opt = None

    def train(self, training_set=None, valid_set=None, test_set=None, config=None):
        if config.device is not None:
            self.model.to(config.device)
        if list(self.model.parameters()):
            self.opt = config.opt(self.model.parameters())
        else:
            self.opt = None
        for epoch in range(config.epoch):
            self.logger.info('Epoch: %d', epoch)

            if training_set is not None:
                self.logger.info('Training:')
                consume(tqdm(self.train_epoch(training_set, config.train_inference), total=len(training_set), desc='Epoch {} Training'.format(epoch)))
                self.logger.info(' - loss:')
                self.print_metric(self.model.loss)
                self.logger.info(' - metric:')
                self.print_metric(self.model.metric)

            if valid_set is not None:
                self.logger.info('Validation:')
                consume(tqdm(self.test(valid_set, config.valid_inference), total=len(valid_set), desc='Epoch {} Validation'.format(epoch)))
                self.logger.info(' - loss:')
                self.print_metric(self.model.loss)
                self.logger.info(' - metric:')
                self.print_metric(self.model.metric)

        if test_set is not None:
            self.logger.info('Testing:')
            consume(tqdm(self.test(test_set, config.valid_inference), total=len(test_set), desc='Epoch {} Testing'.format(epoch)))
            self.logger.info(' - loss:')
            self.print_metric(self.model.loss)
            self.logger.info(' - metric:')
            self.print_metric(self.model.metric)

    def print_metric(self, metric):
        for (pred, _), value in metric.value().items():
            self.logger.info('   - %s: %s', pred.sup.prop_name.name, print_reformat(value))

    def train_epoch(self, dataset, inference=False):
        self.model.train()
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

class PrimalDualLearningBasedProgram(LearningBasedProgram):
    def __init__(self, graph, config):
        super().__init__(graph, config)
        self.cmodel = PrimalDualModel(graph)
        self.copt = None

    def train(self, training_set=None, valid_set=None, test_set=None, config=None):
        if config.device is not None:
            self.cmodel.to(config.device)
        if list(self.cmodel.parameters()):
            self.copt = config.copt(self.cmodel.parameters())
        else:
            self.copt = None
        return super().train(training_set, valid_set, test_set, config)

    def train_epoch(self, dataset, inference=False):
        self.model.train()
        self.cmodel.train()
        for data in dataset:
            if self.opt is not None:
                self.opt.zero_grad()
            if self.copt is not None:
                self.copt.zero_grad()
            loss, metric, output = self.model(data, inference=inference)
            closs, coutput = self.cmodel(output)
            if self.opt is not None:
                loss.backward()
                self.opt.step()
            if self.copt is not None:
                closs.backward()
                self.copt.step()
            yield loss + closs, metric, output
