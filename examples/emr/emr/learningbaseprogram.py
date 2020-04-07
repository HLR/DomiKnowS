import torch
from tqdm import tqdm

from .graph.torch import TorchModel
from .utils import seed, consume, print_result


class LearningBasedProgram():
    def __init__(self, graph, **config):
        self.graph = graph
        self.model = TorchModel(graph, **config)

    def train(self, training_set, valid_set=None, test_set=None, config=None):
        if config.device is not None:
            self.model.to(config.device)
        if list(self.model.parameters()):
            opt = config.opt(self.model.parameters())
        else:
            opt = None
        for epoch in range(config.epoch):
            print('Epoch:', epoch)

            print('Training:')
            consume(tqdm(self.train_epoch(training_set, opt, config.train_inference), total=len(training_set)))
            print_result(self.model, epoch, 'Training')

            if valid_set is not None:
                print('Validation:')
                consume(tqdm(self.test(valid_set, config.valid_inference), total=len(valid_set)))
                print_result(self.model, epoch, 'Validation')

        if test_set is not None:
            print('Testing:')
            consume(tqdm(self.test(test_set, config.valid_inference), total=len(test_set)))
            print_result(self.model, epoch, 'Testing')

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
