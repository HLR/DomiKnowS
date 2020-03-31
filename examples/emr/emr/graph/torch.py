from itertools import combinations

import torch
from tqdm import tqdm

from regr.graph.property import Property
from regr.solver.ilpOntSolverFactory import ilpOntSolverFactory
from regr.solver.nologInferenceSolver import NoLogInferenceSolver

from emr.sensor.sensor import DataSensor
from ..sensor.learner import TorchSensor, ModuleLearner
from ..utils import seed, consume, print_result


class Solver(NoLogInferenceSolver):
    def __init__(self, graph, ontologiesTuple, _ilpConfig,):
        def input_sensor_type(sensor):
            return isinstance(sensor, DataSensor) and not sensor.target
        super().__init__(graph, ontologiesTuple, _ilpConfig, input_sensor_type, ModuleLearner)

    def get_prop_result(self, prop, data):
        output_sensor, target_sensor = self.prop_dict[prop]

        logit = output_sensor(data)
        logit = torch.cat((1-logit, logit), dim=-1)
        mask = output_sensor.mask(data)
        labels = target_sensor(data)
        labels = labels.float()
        return labels, logit, mask

    def inferSelection(self, graph, data, prop_list, prop_dict):
        self.prop_dict = prop_dict
        return super().inferSelection(graph, data, prop_list)


def all_properties(node):
    if isinstance(node, Property):
        return node
    return None


class TorchModel(torch.nn.Module):
    def __init__(self, graph, loss, metric=None):
        super().__init__()
        self.graph = graph
        self.loss = loss
        self.metric = metric

        for node in self.graph.traversal_apply(all_properties):
            for _, sensor in node.find(ModuleLearner):
                self.add_module(sensor.fullname, sensor.module)

        self.solver = ilpOntSolverFactory.getOntSolverInstance(self.graph, Solver)

    def move(self, value, device=None):
        device = device or next(self.parameters()).device
        if isinstance(value, torch.Tensor):
            return value.to(device)
        elif isinstance(value, list):
            return [self.move(v, device) for v in value]
        elif isinstance(value, tuple):
            return (self.move(v, device) for v in value)
        elif isinstance(value, dict):
            return {k: self.move(v, device) for k, v in value.items()}
        else:
            return value

    def poi(self):
        for prop in self.graph.traversal_apply(all_properties):
            for (_, sensor1), (_, sensor2) in combinations(prop.find(TorchSensor), r=2):
                if sensor1.target:
                    target_sensor = sensor1
                    output_sensor = sensor2
                elif sensor2.target:
                    target_sensor = sensor2
                    output_sensor = sensor1
                else:
                    # TODO: should different learners get closer?
                    continue
                if output_sensor.target:
                    # two targets, skip
                    continue
                yield prop, output_sensor, target_sensor

    def forward(self, data):
        data = self.move(data)
        loss = 0
        metric = {}
        for prop, output_sensor, target_sensor in self.poi():
            output_sensor(data)
        data = self.inference(data)

        for prop, output_sensor, target_sensor in self.poi():
            logit = output_sensor(data)
            logit = logit.squeeze()
            mask = output_sensor.mask(data)
            labels = target_sensor(data)
            labels = labels.float()

            if self.loss:
                local_loss = self.loss[output_sensor, target_sensor](logit, labels, mask)
                loss += local_loss
            if self.metric:
                local_metric = self.metric[output_sensor, target_sensor](logit, labels, mask)
                metric[output_sensor, target_sensor] = local_metric
        return loss, metric, data

    def inference(self, data):
        prop_list = [prop for prop, _, _ in self.poi()]
        prop_dict = {prop: (output_sensor, target_sensor) for prop, output_sensor, target_sensor in self.poi()}

        data = self.solver.inferSelection(self.graph, data, prop_list=prop_list, prop_dict=prop_dict)
        return data

class LearningBasedProgram():
    def __init__(self, graph, **config):
        self.graph = graph
        self.model = TorchModel(graph, **config)

    def train(self, training_set, valid_set, config=None):
        self.model.cuda()
        seed()
        if list(self.model.parameters()):
            opt = torch.optim.Adam(self.model.parameters())
        else:
            opt = None
        for epoch in range(10):
            print('Epoch:', epoch)

            print('Training:')
            consume(tqdm(self.train_epoch(training_set, opt), total=len(training_set)))
            print_result(self.model, epoch, 'Training')

            print('Validation:')
            consume(tqdm(self.test(valid_set), total=len(valid_set)))
            print_result(self.model, epoch, 'Validation')

    def train_epoch(self, dataset, opt=None):
        self.model.train()
        for data in dataset:
            if opt is not None:
                opt.zero_grad()
            loss, metric, output = self.model(data)
            if opt is not None:
                loss.backward()
                opt.step()
            yield loss, metric, output

    def test(self, dataset):
        self.model.eval()
        self.model.loss.reset()
        self.model.metric.reset()
        with torch.no_grad():
            for data in dataset:
                loss, metric, output = self.model(data)
                yield loss, metric, output

    def eval(self, dataset):
        self.model.eval()
        self.model.loss.reset()
        self.model.metric.reset()
        with torch.no_grad():
            for data in dataset:
                _, _, output = self.model(data)
                yield output

    def eval_one(self, data):
        # TODO: extend one sample data to 1-batch data
        self.model.eval()
        self.model.loss.reset()
        self.model.metric.reset()
        with torch.no_grad():
            _, _, output = self.model(data)
            return output
