import torch
from torch.nn import functional as F

from regr.solver.context_solver import ContextSolver
from regr.sensor.torch.sensor import DataSensor
from regr.sensor.torch.learner import ModuleLearner


class Solver(ContextSolver):
    def get_raw_input(self, data_item):
        graph = next(iter(self.myGraph))
        _, sentence_sensor = graph.get_sensors(DataSensor, lambda s: not s.target)[0]
        sentences = sentence_sensor(data_item)
        mask_len = [len(s) for s in sentences]  # (b, )
        return sentences, mask_len

    def get_prop_result(self, data_item, prop):
        graph = next(iter(self.myGraph))
        output_sensor, _ = graph.poi[prop]

        logit = output_sensor(data_item)
        #score = -F.logsigmoid(logit)
        score = torch.sigmoid(logit)
        mask = output_sensor.mask(data_item)
        return score, mask

    def set_prop_result(self, data_item, prop, value):
        data_item[prop.fullname] = value
