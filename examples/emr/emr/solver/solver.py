import torch
from torch.nn import functional as F

from regr.solver.context_solver import ContextSolver

from ..sensor.learner import ModuleLearner
from ..sensor.sensor import DataSensor


class Solver(ContextSolver):
    def get_raw_input(self, data):
        graph = next(iter(self.myGraph))
        _, sentence_sensor = graph.get_sensors(DataSensor, lambda s: not s.target)[0]
        sentences = sentence_sensor(data)
        mask_len = [len(s) for s in sentences]  # (b, )
        return sentences, mask_len

    def get_prop_result(self, prop, data):
        graph = next(iter(self.myGraph))
        output_sensor, _ = graph.poi[prop]

        logit = output_sensor(data)
        #score = -F.logsigmoid(logit)
        score = torch.sigmoid(logit)
        mask = output_sensor.mask(data)
        return score, mask

    def set_prop_result(self, prop, data, value):
        data[prop.fullname] = value
