from itertools import combinations

import torch

from regr.graph import Property
from regr.program.model.torch import PoiModel


class SolverModel(PoiModel):
    def __init__(self, graph, loss=None, metric=None, Solver=None):
        super().__init__(graph, loss, metric)
        if Solver:
            self.solver = Solver(self.graph)
        else:
            self.solver = None

    def inference(self, data_item):
        data_item = self.solver.inferSelection(data_item, list(self.poi))
        return data_item

    def forward(self, data_item, inference=True):
        data_item = self.move(data_item)
        if inference:
            data_item = self.inference(data_item)
        return super().forward(data_item, inference)


class IMLModel(SolverModel):
    def poi_loss(self, data_item, prop, output_sensor, target_sensor):
        logit = output_sensor(data_item)
        mask = output_sensor.mask(data_item)
        labels = target_sensor(data_item)
        inference = prop(data_item)

        if self.loss:
            local_loss = self.loss[output_sensor, target_sensor](logit, inference, labels, mask)
            return local_loss
