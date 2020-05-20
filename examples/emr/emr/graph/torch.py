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

    def inference(self, context):
        context = self.solver.inferSelection(context, list(self.poi))
        return context

    def forward(self, context, inference=True):
        context = self.move(context)
        if inference:
            context = self.inference(context)
        return super().forward(context, inference)


class IMLModel(SolverModel):
    def poi_loss(self, context, prop, output_sensor, target_sensor):
        logit = output_sensor(context)
        mask = output_sensor.mask(context)
        labels = target_sensor(context)
        inference = prop(context)

        if self.loss:
            local_loss = self.loss[output_sensor, target_sensor](logit, inference, labels, mask)
            return local_loss
