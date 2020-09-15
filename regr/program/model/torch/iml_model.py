from itertools import combinations

import torch

from regr.graph import Property
from regr.program.model.torch import SolverModel


class IMLModel(SolverModel):
    def poi_loss(self, data_item, prop, output_sensor, target_sensor):
        logit = output_sensor(data_item)
        mask = output_sensor.mask(data_item)
        labels = target_sensor(data_item)
        inference = prop(data_item)

        if self.loss:
            local_loss = self.loss[output_sensor, target_sensor](logit, inference, labels, mask)
            return local_loss
