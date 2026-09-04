from itertools import cycle, repeat
from typing import Callable, List
from dataclasses import dataclass

import torch

from ..utils import consume, entuple
from .model.base import Mode
from .program import LearningBasedProgram


class BatchProgram(LearningBasedProgram):
    def __init__(self, *args, batch_size=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def train_epoch(self, dataset):
        # do not use super() because it call zero_grad for every step definitely
        self.model.mode(Mode.TRAIN)
        self.model.reset()
        self.opt.zero_grad()
        for index, data_item in enumerate(dataset):
            with self._autocast_ctx():
                loss, metric, *output = self.model(data_item)
            if self.opt and torch.is_tensor(loss) and loss.requires_grad:
                # Gradient accumulation: backward every step, step/zero at
                # batch boundary. _backward_and_step handles AMP scaling.
                is_boundary = index % self.batch_size == self.batch_size - 1
                self._backward_and_step(loss, zero_grad=False, step=is_boundary)
                if is_boundary:
                    self.opt.zero_grad()
            yield (loss, metric, *output[:1])
