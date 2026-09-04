from itertools import repeat
from typing import Callable, List
from dataclasses import dataclass

from ..utils import consume, entuple
from .model.base import Mode
from .program import LearningBasedProgram
import torch

class ProgramStorageCallback():
    def __init__(self, program, fn) -> None:
        self.program = program
        self.fn = fn
        self.storage = tuple()

    def __call__(self):
        self.storage = self.fn(self.program, *entuple(self.storage))


def hook(callbacks, *args, **kwargs):
    if callbacks:
        consume(callback(*args, **kwargs) for callback in callbacks)


class CallbackProgram(LearningBasedProgram):
    def default_before_train_step(self):
        if self.opt is not None:
            self.opt.zero_grad()

    def default_after_train_step(self, output=None):
        loss, *_ = output
        # Delegate to the AMP-/compile-aware helper on the base class.
        # zero_grad is handled by default_before_train_step, so skip it here.
        self._backward_and_step(loss, zero_grad=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.before_train = None
        self.after_train = None
        self.before_train_epoch = None
        self.after_train_epoch = None
        self.before_train_step = [self.default_before_train_step]
        self.after_train_step = [self.default_after_train_step]
        self.before_test = None
        self.after_test = None
        self.before_test_epoch = None
        self.after_test_epoch = None
        self.before_test_step = None
        self.after_test_step = None

    def train(self, *args, **kwargs):
        hook(self.before_train)
        super().train(*args, **kwargs)
        hook(self.after_train)

    def train_pure_epoch(self, dataset, **kwargs):
        """Pass kwargs through to parent."""
        self.model.mode(Mode.TRAIN)
        self.model.reset()
        for data_item in dataset:
            with self._autocast_ctx():
                loss, metric, *output = self.model(data_item)
            yield (loss, metric, *output[:1])

    def train_epoch(self, dataset, **kwargs):
        """Pass kwargs through and fire hooks."""
        hook(self.before_train_epoch)
        for _, output in zip(
            map(hook, repeat(self.before_train_step)),
            super().train_epoch(dataset, **kwargs),  # Pass kwargs to parent
            ):
            hook(self.after_train_step, output)
            yield output
        hook(self.after_train_epoch)

    def test(self, *args, **kwargs):
        hook(self.before_test)
        super().test(*args, **kwargs)
        hook(self.after_test)

    def test_epoch(self, dataset, **kwargs):
        """Pass kwargs through and fire hooks."""
        hook(self.before_test_epoch)
        for _, output in zip(
            map(hook, repeat(self.before_test_step)),
            super().test_epoch(dataset, **kwargs),  # Pass kwargs to parent
            ):
            hook(self.after_test_step, output)
            yield output
        hook(self.after_test_epoch)
