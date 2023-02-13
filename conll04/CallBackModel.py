from itertools import repeat
from typing import Callable, List
from dataclasses import dataclass

from regr.program.lossprogram import PrimalDualProgram, SampleLossProgram
from regr.program.callbackprogram import ProgramStorageCallback, hook


class CallbackPrimalProgram(PrimalDualProgram):
    def default_before_train_step(self):
        if self.opt is not None:
            self.opt.zero_grad()

#     def default_after_train_step(self, output=None):
#         loss, *_ = output
#         if self.opt and loss:
#             loss.backward()
#             self.opt.step()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.before_train = None
        self.after_train = None
        self.before_train_epoch = None
        self.after_train_epoch = None
        self.before_train_step = None
        self.after_train_step = None
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


    def train_epoch(self, *args, **kwargs):
        hook(self.before_train_epoch)
        for _, output in zip(
            map(hook, repeat(self.before_train_step)),
            super().train_epoch(*args, **kwargs),
            ):
            hook(self.after_train_step, output)
            yield output
        hook(self.after_train_epoch)

    def test(self, *args, **kwargs):
        hook(self.before_test)
        super().test(*args, **kwargs)
        hook(self.after_test)

    def test_epoch(self, *args, **kwargs):
        hook(self.before_test_epoch)
        for _, output in zip(
            map(hook, repeat(self.before_test_step)),
            super().test_epoch(*args, **kwargs),
            ):
            hook(self.after_test_step, output)
            yield output
        hook(self.after_test_epoch)
        
        
class CallbackSamplingProgram(SampleLossProgram):
    def default_before_train_step(self):
        if self.opt is not None:
            self.opt.zero_grad()

#     def default_after_train_step(self, output=None):
#         loss, *_ = output
#         if self.opt and loss:
#             loss.backward()
#             self.opt.step()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.before_train = None
        self.after_train = None
        self.before_train_epoch = None
        self.after_train_epoch = None
        self.before_train_step = None
        self.after_train_step = None
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


    def train_epoch(self, *args, **kwargs):
        hook(self.before_train_epoch)
        for _, output in zip(
            map(hook, repeat(self.before_train_step)),
            super().train_epoch(*args, **kwargs),
            ):
            hook(self.after_train_step, output)
            yield output
        hook(self.after_train_epoch)

    def test(self, *args, **kwargs):
        hook(self.before_test)
        super().test(*args, **kwargs)
        hook(self.after_test)

    def test_epoch(self, *args, **kwargs):
        hook(self.before_test_epoch)
        for _, output in zip(
            map(hook, repeat(self.before_test_step)),
            super().test_epoch(*args, **kwargs),
            ):
            hook(self.after_test_step, output)
            yield output
        hook(self.after_test_epoch)