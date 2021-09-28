import logging
import torch
import numpy as np

from .program import LearningBasedProgram
from .model.lossModel import PrimalDualModel, SampleLosslModel

# Primal-dual need multiple backward through constraint loss.
# It requires retain_graph=True.
# Memory leak problem with `backward(create_graph=True)`
# https://github.com/pytorch/pytorch/issues/4661#issuecomment-596526199
# workaround with following functions based on `grad()`
def backward(loss, parameters):
    parameters = list(parameters)
    grads = torch.autograd.grad(outputs=loss, inputs=parameters, retain_graph=False)
    assert len(grads) == len(parameters)
    for grad, parameter in zip(grads, parameters):
        parameter.grad = grad

def unset_backward(parameters):
    for parameter in parameters:
        parameter.grad = None

def reverse_sign_grad(parameters, factor=-1.):
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad = factor * parameter.grad

class LossProgram(LearningBasedProgram):
    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model=PrimalDualModel, beta=1, **kwargs):
        super().__init__(graph, Model, **kwargs)
        
        self.copt = None
        self.beta = beta

    def to(self, device):
        super().to(device=device)
        if self.device is not None:
            self.model.to(self.device)

    def train(
        self,
        training_set,
        valid_set=None,
        test_set=None,
        # COptim=None,  # SGD only
        c_lr=0.05,
        c_momentum=0.9,
        c_warmup_iters=100,  # warmup
        c_freq=10,
        c_freq_increase=5,  # d
        c_freq_increase_freq=1,
        c_lr_decay=4,  # strategy
        c_lr_decay_param=1,  # param in the strategy
        **kwargs):
        
        # if COptim is None:
        #     COptim = Optim
        # if COptim is not None and list(self.model.parameters()):
        #     self.copt = COptim(self.model.parameters())
        if list(self.model.parameters()):
            self.copt = torch.optim.SGD(self.model.parameters(), lr=c_lr, momentum=c_momentum)
        else:
            self.copt = None
            
        # To provide a session cache for cross-epoch variables like iter-count
        c_session = {'iter':0, 'c_update_iter':0, 'c_update_freq':c_freq, 'c_update':0}  
        
        return super().train(
            training_set,
            valid_set=valid_set,
            test_set=test_set,
            c_lr=c_lr,
            c_warmup_iters=c_warmup_iters,
            c_freq_increase=c_freq_increase,
            c_freq_increase_freq=c_freq_increase_freq,
            c_lr_decay=c_lr_decay,
            c_lr_decay_param=c_lr_decay_param,
            c_session=c_session,
            **kwargs)

    def train_epoch(
        self, dataset,
        c_lr=1,
        c_warmup_iters=100,  # warmup
        c_freq_increase=1,  # d
        c_freq_increase_freq=1,
        c_lr_decay=0,  # strategy
        c_lr_decay_param=1,
        c_session={},
        **kwargs):
        assert c_session
        iter = c_session['iter']
        c_update_iter = c_session['c_update_iter']
        c_update_freq = c_session['c_update_freq']
        c_update = c_session['c_update']
        self.model.train()
        self.model.reset()
        self.model.train()
        self.model.reset()
        for data in dataset:
            if self.opt is not None:
                self.opt.zero_grad()
            if self.copt is not None:
                self.copt.zero_grad()
            mloss, metric, *output = self.model(data)  # output = (datanode, builder)
            if iter < c_warmup_iters:
                loss = mloss
            else:
                closs, *_ = self.model(output[1])
                loss = mloss + self.beta * closs
            if self.opt is not None and loss:
                loss.backward()
                self.opt.step()
                iter += 1
            if (
                self.copt is not None and
                loss and
                iter > c_warmup_iters and
                iter - c_update_iter > c_update_freq
            ):
                # NOTE: Based on the face the gradient of lambda in dual
                #       is reversing signs of their gradient in primal,
                #       we avoid a repeated backward (and also pytorch's
                #       retain_graph problem), we simply reverse the sign
                #       for the dual. Don't zero_grad() here!
                reverse_sign_grad(self.model.parameters())
                self.copt.step()
                # update counting
                c_update_iter = iter
                c_update += 1
                # update freq
                if c_freq_increase_freq > 0 and c_update % c_freq_increase_freq == 0:
                    c_update_freq += c_freq_increase
                # update c_lr
                if c_lr_decay == 0:
                    # on the paper
                    # c_lr_decay_param = beta
                    def update_lr(lr):
                        return c_lr * 1. / (1 + c_lr_decay_param * c_update)
                elif c_lr_decay == 1:
                    # in pd code / srl strategy 1
                    # c_lr_decay_param = lr_decay_after
                    def update_lr(lr):
                        return lr * np.sqrt(((c_update-1.) / c_lr_decay_param + 1.) / (c_update / c_lr_decay_param + 1.))
                elif c_lr_decay == 2:
                    # in pd code / srl strategy 2
                    # c_lr_decay_param = lr_decay_after
                    def update_lr(lr):
                        return lr * (((c_update-1.) / c_lr_decay_param + 1.) / (c_update / c_lr_decay_param + 1.))
                elif c_lr_decay == 3:
                    # in pd code / srl strategy 3
                    # c_lr_decay_param = lr_decay_after
                    assert c_lr_decay_param <= 1.
                    def update_lr(lr):
                        return lr * c_lr_decay_param
                elif c_lr_decay == 4:
                    # in pd code / ner strategy 1
                    def update_lr(lr):
                        return lr * np.sqrt((c_update+1) / (c_update+2))
                else:
                    raise ValueError(f'c_lr_decay={c_lr_decay} not supported.')
                for param_group in self.copt.param_groups:
                    param_group['lr'] = update_lr(param_group['lr'])
            yield (loss, metric, *output[:1])

        c_session['iter'] = iter
        c_session['c_update_iter'] = c_update_iter
        c_session['c_update_freq'] = c_update_freq
        c_session['c_update'] = c_update

    def test_epoch(self, dataset, **kwargs):
        # just to consum kwargs
        yield from super().test_epoch(dataset)
        
class PrimalDualProgram(LossProgram):
    logger = logging.getLogger(__name__)

    def __init__(self, graph, beta=1, **kwargs):
        super().__init__(graph, Model=PrimalDualModel, **kwargs)
        
class SampleLossProgram(LossProgram):
    logger = logging.getLogger(__name__)

    def __init__(self, graph, beta=1, **kwargs):
        super().__init__(graph, Model = SampleLosslModel, **kwargs)