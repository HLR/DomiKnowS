import logging
import torch

from .program import LearningBasedProgram
from .model.primaldual import PrimalDualModel


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

class PrimalDualProgram(LearningBasedProgram):
    DEFAULTCMODEL = PrimalDualModel

    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, CModel=None, beta=1, **kwargs):
        super().__init__(graph, Model, **kwargs)
        if CModel is None:
            CModel = self.DEFAULTCMODEL
        self.cmodel = CModel(graph)
        self.copt = None
        self.beta = beta

    def to(self, device):
        super().to(device=device)
        if self.device is not None:
            self.cmodel.to(self.device)

    def train(
        self,
        training_set,
        valid_set=None,
        test_set=None,
        Optim=None,
        COptim=None,
        **kwargs):
        if COptim is None:
            COptim = Optim
        if COptim is not None and list(self.cmodel.parameters()):
            self.copt = COptim(self.cmodel.parameters())
        else:
            self.copt = None
        return super().train(
            training_set,
            valid_set=valid_set,
            test_set=test_set,
            **kwargs)

    def train_epoch(self, dataset, callback=None):
        self.model.train()
        self.model.reset()
        self.cmodel.train()
        self.cmodel.reset()
        for data in dataset:
            if self.opt is not None:
                self.opt.zero_grad()
            if self.copt is not None:
                self.copt.zero_grad()
            mloss, metric, output = self.model(data)
            closs, cmetric, coutput = self.cmodel(data)
            loss = mloss + self.beta * closs
            if self.opt is not None or self.copt is not None and loss:
                loss.backward()
            if self.opt is not None:
                self.opt.step()
            if self.copt is not None:
                # NOTE: Based on the face the gradient of lambda in dual
                #       is reversing signs of their gradient in primal,
                #       we avoid a repeated backward (and also pytorch's
                #       retain_graph problem), we simply reverse the sign
                #       for the dual. Don't zero_grad() here!
                reverse_sign_grad(self.cmodel.parameters())
                self.copt.step()
            yield loss, metric, output
        if callable(callback):
            callback()
