import logging
import torch
import numpy as np
from tqdm import tqdm

from .program import LearningBasedProgram, get_len
from ..utils import consume, entuple, detuple
from .model.lossModel import PrimalDualModel, SampleLosslModel
from .model.base import Mode

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
    DEFAULTCMODEL = PrimalDualModel

    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, CModel=None, beta=1, **kwargs):
        super().__init__(graph, Model, **kwargs)
        if CModel is None:
            CModel = self.DEFAULTCMODEL
            
        from inspect import signature
        cmodelSignature = signature(CModel.__init__)
        
        cmodelKwargs = {}
        for param in cmodelSignature.parameters.values():
            paramName = param.name
            if paramName in kwargs:
                cmodelKwargs[paramName] = kwargs[paramName]
        self.cmodel = CModel(graph, **cmodelKwargs)
        self.copt = None
        self.beta = beta

    def to(self, device):
        super().to(device=device)
        if self.device is not None:
            self.model.to(self.device)
            self.cmodel.device = self.device
            self.cmodel.to(self.device)

    def train(
        self,
        training_set,
        valid_set=None,
        test_set=None,
        # COptim=None,  # SGD only
        c_lr=0.05,
        c_momentum=0.9,
        c_warmup_iters=10,  # warmup
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
        if list(self.cmodel.parameters()):
            self.copt = torch.optim.Adam(self.cmodel.parameters(), lr=c_lr)
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
    
    def call_epoch(self, name, dataset, epoch_fn, **kwargs):
        if dataset is not None:
            self.logger.info(f'{name}:')
            desc = name if self.epoch is None else f'Epoch {self.epoch} {name}'

            consume(tqdm(epoch_fn(dataset, **kwargs), total=get_len(dataset), desc=desc))

            if self.model.loss:
                self.logger.info(' - loss:')
                self.logger.info(self.model.loss)

                metricName = 'loss'
                metricResult = self.model.loss
                    
                if self.dbUpdate is not None:
                    self.dbUpdate(desc, metricName, metricResult)
                    
            if self.cmodel.loss and self.cmodel.loss is not None:
                desc = name if self.epoch is None else f'Epoch {self.epoch} {name}'
                self.logger.info(' - Constraint loss:')
                self.logger.info(self.cmodel.loss)

                metricName = 'Constraint_loss'
                metricResult = self.cmodel.loss

                if self.dbUpdate is not None:
                    self.dbUpdate(desc, metricName, metricResult)

            ilpMetric = None
            softmaxMetric = None

            if self.model.metric:
                self.logger.info(' - metric:')
                for key, metric in self.model.metric.items():
                    self.logger.info(f' - - {key}')
                    self.logger.info(metric)
                    try:
                        self.f.write(f' - - {name}')
                        self.f.write(f' - - {key}')
                        self.f.write("\n")
                        self.f.write(str(metric))
                        self.f.write("\n")
                    except:
                        pass
                    metricName = key
                    metricResult = metric
                    if self.dbUpdate is not None:
                        self.dbUpdate(desc, metricName, metricResult)

                    if key == 'ILP':
                        ilpMetric = metric

                    if key == 'softmax':
                        softmaxMetric = metric
#         super().call_epoch(name=name, dataset=dataset, epoch_fn=epoch_fn, **kwargs)
        

    

    def train_epoch(
        self, dataset,
        c_lr=1,
        c_warmup_iters=10,  # warmup
        c_freq_increase=1,  # d
        c_freq_increase_freq=1,
        c_lr_decay=0,  # strategy
        c_lr_decay_param=1,
        c_session={},
        **kwargs):
        assert c_session
        from torch import autograd
        torch.autograd.set_detect_anomaly(False)
        self.model.mode(Mode.TRAIN)
#         self.cmodel.mode(Mode.TRAIN)
        iter = c_session['iter']
        c_update_iter = c_session['c_update_iter']
        c_update_freq = c_session['c_update_freq']
        c_update = c_session['c_update']
        self.model.train()
        self.model.reset()
        self.cmodel.train()
        self.cmodel.reset()
        for data in dataset:
            if self.opt is not None:
                self.opt.zero_grad()
            if self.copt is not None:
                self.copt.zero_grad()
            mloss, metric, *output = self.model(data)  # output = (datanode, builder)
            if iter < c_warmup_iters:
                loss = mloss
            else:
                closs, *_ = self.cmodel(output[1])
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
#         self.cmodel.mode(Mode.TEST)
        yield from super().test_epoch(dataset)
        
    def populate_epoch(self, dataset):
        self.model.mode(Mode.POPULATE)
#         self.cmodel.mode(Mode.POPULATE)
        self.model.reset()
        with torch.no_grad():
            for i, data_item in enumerate(dataset):
                _, _, *output = self.model(data_item)
                yield detuple(*output[:1])
        
class PrimalDualProgram(LossProgram):
    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, beta=1, **kwargs):
        super().__init__(graph, Model, CModel=PrimalDualModel, beta=beta, **kwargs)
        
class SampleLossProgram(LossProgram):
    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, beta=1, **kwargs):
        super().__init__(graph, Model, CModel=SampleLosslModel, beta=beta, **kwargs)


    def train(
        self,
        training_set,
        valid_set=None,
        test_set=None,
        # COptim=None,  # SGD only
        c_lr=0.05,
        c_momentum=0.9,
        c_warmup_iters=10,  # warmup
        c_freq=10,
        c_freq_increase=5,  # d
        c_freq_increase_freq=1,
        c_lr_decay=4,  # strategy
        c_lr_decay_param=1,  # param in the strategy
        **kwargs):
        
        return super().train(
            training_set=training_set,
            valid_set=valid_set,
            test_set=test_set,
            c_lr=c_lr,
            c_momentum=c_momentum,
            c_warmup_iters=c_warmup_iters,  # warmup
            c_freq=c_freq,
            c_freq_increase=c_freq_increase,  # d
            c_freq_increase_freq=c_freq_increase_freq,
            c_lr_decay=c_lr_decay,  # strategy
            c_lr_decay_param=c_lr_decay_param,  # param in the strategy
            **kwargs)


    def train_epoch(
        self, dataset,
        c_warmup_iters=0,  # warmup
        c_session={},
        **kwargs):
        self.model.mode(Mode.TRAIN)
#         self.cmodel.mode(Mode.TRAIN)
        assert c_session
        iter = c_session['iter']
        self.model.train()
        self.model.reset()
        self.cmodel.train()
        self.cmodel.reset()
        for data in dataset:
            if self.opt is not None:
                self.opt.zero_grad()
            if self.copt is not None:
                self.copt.zero_grad()
            mloss, metric, *output = self.model(data)  # output = (datanode, builder)
            if iter < c_warmup_iters:
                loss = mloss
            else:
                closs, *_ = self.cmodel(output[1])
                if torch.is_tensor(closs):
                    loss = mloss + self.beta * closs
                else:
                    loss = mloss
                    
                if loss != loss:
                    raise Exception("Calculated loss is nan")
                
            if self.opt is not None and loss:
                loss.backward()
                # for name, param in self.model.named_parameters():
                #     if param.requires_grad:
                #         print (name, param.grad)

                self.opt.step()
                iter += 1
            
            if (
                self.copt is not None and
                loss
            ):
                self.copt.step()
            
            yield (loss, metric, *output[:1])

        c_session['iter'] = iter