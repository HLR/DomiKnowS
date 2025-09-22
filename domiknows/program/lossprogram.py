import logging
import torch
import numpy as np
from tqdm import tqdm

from domiknows.program.model.pytorch import SolverModel
#from domiknows.graph.LeftLogic import LeftLogicElementOutput

from .program import LearningBasedProgram, get_len
from ..utils import consume, entuple, detuple
from .model.lossModel import PrimalDualModel, SampleLossModel, InferenceModel
from .model.base import Mode

from .model.gbi import GBIModel

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
        batch_size = 1,
        dataset_size = None, # provide dataset_size if dataset does not have len implemented
        print_loss = True, # print loss on each grad update
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
            batch_size=batch_size,
            dataset_size=dataset_size,
            print_loss=print_loss,
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
                    
            if self.cmodel.loss is not None and  repr(self.cmodel.loss) == "'None'":
                losSTr = str(self.cmodel.loss)
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
        batch_size = 1,
        dataset_size = None, # provide dataset_size if dataset does not have len implemented
        print_loss = True, # print loss on each grad update
        **kwargs):

        if batch_size < 1:
            raise ValueError(f'batch_size must be at least 1, but got batch_size={batch_size}')

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

        # try to get the number of iterations
        num_data_iters = dataset_size
        if num_data_iters is None:
            if not hasattr(dataset, '__len__'):
                raise ValueError(f'dataset must have attribute __len__ if dataset_size is not provided')

            num_data_iters = len(dataset)

        batch_loss = 0.0 # track accumulated loss across batch for logging
        for data_idx, data in enumerate(dataset):
            # forward pass
            mloss, metric, *output = self.model(data)  # output = (datanode, builder)
            if iter < c_warmup_iters:
                loss = mloss
            else:
                closs, *_ = self.cmodel(output[1])
                if torch.is_nonzero(closs):
                    loss = mloss + self.beta * closs
                    # self.logger.info('closs is not zero')
                else:
                    loss = mloss

            if not loss:
                continue

            # get hypothetical position in the batch
            batch_pos = data_idx % batch_size

            # calculate gradients for item
            loss /= batch_size
            batch_loss += loss.item()
            loss.backward()

            # only update if we're the last item in the batch
            # or if we're the last item in the dataset
            do_update = (
                (batch_pos == (batch_size - 1)) or
                (data_idx == (num_data_iters - 1))
            )

            # log loss on each update
            if do_update:
                # self.logger.info(f'loss (i={data_idx}) = {batch_loss:.3f}')
                pass

            # do backwards pass update
            if self.opt is not None and do_update:
                self.opt.step()
            iter += 1

            if (
                self.copt is not None and
                iter > c_warmup_iters and
                do_update and
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

            # only zero out grads & loss tracker if we've done an update
            if do_update:
                batch_loss = 0.0
                if self.opt is not None:
                    self.opt.zero_grad()
                if self.copt is not None:
                    self.copt.zero_grad()

        c_session['iter'] = iter
        c_session['c_update_iter'] = c_update_iter
        c_session['c_update_freq'] = c_update_freq
        c_session['c_update'] = c_update

    def test_epoch(self, dataset, **kwargs):
        # just to consum kwargs
#         self.cmodel.mode(Mode.TEST)
        yield from super().test_epoch(dataset)
        
    def populate_epoch(self, dataset, grad = False):
        self.model.mode(Mode.POPULATE)
#         self.cmodel.mode(Mode.POPULATE)
        self.model.reset()
        if not grad:
            with torch.no_grad():
                for i, data_item in enumerate(dataset):
                    _, _, *output = self.model(data_item)
                    yield detuple(*output[:1])
        else:
            for i, data_item in enumerate(dataset):
                for dataKey in data_item:
                    if data_item[dataKey].dtype in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
                        data_item[dataKey].requires_grad= True
                    
                _, _, *output = self.model(data_item)
                yield detuple(*output[:1])
                
class PrimalDualProgram(LossProgram):
    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, beta=1, **kwargs):
        super().__init__(graph, Model, CModel=PrimalDualModel, beta=beta, **kwargs)

class InferenceProgram(LossProgram):
    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, beta=1, **kwargs):
        super().__init__(graph, Model, CModel=InferenceModel, beta=beta, **kwargs)

    def evaluate_condition(self, evaluate_data, device="cpu"):
        acc = 0
        total = 0

        for datanode in tqdm(self.populate(evaluate_data, device=device), total=len(evaluate_data), desc="Evaluating"):

            # Finding the label of constraints/condition
            find_constraints_label = datanode.myBuilder.findDataNodesInBuilder(select=datanode.graph.constraint)
            if len(find_constraints_label) < 1:
                self.logger.error("No Constraint Labels found")
                continue
            find_constraints_label = find_constraints_label[0]
            constraint_labels_dict = find_constraints_label.getAttributes()
            active_lc_name = set(x.split('/')[0] for x in constraint_labels_dict)

            # Follow code for set active/non-active before querying
            for lc_name, lc in self.graph.logicalConstrains.items():
                assert lc_name == str(lc)

                if lc_name in active_lc_name:
                    lc.active = True
                else:
                    lc.active = False

            total += 1
            # Inference the final output
            verify_constrains = datanode.verifyResultsLC()
            if not verify_constrains:
                continue
            # Getting label of constraints and convert to 0-1
            verify_constrains = {k: v for k, v in verify_constrains.items() if k in active_lc_name}

            condition_list = [1 if verify_constrains[lc]["satisfied"] == 100.0 else 0 for lc in verify_constrains]
            constraint_labels = [int(constraint_labels_dict[lc + "/label"].item()) for lc in active_lc_name]
            acc += int(constraint_labels == condition_list)

        if total == 0:
            self.logger.error("No Valid Constraint found for this dataset.")
            return None

        return acc / total


class SampleLossProgram(LossProgram):
    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, beta=1, **kwargs):
        super().__init__(graph, Model, CModel=SampleLossModel, beta=beta, **kwargs)


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
        
class GBIProgram(LossProgram):
    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, poi, beta=1, **kwargs):
        super().__init__(graph, Model, CModel=GBIModel, beta=beta, poi=poi, **kwargs)
        from domiknows.utils import setDnSkeletonMode
        setDnSkeletonMode(True)