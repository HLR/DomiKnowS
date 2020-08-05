from regr.graph import Graph, Concept, Relation, Property
import torch
import torch.optim as optim
from Graphs.Learners.mainLearners import CallingLearner
# from ..Graphs.Learners.mainLearners import CallingLearner
from Graphs.Sensors.mainSensors import CallingSensor, ReaderSensor
from Graphs.Sensors.conceptSensors import LabelSensor
from regr.utils import WrapperMetaClass

# from ..Graphs.Sensors.mainSensors import CallingSensor, ReaderSensor

from typing import Union, List
import numpy
import torch
import itertools
from tqdm import tqdm


def sequence_cross_entropy_with_logits(logits: torch.Tensor,
                                       targets: torch.Tensor,
                                       weights: torch.Tensor,
                                        average: str = "batch",
                                       label_smoothing: float = None,
                                       gamma: float = None,
                                       eps: float = 1e-8,
                                       alpha: Union[float, List[float], torch.FloatTensor] = None
                                      ) -> torch.FloatTensor:
    """
    Computes the cross entropy loss of a sequence, weighted with respect to
    some user provided weights. Note that the weighting here is not the same as
    in the :func:`torch.nn.CrossEntropyLoss()` criterion, which is weighting
    classes; here we are weighting the loss contribution from particular elements
    in the sequence. This allows loss computations for models which use padding.

    Parameters
    ----------
    logits : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    targets : ``torch.LongTensor``, required.
        A ``torch.LongTensor`` of size (batch, sequence_length) which contains the
        index of the true class for each corresponding step.
    weights : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch, sequence_length)
    average: str, optional (default = "batch")
        If "batch", average the loss across the batches. If "token", average
        the loss across each item in the input. If ``None``, return a vector
        of losses per batch element.
    label_smoothing : ``float``, optional (default = None)
        Whether or not to apply label smoothing to the cross-entropy loss.
        For example, with a label smoothing value of 0.2, a 4 class classification
        target would look like ``[0.05, 0.05, 0.85, 0.05]`` if the 3rd class was
        the correct label.
    gamma : ``float``, optional (default = None)
        Focal loss[*] focusing parameter ``gamma`` to reduces the relative loss for
        well-classified examples and put more focus on hard. The greater value
        ``gamma`` is, the more focus on hard examples.
    alpha : ``float`` or ``List[float]``, optional (default = None)
        Focal loss[*] weighting factor ``alpha`` to balance between classes. Can be
        used independently with ``gamma``. If a single ``float`` is provided, it
        is assumed binary case using ``alpha`` and ``1 - alpha`` for positive and
        negative respectively. If a list of ``float`` is provided, with the same
        length as the number of classes, the weights will match the classes.
        [*] T. Lin, P. Goyal, R. Girshick, K. He and P. Dollár, "Focal Loss for
        Dense Object Detection," 2017 IEEE International Conference on Computer
        Vision (ICCV), Venice, 2017, pp. 2999-3007.

    Returns
    -------
    A torch.FloatTensor representing the cross entropy loss.
    If ``average=="batch"`` or ``average=="token"``, the returned loss is a scalar.
    If ``average is None``, the returned loss is a vector of shape (batch_size,).

    """
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of "
                         "None, 'token', or 'batch'")

    # make sure weights are float
    weights = weights.float()
    # sum all dim except batch
    non_batch_dims = tuple(range(1, len(weights.shape)))
    # shape : (batch_size,)
    weights_batch_sum = weights.sum(dim=non_batch_dims)
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()
    # focal loss coefficient
    if gamma:
        # shape : (batch * sequence_length, num_classes)
        probs_flat = log_probs_flat.exp()
        eps = torch.tensor(eps, device=probs_flat.device)
        probs_flat = probs_flat.min(1 - eps)
        probs_flat = probs_flat.max(eps)
        # shape : (batch * sequence_length,)
        probs_flat = torch.gather(probs_flat, dim=1, index=targets_flat)
        # shape : (batch * sequence_length,)
        focal_factor = (1. - probs_flat) ** gamma
        # shape : (batch, sequence_length)
        focal_factor = focal_factor.view(*targets.size())
        weights = weights * focal_factor

    if alpha is not None:
        # shape : () / (num_classes,)
        if isinstance(alpha, (float, int)):
            # pylint: disable=not-callable
            # shape : (2,)
            alpha_factor = torch.tensor([1. - float(alpha), float(alpha)],
                                        dtype=weights.dtype, device=weights.device)
            # pylint: enable=not-callable
        elif isinstance(alpha, (list, numpy.ndarray, torch.Tensor)):
            # pylint: disable=not-callable
            # shape : (c,)
            alpha_factor = torch.tensor(alpha, dtype=weights.dtype, device=weights.device)
            # pylint: enable=not-callable
            if not alpha_factor.size():
                # shape : (1,)
                alpha_factor = alpha_factor.view(1)
                # shape : (2,)
                alpha_factor = torch.cat([1 - alpha_factor, alpha_factor])
        else:
            raise TypeError(('alpha must be float, list of float, or torch.FloatTensor, '
                             '{} provided.').format(type(alpha)))
        # shape : (batch, max_len)
        alpha_factor = torch.gather(alpha_factor, dim=0, index=targets_flat.view(-1)).view(*targets.size())
        weights = weights * alpha_factor

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = - log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
        num_non_empty_sequences = ((weights_batch_sum > 0).float().sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights_batch_sum.sum() + 1e-13)
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
        return per_batch_loss


class NewGraph(Graph, metaclass=WrapperMetaClass):

    __metaclass__ = WrapperMetaClass

    def __init__(
            self,
            *args,
            **kwargs
    ):
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def get_multiassign(self):
        def func(node):
            if isinstance(node, Property) and len(node) > 1:
                return node
            return None
        return list(self.traversal_apply(func))

    @property
    def poi(self):
        return self.get_multiassign()

    def get_sensors(self, *tests):
        def func(node):
            if isinstance(node, Property):
                yield from node.find(*tests)
        return list(self.traversal_apply(func))

    @property
    def reader(self):
        sentence_sensors = self.get_sensors(ReaderSensor)
        readers = {sensor for name, sensor in sentence_sensors}
        assert len(readers) == 1 # consider only 1 reader now
        return readers.pop()


class PytorchSolverGraph(NewGraph, metaclass=WrapperMetaClass):

    __metaclass__ = WrapperMetaClass

    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.filename = "saves"

    @property
    def parameters(self):
        _list = []
        for prop1 in self.poi:
            _list.extend(list(list(prop1.find(CallingLearner))[0][1].parameters))
        return set(_list)

    @property
    def optimizer(self):
        optimizer = optim.SGD(self.parameters, lr=0.1)
        return optimizer

    def weights(self, info, truth):
        weights = []
        for item, value in info.items():
            weights.append([1,1])
        return torch.tensor(weights, device=self.device)

    def train(self, iterations):
        train_data = self.reader.reader.readTrain()
        _array = itertools.tee(train_data, iterations)

        for i in tqdm(range(iterations), "Iterations: "):
            self.reader.data = _array[i]
#             print("here")
            while True:
                try:
                    truth = []
                    pred = []
                    info = {}
                    data_item = {}
                    for prop1 in self.poi:
                        entity = prop1.sup.name
                        prop_name = prop1.name
                        if entity not in info:
                            info[entity] = {}
                        if prop_name not in info[entity]:
                            info[entity][prop_name] = {"start" : len(truth)}
                        list(prop1.find(LabelSensor))[0][1](data_item)
                        list(prop1.find(CallingLearner))[0][1](data_item)
                        # check this with quan
                        truth.append(data_item[list(prop1.find(LabelSensor))[0][1].fullname])
                        pred.append(data_item[list(prop1.find(CallingLearner))[0][1].fullname])

                    total_loss = 0
                    weights = self.weights(info=info, truth=truth).float()
                    loss_fn = []
                    for _it in range(len(truth)):
                        loss_fn.append(torch.nn.CrossEntropyLoss(weight=weights[_it]))
                        truth[_it] = truth[_it].long()
                        pred[_it] =+ pred[_it].float()
                        total_loss += loss_fn[_it](pred[_it], truth[_it])
#                     print(total_loss)
                    total_loss.backward()

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                except StopIteration:
                    break
        self.save()

    def save(self, ):
        learners = self.get_sensors(CallingLearner)
        _learners = [learner for name, learner in learners]
        for item in _learners:
            item.save(self.filename)

    def test(self, mode):
        if mode == "valid":
            valid_data = self.reader.reader.readValid()
            self.reader.data = valid_data
        elif mode == "test":
            test_data = self.reader.reader.readTest()
            self.reader.data = test_data
        else:
            print("you have to specify one of valid or test to run this function")
            raise
        metrics = {'f1_score': 0, 'precision': 0, 'recall': 0}
        while True:
            try:
                truth = []
                pred = []
                data_item = {}
                for prop1 in self.poi:
                    list(prop1.find(LabelSensor))[0][1](data_item)
                    list(prop1.find(CallingLearner))[0][1](data_item)
                    truth.append(data_item[list(prop1.find(LabelSensor))[0][1].fullname])
                    pred.append(data_item[list(prop1.find(CallingLearner))[0][1].fullname])
            except StopIteration:
                break



class ACEGraph(PytorchSolverGraph, metaclass=WrapperMetaClass):
    __metaclass__ = WrapperMetaClass

    def __init__(
            self,
            *args,
            **kwargs
    ):
        pass

    def set_reader_instance(self, reader):
        self.reader_instance = reader

    @property
    def target_weights(self):
        class_weights = self.reader_instance.weights
        targets = {'FAC': 1, 'PER': 1, 'ORG': 1, '-O-': 1, 'WEA' : 1, "VEH": 1, 'LOC': 1, 'GPE': 1 }
        for label, weight in targets.items():
            targets[label] = class_weights[self.reader_instance.lableToInt(label)]
        return targets

    def weights(self, info, truth):
        # weights = []
        # for _it in range(len(truth)):
        #     weights.append(torch.ones(1, truth[_it].shape[0]))
        # target_weights = self.target_weights()
        # for entity, _dict in info.items():
        #     for prop1, _values in _dict.items():
        #         for i in range(len(truth[_values['start']])):
        #             if truth[_values['start']][i] > 0.5:
        #                 weights[_values['start']][0][i] = target_weights[entity]
        #             else:
        #                 weights[_values['start']][0][i] = target_weights['-O-']
        # return torch.stack(weights)
        weights = []
        for item, value in info.items():
            weight = self.target_weights[item]
            weights.append([weight, 1 - weight])
        return torch.tensor(weights, device=self.device)





