from regr.graph import Graph, Concept, Relation, Property
import torch
import torch.nn
import torch.optim as optim
from regr.sensor.pytorch.learners import TorchLearner, FullyConnectedLearner
# from ..Graphs.Learners.mainLearners import CallingLearner
from regr.sensor.pytorch.sensors import TorchSensor, ReaderSensor
from regr.utils import WrapperMetaClass

# from ..Graphs.Sensors.mainSensors import CallingSensor, ReaderSensor
import os
from typing import Union, List
import numpy
import torch
import itertools
from tqdm import tqdm
from data.reader import SimpleReader
from Graphs.graph import pair, ART, word, phrase, PART_WHOLE, ART, ORG_AFF, PER_SOC, METONYMY, PHYS, GEN_AFF
from Graphs.solver.solver import ACELogicalSolver
from regr.solver.ilpOntSolverFactory import ilpOntSolverFactory
import json


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


def structured_perceptron_exact_with_logits(logits: torch.FloatTensor,
                                            targets: torch.LongTensor,
                                            weights: torch.FloatTensor,
                                            inferences: torch.FloatTensor = None,
                                            average: str = "batch",
                                            label_smoothing: float = None,
                                            gamma: float = None,
                                            eps: float = 1e-8,
                                            alpha: Union[float, List[float], torch.FloatTensor] = None,
                                            soft_penalty=0.6
                                           ) -> torch.FloatTensor:
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

    # inference coefficient
    if inferences is not None:
        if len(inferences.shape) == len(logits.shape): # inferences shape: (batch, sequence_length, ..., num_classes)
            # shape: (batch, sequence_length, ...)
            inferences_weights = inferences.select(-1, 1)
            #inferences_weights = torch.gather(inferences, dim=-1, index=targets)
        else: # inferences shape: (batch, sequence_length, ...)
            # shape: (batch, sequence_length, ...)
            inferences_weights = inferences

        #inferences_weights = (inferences_weights.long() != targets.long()).float()
          # 0 - structured loss only; 1 - conventional loss
        inferences_true = (inferences_weights.long() == targets.long()).float()
        inferences_weights = 1 - inferences_true * (1 - soft_penalty)

        weights = weights * inferences_weights

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
    def readers(self):
        sentence_sensors = self.get_sensors(ReaderSensor)
        readers = [sensor for name, sensor in sentence_sensors]
        return readers


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
        learners = self.get_sensors(TorchLearner)
        for _, learner in learners:
            _list.extend(learner.parameters)
        return set(_list)

    @property
    def optimizer(self):
        optimizer = optim.SGD(self.parameters, lr=0.04)
        return optimizer

    def weights(self, info, truth):
        weights = []
        for item in info:
            weights.append([0.07, 0.93])
        return torch.tensor(weights, device=self.device)

    def load(self):
        learners = self.get_sensors(TorchLearner)
        _learners = [learner for name, learner in learners]
        for item in _learners:
            item.load(self.filename)

    def save(self, ):
        learners = self.get_sensors(TorchLearner)
        _learners = [learner for name, learner in learners]
        for item in _learners:
            item.save(self.filename)

    def test(self, paths):
        reader_sensors = self.readers
        _array = []
        for _iter in tqdm(range(len(paths)), "READERS: "):
            loader = SimpleReader(paths[_iter])
            reader = loader.data()
            _array.append(reader)
        predict = 0
        total = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        metrics = {}
        for prop1 in self.poi:
            metrics[prop1.sup.name + "-" + prop1.name.name] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        for _iter in tqdm(range(len(paths)), "Readers Execution: "):
            while True:
                try:
                    value = next(_array[_iter])
                    for item in reader_sensors:
                        item.fill_data(value)
                    truth = []
                    pred = []
                    context = {}
                    # learners = self.get_sensors(FullyConnectedLearner)
                    # _learners = [learner for name, learner in learners]
                    for prop1 in self.poi:
                        Do = True
                        dict_key = prop1.sup.name + "-" + prop1.name.name
                        list(prop1.find(ReaderSensor))[0][1](context=context)
                        list(prop1.find(TorchLearner))[0][1](context=context)
                        if prop1.sup == pair:
                            list(phrase['ground_bound'].find(ReaderSensor))[0][1](context=context)
                            phrases_gr = context[phrase['ground_bound'].fullname]
                            phrases = context[phrase['raw'].fullname]
                            matches = []
                            for _ph in phrases:
                                check = False
                                for _tr in phrases_gr:
                                    if _ph[0] == _tr[0] and _ph[1] == _tr[1]:
                                        matches.append(_tr)
                                        check = True
                                        break
                                    elif _ph[0] == _tr[0] and _tr[1] - 1 <= _ph[1] <= _tr[1] + 1:
                                        matches.append(_tr)
                                        check = True
                                        break
                                    elif _ph[1] == _tr[1] and _tr[0] - 1 <= _ph[0] <= _tr[0] + 1:
                                        matches.append(_tr)
                                        check = True
                                        break
                                if not check:
                                    matches.append("NONE")
                            pairs = context[pair['index'].fullname]
                            pairs_gr = context[list(prop1.find(ReaderSensor))[0][1].fullname]
                            _truth = []
                            for _iteration in range(len(pairs)):
                                check = False
                                for item in pairs_gr:
                                    if matches[pairs[_iteration][0]] == item[0] and matches[pairs[_iteration][1]] == \
                                            item[1]:
                                        _truth.append(1)
                                        check = True
                                        break
                                if not check:
                                    _truth.append(0)
                            _truth = torch.tensor(_truth, device=self.device)
                            context[list(prop1.find(ReaderSensor))[0][1].fullname] = _truth
                            if not len(pairs):
                                Do = False

                        if Do:
                            truth.append(context[list(prop1.find(ReaderSensor))[0][1].fullname])
                            pred.append(context[list(prop1.find(TorchLearner))[0][1].fullname])
                            total += len(truth[-1])
                            for item in range(len(pred[-1])):
                                _, index = torch.max(pred[-1][item], dim=0)
                                if index == truth[-1][item] and index == 1:
                                    metrics[dict_key]["tp"] += 1
                                elif index == truth[-1][item] and index == 0:
                                    metrics[dict_key]["tn"] += 1
                                elif index != truth[-1][item] and index == 1:
                                    metrics[dict_key]["fp"] += 1
                                elif index != truth[-1][item] and index == 0:
                                    metrics[dict_key]["fn"] += 1

                except StopIteration:
                    break
        # recall = tp / (tp + fn)
        # precision = tp / (tp + fp)
        # f1 = 2 * (precision * recall) / (precision + recall)
        # print("precision is " + str(precision))
        # print("recall is " + str(recall))
        # print("f1 score is " + str(f1))
        print("check the result file\n")
        filename = 'test_results.txt'
        data_json = json.dumps(metrics)
        if os.path.exists(filename):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not

        loggerFile = open(filename, append_write)
        loggerFile.write(data_json)
        loggerFile.close()


class ACEGraph(PytorchSolverGraph, metaclass=WrapperMetaClass):
    __metaclass__ = WrapperMetaClass

    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.solver = ilpOntSolverFactory.getOntSolverInstance(self, ACELogicalSolver)
        # self.solver = None

    # def set_reader_instance(self, reader):
    #     self.reader_instance = reader

    def weights(self, info, truth):
        weights = []
        defaults = [pow(0.96312867737412, 4), pow(0.9894575918349645, 4), pow(0.8696203721105056, 4),
                    pow(0.9933500065054284, 4), pow(0.9916224538475995, 4), pow(0.9943908750524049, 4),
                    pow(0.9453833142989316, 4)]
        _list = ["ORG", "FAC", "PER", "VEH", "LOC", "WEA", "GPE"]
        _pairs = ["ART", "PER-SOC", "ORG-AFF", "METONYMY", "GEN-AFF", "PART-WHOLE", "PHYS"]
        for _it in range(len(info)):
            item = info[_it]
            if item not in _pairs:
                _weight = []
                for data in truth[_it]:
                    if data:
                        _weight.append(defaults[_list.index(item)])
                    else:
                        _weight.append(1 - defaults[_list.index(item)])
                weights.append(_weight)
            else:
                _weight = []
                for data in truth[_it]:
                    if data:
                        _weight.append(0.90)
                    else:
                        _weight.append(0.10)
                weights.append(_weight)
        return weights

    def predConstraint(self, paths):
        reader_sensors = self.readers
        _array = []
        for _iter in range(len(paths)):
            #loader = SimpleReader(paths[_iter])
            loader = SimpleReader(paths[_iter])
            reader = loader.data()
            _array.append(itertools.tee(reader, 1))

        for i in tqdm(range(1), "Iterations: "):
            predict = 0
            total = 0
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            metrics = {}
            inference_metrics = {}
            extra = {}
            for prop1 in self.poi:
                metrics[prop1.name.name] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
                inference_metrics[prop1.name.name] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
                extra[prop1.name.name] = 0

            for j in tqdm(range(len(paths)), "READER : "):
                while True:
                    try:
                        value = next(_array[j][i])
                        for item in reader_sensors:
                            item.fill_data(value)
                        truth = []
                        pred = []
                        info = []
                        context = {}
                        # print(list(phrase['tag_encode'].find(TorchSensor))[0][1](context=context))
                        # print("end")
                        for prop1 in self.poi:
                            Do = True
                            entity = prop1.sup.name
                            prop_name = prop1.name
                            dict_key = prop1.name.name
                            list(prop1.find(ReaderSensor))[0][1](context=context)
                            list(prop1.find(TorchLearner))[0][1](context=context)
                            if prop1.sup == pair:
                                list(phrase['ground_bound'].find(ReaderSensor))[0][1](context=context)
                                phrases_gr = context[phrase['ground_bound'].fullname]
                                phrases = context[phrase['raw'].fullname]
                                matches = []
                                for _ph in phrases:
                                    check = False
                                    for _tr in phrases_gr:
                                        if _ph[0] == _tr[0] and _ph[1] == _tr[1]:
                                            matches.append(_tr)
                                            check = True
                                            break
                                        elif _ph[0] == _tr[0] and _tr[1] - 1 <= _ph[1] <= _tr[1] + 1:
                                            matches.append(_tr)
                                            check = True
                                            break
                                        elif _ph[1] == _tr[1] and _tr[0] - 1 <= _ph[0] <= _tr[0] + 1:
                                            matches.append(_tr)
                                            check = True
                                            break
                                    if not check:
                                        matches.append("NONE")
                                pairs = context[pair['index'].fullname]
                                pairs_gr = context[list(prop1.find(ReaderSensor))[0][1].fullname]
                                _truth = []
                                extra[prop1.name.name] += len(pairs_gr)
                                for _iteration in range(len(pairs)):
                                    check = False
                                    for item in pairs_gr:
                                        if matches[pairs[_iteration][0]] == item[0] and matches[pairs[_iteration][1]] == \
                                                item[1] or (matches[pairs[_iteration][0]] == item[1] and matches[pairs[_iteration][1]] == item[0]):
                                            _truth.append(1)
                                            check = True
                                            break
                                    if not check:
                                        _truth.append(0)
                                _truth = torch.tensor(_truth, device=self.device)
                                context[list(prop1.find(ReaderSensor))[0][1].fullname] = _truth
                                if not len(pairs):
                                    Do = False

                            # check this with quan
                            if Do:
                                info.append(prop1.name.name)
                                truth.append(context[list(prop1.find(ReaderSensor))[0][1].fullname])
                                pred.append(context[list(prop1.find(TorchLearner))[0][1].fullname])
                                total += len(truth[-1])

                        result = self.solver.inferILPConstrains(context=context, info=info)
                        entities = ["FAC", "VEH", "PER", "ORG", "GPE", "LOC", "WEA"]
                        relations = ["ART", "GEN-AFF", "ORG-AFF", "PER-SOC", "METONYMY", "PART-WHOLE", "PHYS"]


                        for _val in range(len(pred)):
                            for item in range(len(pred[_val])):
                                _, index = torch.max(pred[_val][item], dim=0)
                                if index == truth[_val][item] and index == 1:
                                    metrics[info[_val]]["tp"] += 1
                                elif index == truth[_val][item] and index == 0:
                                    metrics[info[_val]]["tn"] += 1
                                elif index != truth[_val][item] and index == 1:
                                    metrics[info[_val]]["fp"] += 1
                                elif index != truth[_val][item] and index == 0:
                                    metrics[info[_val]]["fn"] += 1


                        for _it in range(len(info)):
                            if info[_it] in entities:
                                pred[_it] = result[0][info[_it]]
                            elif info[_it] in relations:
                                pred[_it] = result[1][info[_it]]
                        for _val in range(len(pred)):
                            for item in range(len(pred[_val])):
                                index = pred[_val][item].item()
                                if index == truth[_val][item] and index == 1:
                                    inference_metrics[info[_val]]["tp"] += 1
                                elif index == truth[_val][item] and index == 0:
                                    inference_metrics[info[_val]]["tn"] += 1
                                elif index != truth[_val][item] and index == 1:
                                    inference_metrics[info[_val]]["fp"] += 1
                                elif index != truth[_val][item] and index == 0:
                                    inference_metrics[info[_val]]["fn"] += 1

                    #                         import pickle
                    #                         file = open('imp.pkl', 'wb')

                    #                         # dump information to that file
                    #                         pickle.dump(result, file)

                    #                         # close the file
                    #                         file.close()
                    #                         print(fg)
                    #                         total_loss = 0
                    #                         weights = self.weights(info=info, truth=truth)
                    #                         loss_fn = []
                    #                         # for _it in range(len(truth)):
                    #                         #     loss_fn.append(torch.nn.CrossEntropyLoss(weight=weights[_it]))
                    #                         #     truth[_it] = truth[_it].long()
                    #                         #     pred[_it] = pred[_it].float()
                    #                         #     total_loss += loss_fn[_it](pred[_it], truth[_it])
                    #                         for _it in range(len(truth)):
                    #                             truth[_it] = truth[_it].long()
                    #                             pred[_it] = pred[_it].float()
                    #                             total_loss += sequence_cross_entropy_with_logits(pred[_it], truth[_it],
                    #                                                                              weights=torch.tensor(weights[_it], device=self.device).float(), gamma=2)
                    #                         # print(total_loss)
                    #                         total_loss.backward(retain_graph=True)

                    #                         self.optimizer.step()
                    #                         self.optimizer.zero_grad()
                    except StopIteration:
                        break
            # recall = tp / (tp + fn)
            # precision = tp / (tp + fp)
            # f1 = 2 * (precision * recall) / (precision + recall)
            # print("precision is " + str(precision) + " recall is " + str(recall) + " f1 score is " + str(f1))
            print("check the test_result file\n")
            filename = 'test_results.txt'
            data_json = json.dumps(metrics)
            data1_json = json.dumps(inference_metrics)
            extra_json = json.dumps(extra)
            if os.path.exists(filename):
                append_write = 'a'  # append if already exists
            else:
                append_write = 'w'  # make a new file if not

            loggerFile = open(filename, append_write)
            loggerFile.write("iteration is: " + str(i) + '\n')
            loggerFile.write(data_json)
            print("\n")
            loggerFile.write("extra is: " + '\n')
            loggerFile.write(extra_json)
            print("\n")
            loggerFile.write("inference result is: " + '\n')
            loggerFile.write(data1_json)
            print("\n")
            loggerFile.write("End of Iteration\n")
            loggerFile.close()

    def PredictionTime(self, sentence):
        value = {'raw': sentence}
        for item in self.readers:
            item.fill_data(value)
        truth = []
        pred = []
        info = []
        context = {}
        # print(list(phrase['tag_encode'].find(TorchSensor))[0][1](context=context))
        # print("end")
        for prop1 in self.poi:
            Do = True
            entity = prop1.sup.name
            prop_name = prop1.name
            dict_key = prop1.name.name
            list(prop1.find(TorchLearner))[0][1](context=context)

            info.append(prop1.name.name)
            pred.append(context[list(prop1.find(TorchLearner))[0][1].fullname])

        result = self.solver.inferILPConstrains(context=context, info=info)
        entities = ["FAC", "VEH", "PER", "ORG", "GPE", "LOC", "WEA"]
        relations = ["ART", "GEN-AFF", "ORG-AFF", "PER-SOC", "METONYMY", "PART-WHOLE", "PHYS"]

        print("ORDER")
        print(info)
        print("BEFORE RESULTS")
        print(pred)

        for _it in range(len(info)):
            if info[_it] in entities:
                pred[_it] = result[0][info[_it]]
            elif info[_it] in relations:
                pred[_it] = result[1][info[_it]]
        print("AFTER RESULT")
        print(pred)

    def structured_train_constraint(self, iterations, paths, ratio):
        reader_sensors = self.readers
        _array = []
        for _iter in range(len(paths)):
            #loader = SimpleReader(paths[_iter])
            loader = SimpleReader(paths[_iter])
            reader = loader.data()
            _array.append(itertools.tee(reader, iterations))

        for i in tqdm(range(iterations), "Iterations: "):
            predict = 0
            total = 0
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            metrics = {}
            inference_metrics = {}
            extra = {}
            for prop1 in self.poi:
                metrics[prop1.name.name] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
                inference_metrics[prop1.name.name] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
                extra[prop1.name.name] = 0

            for j in tqdm(range(len(paths)), "READER : "):
                while True:
                    try:
                        value = next(_array[j][i])
                        for item in reader_sensors:
                            item.fill_data(value)
                        truth = []
                        pred = []
                        info = []
                        context = {}
                        # print(list(phrase['tag_encode'].find(TorchSensor))[0][1](context=context))
                        # print("end")
                        for prop1 in self.poi:
                            Do = True
                            entity = prop1.sup.name
                            prop_name = prop1.name
                            dict_key = prop1.name.name
                            list(prop1.find(ReaderSensor))[0][1](context=context)
                            list(prop1.find(TorchLearner))[0][1](context=context)
                            if prop1.sup == pair:
                                list(phrase['ground_bound'].find(ReaderSensor))[0][1](context=context)
                                phrases_gr = context[phrase['ground_bound'].fullname]
                                phrases = context[phrase['raw'].fullname]
                                matches = []
                                for _ph in phrases:
                                    check = False
                                    for _tr in phrases_gr:
                                        if _ph[0] == _tr[0] and _ph[1] == _tr[1]:
                                            matches.append(_tr)
                                            check = True
                                            break
                                        elif _ph[0] == _tr[0] and _tr[1] - 1 <= _ph[1] <= _tr[1] + 1:
                                            matches.append(_tr)
                                            check = True
                                            break
                                        elif _ph[1] == _tr[1] and _tr[0] - 1 <= _ph[0] <= _tr[0] + 1:
                                            matches.append(_tr)
                                            check = True
                                            break
                                    if not check:
                                        matches.append("NONE")
                                pairs = context[pair['index'].fullname]
                                pairs_gr = context[list(prop1.find(ReaderSensor))[0][1].fullname]
                                extra[prop1.name.name] += len(pairs_gr)
                                _truth = []
                                for _iteration in range(len(pairs)):
                                    check = False
                                    for item in pairs_gr:
                                        if matches[pairs[_iteration][0]] == item[0] and matches[pairs[_iteration][1]] == \
                                                item[1] or (matches[pairs[_iteration][0]] == item[1] and matches[pairs[_iteration][1]] == item[0]):
                                            _truth.append(1)
                                            check = True
                                            break
                                    if not check:
                                        _truth.append(0)
                                _truth = torch.tensor(_truth, device=self.device)
                                context[list(prop1.find(ReaderSensor))[0][1].fullname] = _truth
                                if not len(pairs):
                                    Do = False

                            # check this with quan
                            if Do:
                                info.append(prop1.name.name)
                                truth.append(context[list(prop1.find(ReaderSensor))[0][1].fullname])
                                pred.append(context[list(prop1.find(TorchLearner))[0][1].fullname])
                                total += len(truth[-1])
                        if self.solver:
                            result = self.solver.inferILPConstrains(context=context, info=info)
                            entities = ["FAC", "VEH", "PER", "ORG", "GPE", "LOC", "WEA"]
                            relations = ["ART", "GEN-AFF", "ORG-AFF", "PER-SOC", "METONYMY", "PART-WHOLE", "PHYS"]

                            inferences = [torch.zeros(1).float().to(self.device) for i in range(len(info))]
                            for _it in range(len(info)):
                                if info[_it] in entities:
                                    inferences[_it] = result[0][info[_it]]
                                elif info[_it] in relations:
                                    inferences[_it] = result[1][info[_it]]

                        total_loss = 0
                        weights = self.weights(info=info, truth=truth)
                        loss_fn = []
                        # for _it in range(len(truth)):
                        #     loss_fn.append(torch.nn.CrossEntropyLoss(weight=weights[_it]))
                        #     truth[_it] = truth[_it].long()
                        #     pred[_it] = pred[_it].float()
                        #     total_loss += loss_fn[_it](pred[_it], truth[_it])
                        for _it in range(len(truth)):
                            truth[_it] = truth[_it].long()
                            pred[_it] = pred[_it].float()
                            total_loss += structured_perceptron_exact_with_logits(logits=pred[_it], targets=truth[_it], inferences=inferences[_it],
                                                                             weights=torch.tensor(weights[_it], device=self.device).float(), gamma=2, soft_penalty=ratio, label_smoothing=0.01) #Just structure loss
                        # print(total_loss)
                        total_loss.backward(retain_graph=True)

                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        # for _it in range(len(info)):
                        #     if info[_it] in entities:
                        #         pred[_it] = result[0][info[_it]]
                        #     elif info[_it] in relations:
                        #         pred[_it] = result[1][info[_it]]
                        for _val in range(len(pred)):
                            for item in range(len(pred[_val])):
                                _, index = torch.max(pred[_val][item], dim=0)
                                if index == truth[_val][item] and index == 1:
                                    metrics[info[_val]]["tp"] += 1
                                elif index == truth[_val][item] and index == 0:
                                    metrics[info[_val]]["tn"] += 1
                                elif index != truth[_val][item] and index == 1:
                                    metrics[info[_val]]["fp"] += 1
                                elif index != truth[_val][item] and index == 0:
                                    metrics[info[_val]]["fn"] += 1


                        # With INFERENCE
                        if self.solver:
                            for _it in range(len(info)):
                                if info[_it] in entities:
                                    pred[_it] = result[0][info[_it]]
                                elif info[_it] in relations:
                                    pred[_it] = result[1][info[_it]]
                            for _val in range(len(pred)):
                                for item in range(len(pred[_val])):
                                    index = pred[_val][item].item()
                                    if index == truth[_val][item] and index == 1:
                                        inference_metrics[info[_val]]["tp"] += 1
                                    elif index == truth[_val][item] and index == 0:
                                        inference_metrics[info[_val]]["tn"] += 1
                                    elif index != truth[_val][item] and index == 1:
                                        inference_metrics[info[_val]]["fp"] += 1
                                    elif index != truth[_val][item] and index == 0:
                                        inference_metrics[info[_val]]["fn"] += 1
                    except StopIteration:
                        break
            # recall = tp / (tp + fn)
            # precision = tp / (tp + fp)
            # f1 = 2 * (precision * recall) / (precision + recall)
            # print("precision is " + str(precision) + " recall is " + str(recall) + " f1 score is " + str(f1))
            print("check the result file\n")
            filename = 'results.txt'
            data_json = json.dumps(metrics)
            data1_json = json.dumps(inference_metrics)
            extra_json = json.dumps(extra)
            if os.path.exists(filename):
                append_write = 'a'  # append if already exists
            else:
                append_write = 'w'  # make a new file if not

            loggerFile = open(filename, append_write)
            loggerFile.write("iteration is: " + str(i) + '\n')
            loggerFile.write(data_json)
            print("\n")
            loggerFile.write("extra is: " + '\n')
            loggerFile.write(extra_json)
            print("\n")
            if self.solver:
                loggerFile.write("inference result is: " + '\n')
                loggerFile.write(data1_json)
                print("\n")
            loggerFile.write("End of Iteration\n")
            loggerFile.close()
            self.save()

