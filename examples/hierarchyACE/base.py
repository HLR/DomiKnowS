from regr.graph import Graph, Concept, Relation, Property
import torch
import torch.nn
import torch.optim as optim
from regr.sensor.pytorch.learners import TorchLearner, FullyConnectedLearner
from regr.sensor.pytorch.sensors import TorchSensor, ReaderSensor
from regr.utils import WrapperMetaClass
import os
from typing import Union, List
import numpy
import itertools
from tqdm import tqdm
from data.reader import SubSimpleReader
from Graphs.solver.solver import ACELogicalSolver
from regr.solver.ilpOntSolverFactory import ilpOntSolverFactory
import json


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
        for _it in range(len(info)):
            item = info[_it]
            _weight = []
            for data in truth[_it]:
                if item in _list:
                    val = defaults[_list.index(item)]
                else:
                    val = 0.95
                if data:
                    _weight.append(val)
                else:
                    _weight.append(1 - val)
            weights.append(_weight)
        return weights

    def predConstraint(self, paths):
        reader_sensors = self.readers
        _array = []
        for _iter in range(len(paths)):
            loader = SubSimpleReader(paths[_iter])
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

                        for prop1 in self.poi:
                            Do = True
                            entity = prop1.sup.name
                            prop_name = prop1.name
                            dict_key = prop1.name.name
                            list(prop1.find(ReaderSensor))[0][1](context=context)
                            list(prop1.find(TorchLearner))[0][1](context=context)
                            if Do:
                                info.append(prop1.name.name)
                                truth.append(context[list(prop1.find(ReaderSensor))[0][1].fullname])
                                pred.append(context[list(prop1.find(TorchLearner))[0][1].fullname])
                                total += len(truth[-1])

                        result = self.solver.inferILPConstrains(context=context, info=info)

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
                            pred[_it] = result[0][info[_it]]

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

    def structured_train_constraint(self, iterations, paths, ratio):
        reader_sensors = self.readers
        _array = []
        for _iter in range(len(paths)):
            loader = SubSimpleReader(paths[_iter])
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
                        for prop1 in self.poi:
                            Do = True
                            entity = prop1.sup.name
                            prop_name = prop1.name
                            dict_key = prop1.name.name
                            list(prop1.find(ReaderSensor))[0][1](context=context)
                            list(prop1.find(TorchLearner))[0][1](context=context)
                            # check this with quan
                            if Do:
                                info.append(prop1.name.name)
                                truth.append(context[list(prop1.find(ReaderSensor))[0][1].fullname])
                                pred.append(context[list(prop1.find(TorchLearner))[0][1].fullname])
                                total += len(truth[-1])
                        if self.solver:
                            result = self.solver.inferILPConstrains(context=context, info=info)

                            inferences = [torch.zeros(1).float().to(self.device) for i in range(len(info))]
                            for _it in range(len(info)):
                                inferences[_it] = result[0][info[_it]]

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
                                pred[_it] = result[0][info[_it]]
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

