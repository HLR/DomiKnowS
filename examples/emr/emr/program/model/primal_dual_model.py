import logging
from collections import defaultdict
from itertools import combinations

import torch
from torch.nn.parameter import Parameter

from regr.graph import Concept, Property
from regr.solver.constructor.constructor import ProbConstructor
from regr.program.model.torch import BaseModel, TorchModel
from regr.sensor.torch.sensor import DataSensor

from ...solver.primal_dual_session import PrimalDualSession


class PrimalDualModel(TorchModel):
    logger = logging.getLogger(__name__)

    def __init__(self, graph, constructor=None, SessionType=None):
        BaseModel.__init__(self, graph, loss=None, metric=None)
        self.poi = {prop: (output_sensor, target_sensor) for prop, output_sensor, target_sensor in self.find_poi()}

        def find_concept(node):
            if isinstance(node, Concept):
                return node
        def find_relation(node):
            if isinstance(node, Concept):
                for rel_type, rels in node._out.items():
                    yield from rels

        self.names = {concept.name: concept for concept in graph.traversal_apply(find_concept)}
        self.constructor = constructor or ProbConstructor()
        self.SessionType = SessionType or PrimalDualSession

        self.lmbd_idx = {}
        param_idx = 0
        for rel in graph.traversal_apply(find_relation):
            self.lmbd_idx[rel] = param_idx
            param_idx += 1
        # for concept in graph.traversal_apply(find_concept):
        #     self.lmbd_idx[concept] = param_idx
        #     param_idx += 1

        self.lmbd = Parameter(torch.Tensor(param_idx))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.lmbd, 0.5)

    def forward(self, data_item):
        closs = self.inferSelection(data_item, list(self.poi))
        return closs, data_item

    def get_raw_input(self, data_item):
        _, sentence_sensor = self.graph.get_sensors(DataSensor, lambda s: not s.target)[0]
        sentences = sentence_sensor(data_item)
        mask_len = [len(s) for s in sentences]  # (b, )
        return sentences, mask_len

    def get_prop_result(self, data_item, prop):
        output_sensor, _ = self.graph.poi[prop]

        logit = output_sensor(data_item)
        #score = -F.logsigmoid(logit)
        score = torch.sigmoid(logit)
        mask = output_sensor.mask(data_item)
        return score, mask

    def calculateILPSelection(self, data, *predicates_list):
        concepts_list = []
        for predicates in predicates_list:
            concept_dict = {}
            for predicate, v in predicates.items():
                concept = self.names[predicate]
                concept_dict[concept] = v
            concepts_list.append(concept_dict)

        # call to solve
        closs = self.solve_legacy(data, *concepts_list)

        return closs

    def get_lmbd(self, key):
        return self.lmbd[self.lmbd_idx[key]]

    def solve_legacy(self, data, *predicates_list):
        # data is a list of objects of the base type
        # predicates_list is a list of predicates
        # predicates_list[i] is the dict of predicates of the objects of the base type to the power of i
        # predicates_list[i][concept] is the prediction result of the predicate for concept
        self.logger.debug('Start for data %s', data)
        candidates = self.constructor.candidates(data, *predicates_list)

        session = self.SessionType()
        variables, predictions, variables_not, predictions_not = self.constructor.variables(session, candidates, * predicates_list)
        # use predictions instead of variables
        constraints, constraints_not = self.constructor.constraints(session, candidates, predictions, predictions_not, *predicates_list)

        closs = 0
        for key, panelty in constraints.items():
            rel, *_ = key
            closs += self.get_lmbd(rel) * panelty
        # for key, panelty in constraints_not.items():
        #     concept, *_ = key
        #     closs += self.get_lmbd(concept) * panelty

        return closs

    def inferSelection(self, data_item, prop_list):
        # build concept (property) group by rank (# of has-a)
        prop_dict = defaultdict(list)

        def concept_rank(concept):
            # get inheritance rank
            in_rank = 0
            for is_a in concept.is_a():
                in_rank += concept_rank(is_a.dst)
            # get self rank
            rank = len(concept.has_a())
            # TODO: it would be better to have new syntax to support override
            # determine override condition
            if in_rank > rank:
                rank = in_rank
            return rank or 1

        for prop in prop_list:
            if isinstance(prop.prop_name, Concept):
                concept = prop.prop_name
            else:
                concept = prop.sup
            prop_dict[concept_rank(concept)].append(prop)

        if prop_dict:
            max_rank = max(prop_dict.keys())
        else:
            max_rank = 0

        sentences, mask_len = self.get_raw_input(data_item)
        batch_size = len(sentences)

        values = [defaultdict(dict) for _ in range(batch_size)]
        for rank, props in prop_dict.items():
            for prop in props:
                #name = '{}_{}'.format(prop.sup.name, prop.name)
                #name = prop.name  # current implementation has concept name in prop name
                if isinstance(prop.prop_name, Concept):
                    concept = prop.prop_name
                else:
                    concept = prop.sup
                name = concept.name # we need concept name to match that in OWL
                # score - (b, l...*r) / (b, l...*r, c)
                # mask - (b, l...*r)
                score, mask = self.get_prop_result(data_item, prop)
                mask = mask.cpu().detach().to(torch.bool).numpy()
                # copy and detach, time consuming I/O
                batched_value = score#.clone().cpu().detach().numpy()

                for batch_index in range(batch_size):
                    # (l...*r)
                    value = batched_value[batch_index]
                    # apply mask
                    # (l'...*r)
                    value = value[tuple(slice(0, mask[batch_index].sum(r).max()) for r in range(rank))]
                    values[batch_index][rank][name] = value
        #import pdb; pdb.set_trace()

        results = []
        # inference per instance
        for batch_index in range(batch_size):
            # prepare tokens
            tokens = ['{}_{}'.format(i, token)
                      for i, token in enumerate(sentences[batch_index][:mask_len[batch_index]])]
            # prepare tables
            table_list = []
            for rank in range(1, max_rank + 1):
                if rank in values[batch_index]:
                    table_list.append(values[batch_index][rank])
                else:
                    table_list.append(None)

            #import pdb; pdb.set_trace()
            # Do inference
            closs = self.calculateILPSelection(tokens, *table_list)

            results.append(closs)
        closs = torch.stack(results).mean()
        return closs
