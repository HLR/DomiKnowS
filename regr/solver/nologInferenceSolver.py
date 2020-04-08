import abc
from typing import Dict
from collections import defaultdict
import numpy as np
import torch
from ..graph import Concept
from .ilpOntSolver import ilpOntSolver


DataInstance = Dict[str, torch.Tensor]


class NoLogInferenceSolver(ilpOntSolver):
    __metaclass__ = abc.ABCMeta

    def get_raw_input(self, data):
        raise NotImplementedError

    def get_prop_result(self, prop, data):
        raise NotImplementedError

    def set_prop_result(self, prop, data, value):
        raise NotImplementedError

    def inferSelection(self, data: DataInstance, prop_list) -> DataInstance:
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

        sentences, mask_len = self.get_raw_input(data)
        batch_size = len(sentences)
        length = max(mask_len)

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
                score, mask = self.get_prop_result(prop, data)
                # copy and detach, time consuming I/O
                batched_value = score.clone().cpu().detach().numpy()

                for batch_index in range(batch_size):
                    # (l...*r)
                    value = batched_value[batch_index]
                    # apply mask
                    # (l'...*r)
                    value = value[(slice(0, mask_len[batch_index]),) * rank]
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
            result_table_list = self.calculateILPSelection(tokens, *table_list)
            #import pdb; pdb.set_trace()
            if all([result_table is None for result_table in result_table_list]):
                raise RuntimeError('No result from solver. Check any log from the solver.')

            # collect result in batch
            result = defaultdict(dict)
            for rank, props in prop_dict.items():
                for prop in props:
                    if isinstance(prop.prop_name, Concept):
                        concept = prop.prop_name
                    else:
                        concept = prop.sup
                    name = concept.name
                    # (l'...*r)
                    # return structure is a pd?
                    # result_table_list started from 0
                    result[rank][name] = result_table_list[rank - 1][name]
            results.append(result)
        #import pdb; pdb.set_trace()

        # put results back
        for rank, props in prop_dict.items():
            for prop in props:
                score, _ = self.get_prop_result(prop, data)  # for device
                if isinstance(prop.prop_name, Concept):
                    concept = prop.prop_name
                else:
                    concept = prop.sup
                name = concept.name
                instance_value_list = []
                for batch_index in range(batch_size):
                    # (l'...*r)
                    instance_value = results[batch_index][rank][name]
                    # (l...*r)
                    instance_value_pad = np.zeros([length, ] * rank)
                    instance_value_pad[(slice(0, mask_len[batch_index]),) * rank] = instance_value
                    # (l...*r)
                    instance_value_d = torch.tensor(instance_value_pad, device=score.device)
                    instance_value_list.append(instance_value_d)
                # (b, l...*r)
                batched_value = torch.stack(instance_value_list, dim=0)
                # Put it back finally
                #import pdb; pdb.set_trace()
                self.set_prop_result(prop, data, batched_value)

        return data
