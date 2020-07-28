import abc
from typing import Dict
from collections import defaultdict
import numpy as np
import torch
from ..graph import Graph
from ..sensor.allennlp.sensor import SentenceEmbedderSensor, SentenceSensor
from ..sensor.allennlp.base import AllenNlpLearner
from ..utils import get_prop_result
from .ilpOntSolver import ilpOntSolver


DataInstance = Dict[str, torch.Tensor]


class AllennlplogInferenceSolver(ilpOntSolver):
    __metaclass__ = abc.ABCMeta

    def inferSelection(self, graph: Graph, data: DataInstance) -> DataInstance:
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

        for prop in graph.poi:
            concept = prop.sup
            prop_dict[concept_rank(concept)].append(prop)

        if prop_dict:
            max_rank = max(prop_dict.keys())
        else:
            max_rank = 0

        # find base, assume only one base for now
        # FIXME: that [0] means we are only considering problems with only one sentence
        sentence_sensor = graph.get_sensors(SentenceSensor)[0]
        # Note: SentenceEmbedderSensor is not reliable. For example BERT indexer will introduce redundant wordpiece tokens

        sentence = data[sentence_sensor.fullname]
        batch_size = len(sentence)
        mask_len = [len(s) for s in sentence]  # (b, )
        length = max(mask_len)
        device = None

        values = [defaultdict(dict) for _ in range(batch_size)]
        for rank, props in prop_dict.items():
            for prop in props:
                #name = '{}_{}'.format(prop.sup.name, prop.name)
                #name = prop.name  # current implementation has concept name in prop name
                name = prop.sup.name # we need concept name to match that in OWL
                # TODO: what if a concept has several properties to be predict?
                # pred_logit - (b, l...*r, c) - for dim-c, [0] is neg, [1] is pos
                # mask - (b, l...*r)
                label, pred_logit, mask = get_prop_result(prop, data)
                if not device:
                    device = pred_logit.device
                # pred - (b, l...*r, c)
                pred = torch.nn.functional.log_softmax(pred_logit, dim=-1)
                # copy and detach, time consuming I/O
                # batched_value - (b, l...*r, c)
                batched_value = pred.clone().cpu().detach().numpy()

                for batch_index in range(batch_size):
                    # (l...*r, c)
                    value = batched_value[batch_index]
                    # apply mask
                    # (l'...*r, c)
                    value = value[(slice(0, mask_len[batch_index]),) * rank]
                    values[batch_index][rank][name] = value
        #import pdb; pdb.set_trace()

        results = []
        # inference per instance
        for batch_index in range(batch_size):
            # prepare tokens
            tokens = ['{}_{}'.format(i, token)
                      for i, token in enumerate(sentence[batch_index])]
            # prepare tables
            table_list = []
            for rank in range(1, max_rank + 1):
                if rank in values[batch_index]:
                    table_list.append(values[batch_index][rank])
                else:
                    table_list.append(None)
            #import pdb; pdb.set_trace()
            # Do inference
            try:
                # following statement should be equivalent to
                # - EMR:
                # result_list = self.calculateILPSelection(tokens, concept_dict, relation_dict)
                # - SPRL:
                # result_list = self.calculateILPSelection(tokens, concept_dict, None, triplet_dict)
                result_table_list = self.calculateILPSelection(tokens, *table_list)

                #import pdb; pdb.set_trace()
                if all([result_table is None for result_table in result_table_list]):
                    raise RuntimeError('No result from solver. Check any log from the solver.')
            except:
                # whatever, raise it
                raise

            # collect result in batch
            result = defaultdict(dict)
            for rank, props in prop_dict.items():
                for prop in props:
                    name = prop.sup.name
                    # (l'...*r)
                    # return structure is a pd?
                    # result_table_list started from 0
                    result[rank][name] = result_table_list[rank - 1][name]
            results.append(result)
        #import pdb; pdb.set_trace()

        # put results back
        for rank, props in prop_dict.items():
            for prop in props:
                name = prop.sup.name
                instance_value_list = []
                for batch_index in range(batch_size):
                    # (l'...*r)
                    instance_value = results[batch_index][rank][name]
                    # (l...*r)
                    instance_value_pad = np.empty([length, ] * rank)
                    instance_value_pad[(slice(0, mask_len[batch_index]),) * rank] = instance_value
                    # (l...*r)
                    instance_value_d = torch.tensor(instance_value_pad, device=device)
                    instance_value_list.append(instance_value_d)
                # (b, l...*r)
                batched_value = torch.stack(instance_value_list, dim=0)
                # (b, l...*r, 2)
                batched_value = torch.stack([1 - batched_value, batched_value], dim=-1)
                # undo softmax
                logits_value = torch.log(batched_value / (1 - batched_value))  # Go to +- inf
                # Put it back finally
                #import pdb; pdb.set_trace()
                data[prop.fullname] = logits_value
                for learner in prop.find(AllenNlpLearner):
                    data[learner.fullname] = logits_value

        return data
