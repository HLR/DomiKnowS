import logging
from collections import defaultdict
import torch

from regr.graph import Concept, Relation
from regr.sensor.torch.sensor import ReaderSensor
from regr.solver.constructor.constructor import BatchMaskProbConstructor
from regr.solver.session.solver_session import SolverSession


from .primal_dual_model import PrimalDualModel


# NOTE: BatchPrimalDualModel has a problem that its output is not consistent.
#       Use BigBatchPrimalDualModel instead, which is consistent and more efficent.
class BatchPrimalDualModel(PrimalDualModel):
    logger = logging.getLogger(__name__)
    def __init__(self, graph, constructor=None, SessionType=None):
        constructor = constructor or BatchMaskProbConstructor
        super().__init__(graph, constructor=constructor(), SessionType=SessionType)

    def forward(self, data_item):
        closs = self.inferSelection(data_item, list(self.poi))
        return closs, data_item

    def solve_legacy(self, data, *predicates_list):
        # data is a list of objects of the base type
        # predicates_list is a list of predicates
        # predicates_list[i] is the dict of predicates of the objects of the base type to the power of i
        # predicates_list[i][concept] is the prediction result of the predicate for concept
        self.logger.debug('Start for data %s', data)
        candidates = self.constructor.candidates(data, *predicates_list)

        session = self.SessionType()
        _, predictions, _, predictions_not = self.constructor.variables(session, candidates, * predicates_list)
        masks = {}
        for k in predictions:
            pred, mask = predictions[k]
            predictions[k] = pred
            masks[k] = mask
        masks_not = {}
        for k in predictions_not:
            pred, mask = predictions_not[k]
            predictions_not[k] = pred
            masks_not[k] = mask
        # use predictions instead of variables
        constraints, constraints_not = self.constructor.constraints(session, candidates, predictions, predictions_not, *predicates_list)

        closs = 0.
        for key, panelty in constraints.items():
            rel, *args = key
            args = tuple(args)
            if len(args) == 1:
                mask = masks[rel.src, args[0]]
            elif len(args) == 2:
                mask_src = masks[rel.src, args[0]]
                mask_dst = masks[rel.dst, args[1]] # NOTE: need this level of tuple to match the key
                mask = mask_src * mask_dst
            closs += self.get_lmbd(rel) * mask * panelty
        # for key, panelty in constraints_not.items():
        #     concept, *_ = key
        #     closs += self.get_lmbd(concept) * panelty

        return closs.mean()

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
        tokens = list(zip(sentences, mask_len))

        table_list = []
        for rank in range(1, max_rank + 1):
            props = prop_dict.get(rank)
            if not props:
                table_list.append(None)
                continue
            prop_values = {}
            for prop in props:
                if isinstance(prop.prop_name, Concept):
                    concept = prop.prop_name
                else:
                    concept = prop.sup
                name = concept.name # we need concept name to match that in OWL
                # score - (b, l...*r) / (b, l...*r, c)
                # mask - (b, l...*r)
                score, mask = self.get_prop_result(data_item, prop)
                prop_values[name] = (score, mask)
            table_list.append(prop_values)

        #import pdb; pdb.set_trace()

        closs = self.calculateILPSelection(tokens, *table_list)

        return closs.mean()

class BigBatchPrimalDualModel(BatchPrimalDualModel):
    logger = logging.getLogger(__name__)

    def constraints(self, session, *predicates_list):
        constraints = {} # (rel, (object,...)) -> constr

        all_predicates = {}
        for predicates in predicates_list:
            all_predicates.update(predicates)

        # add constraints
        self.logger.debug('add constraints')
        for predicates in predicates_list:
            for concept in predicates:
                self.logger.debug('for %s', concept.name)
                self.logger.debug(' - is_a')
                for rel in concept.is_a():
                    self.logger.debug(' - - %s', rel.name)
                    if rel.src not in all_predicates or rel.dst not in all_predicates:
                        continue 
                    # A is_a B : A(x) <= B(x)
                    val_a, mask_a = all_predicates[rel.src]
                    val_b, mask_b = all_predicates[rel.dst]
                    constr = session.constr(
                            val_a, SolverSession.CTYPE.LE, val_b,
                            name=rel.name)
                    self.logger.debug(' - - add %s', constr)
                    constraints[rel] = (constr, mask_a * mask_b)
                self.logger.debug(' - not_a')
                for rel in concept.not_a():
                    self.logger.debug(' - - %s', rel.name)
                    if rel.src not in all_predicates or rel.dst not in all_predicates:
                        continue
                    # A not_a B : A(x) + B(x) <= 1
                    val_a, mask_a = all_predicates[rel.src]
                    val_b, mask_b = all_predicates[rel.dst]
                    constr = session.constr(
                        val_a + val_b, SolverSession.CTYPE.LE, 1,
                        name=rel.name)
                    self.logger.debug(' - - add %s', constr)
                    constraints[rel] = (constr, mask_a * mask_b)
                self.logger.debug(' - has_a')
                for arg_id, rel in enumerate(concept.has_a()): # TODO: need to include indirect ones like sp_tr is a tr while tr has a lm
                    self.logger.debug(' - - %s', rel.name)
                    if rel.src not in all_predicates or rel.dst not in all_predicates:
                        continue
                    # A has_a B : A(x,y,...) <= B(x)
                    # (B, l, l)
                    val_a, mask_a = all_predicates[rel.src]
                    # (B, l,)
                    val_b, mask_b = all_predicates[rel.dst]
                    # val_b ->
                    # arg_id 0: (B, l, 1) / arg_id 1: (B, 1, l)
                    # arg_id 0: (B, l, 1, 1) / arg_id 1: (B, 1, l, 1) / ...
                    batch_size, *length = val_b.shape
                    shape = [batch_size,]
                    for _ in range(arg_id):
                        shape.append(1)
                    shape.extend(length)
                    for _ in range(arg_id + 1 + len(length), len(val_a.shape)):
                        shape.append(1)
                    val_b = val_b.view(shape)
                    mask_b = mask_b.view(shape)
                    constr = session.constr(
                        val_a, SolverSession.CTYPE.LE, val_b,
                        name=rel.name)
                    self.logger.debug(' - - add %s', constr)
                    constraints[rel] = (constr, mask_a * mask_b)
        return constraints

    def solve_legacy(self, data, *predicates_list):
        # data is a list of objects of the base type
        # predicates_list is a list of predicates
        # predicates_list[i] is the dict of predicates of the objects of the base type to the power of i
        # predicates_list[i][concept] is the prediction result of the predicate for concept
        self.logger.debug('Start for data %s', data)

        session = self.SessionType()
        constraints = self.constraints(session, *predicates_list)

        panelties = []
        for rel, (panelty, mask) in constraints.items():
            panelties.append(self.get_lmbd(rel) * (mask * panelty).view(mask.shape[0], -1))
        closs = torch.cat(panelties, dim=1).sum(dim=1)

        return closs

class ReaderBigBatchPrimalDualModel(BigBatchPrimalDualModel):
    def get_raw_input(self, data_item):
        sentence_sensor = self.graph.get_sensors(ReaderSensor, lambda s: not s.target)[0]
        masks, sentences = sentence_sensor(data_item)
        mask_len = [len(s) for s in sentences]  # (b, )
        return sentences, mask_len
