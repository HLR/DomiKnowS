import logging
from itertools import product, permutations

from domiknows.utils import isbad

from ..session.solver_session import SolverSession


class Constructor():
    logger = logging.getLogger(__name__)

    def __init__(self, lazy_not=True, self_relation=True):
        self.lazy_not = lazy_not
        self.self_relation = self_relation

    def get_predication(self, predicate, idx, negative=False):
        raise NotImplementedError

    def isskip(self, value):
        return isbad(value)

    def candidates(self, data, *predicates_list):
        candidates = {} # concept -> [(object,...), ...]
        if self.self_relation:
            gen = lambda enum_data, arity: product(enum_data, repeat=arity)
        else:
            gen = lambda enum_data, arity: permutations(enum_data, r=arity)
        for arity, predicates in enumerate(predicates_list, 1):
            for concept in predicates:
                #assert concept not in candidates
                # last one change first (c-order)
                # abc rep=3 -> aaa, aab, aac, aba, abb, abc, ...
                candidates[concept] = tuple(gen(enumerate(data), arity))
        return candidates

    def variables(self, session, candidates, *predicates_list):
        variables = {} # (concept, (object,...)) -> variable
        predictions = {} # (concept, (object,...)) -> prediction
        variables_not = {} # (concept, (x,...)) -> variable
        predictions_not = {} # (concept, (x,...)) -> prediction

        # add variables
        self.logger.debug('add variables')
        for predicates in predicates_list:
            for concept, predicate in predicates.items():
                self.logger.debug('for %s', concept.name)
                self.logger.debug(predicate)
                for x in candidates[concept]: # flat: C-order -> last dim first!
                    idx, _ = zip(*x)
                    prediction = self.get_predication(predicate, idx)
                    if self.isskip(prediction): continue
                    var = session.var(
                        session.VTYPE.BIN, 0, 1,
                        name='{}_{}'.format(concept.name, str(x)))
                    self.logger.debug(' - add %s', var)
                    variables[concept, x] = var
                    predictions[concept, x] = prediction

        if self.lazy_not:
            self.logger.debug('lazy negative')

            # add variables
            self.logger.debug('lazy negative add variables')
            for predicates in predicates_list:
                for concept, predicate in predicates.items():
                    self.logger.debug('for %s', concept.name)
                    for x in candidates[concept]:
                        idx, _ = zip(*x)
                        prediction_not = self.get_predication(predicate, idx, negative=True)
                        if self.isskip(prediction_not): continue
                        var = session.var(
                            session.VTYPE.BIN, 0, 1,
                            name='lazy_not_{}_{}'.format(concept.name, str(x)))
                        self.logger.debug(' - add %s', var)
                        variables_not[concept, x] = var
                        predictions_not[concept, x] = prediction_not

        return variables, predictions, variables_not, predictions_not

    def constraints(self, session, candidates, variables, variables_not, *predicates_list):
        constraints = {} # (rel, (object,...)) -> constr
        constraints_not = {} # (rel, (x,...)) -> constr

        # add constraints
        self.logger.debug('add constraints')
        for predicates in predicates_list:
            for concept in predicates:
                self.logger.debug('for %s', concept.name)
                self.logger.debug(' - is_a')
                for rel in concept.is_a():
                    self.logger.debug(' - - %s', rel.name)
                    # A is_a B : A(x) <= B(x)
                    for x in candidates[rel.src]:
                        if (rel.src, x) not in variables: continue
                        if (rel.dst, x) not in variables: continue
                        constr = session.constr(
                            variables[rel.src, x], SolverSession.CTYPE.LE, variables[rel.dst, x],
                            name='{}_{}'.format(rel.name, str(x)))
                        self.logger.debug(' - - add %s', constr)
                        assert (rel, x) not in constraints
                        constraints[rel, x] = constr
                self.logger.debug(' - not_a')
                for rel in concept.not_a():
                    self.logger.debug(' - - %s', rel.name)
                    # A not_a B : A(x) + B(x) <= 1
                    for x in candidates[rel.src]:
                        if (rel.src, x) not in variables: continue
                        if (rel.dst, x) not in variables: continue
                        constr = session.constr(
                            variables[rel.src, x] + variables[rel.dst, x], SolverSession.CTYPE.LE, 1,
                            name='{}_{}'.format(rel.name, str(x)))
                        self.logger.debug(' - - add %s', constr)
                        assert (rel, x) not in constraints
                        constraints[rel, x] = constr
                self.logger.debug(' - has_a')
                for arg_id, rel in enumerate(concept.has_a()): # TODO: need to include indirect ones like sp_tr is a tr while tr has a lm
                    self.logger.debug(' - - %s', rel.name)
                    # A has_a B : A(x,y,...) <= B(x)
                    for xy in candidates[rel.src]:
                        x = xy[arg_id]
                        if (rel.src, xy) not in variables: continue
                        if (rel.dst, (x,)) not in variables: continue
                        constr = session.constr(
                            variables[rel.src, xy], SolverSession.CTYPE.LE, variables[rel.dst, (x,)],
                            name='{}_{}_{}'.format(rel.name, str(xy), str(x)))
                        self.logger.debug(' - - add %s', constr)
                        assert (rel, xy, (x,)) not in constraints
                        constraints[rel, xy, (x,)] = constr

        if self.lazy_not:
            self.logger.debug('lazy negative add constraints')
            for predicates in predicates_list:
                for concept in predicates:
                    self.logger.debug('for %s', concept.name)
                    for x in candidates[concept]:
                        if (concept, x) not in variables: continue
                        if (concept, x) not in variables_not: continue
                        constr = session.constr(
                            variables[concept, x] + variables_not[concept, x], SolverSession.CTYPE.EQ, 1,
                            name='lazy_not_{}_{}'.format(concept.name, str(x)))
                        self.logger.debug(' - add %s', constr)
                        constraints_not[concept, x] = constr
    
        return constraints, constraints_not

    def objective(self, candidates, variables, predictions, variables_not, predictions_not, *predicates_list):
        self.logger.debug('set objective')
        objective = None
        for predicates in predicates_list:
            for concept in predicates:
                for x in candidates[concept]:
                    if (concept, x) not in variables: continue
                    objective += variables[concept, x] * predictions[concept, x]
        if self.lazy_not:
            for predicates in predicates_list:
                for concept in predicates:
                    for x in candidates[concept]:
                        if (concept, x) not in variables_not: continue
                        objective += variables_not[concept, x] * predictions_not[concept, x]
        return objective


class ScoreConstructor(Constructor):
    def get_predication(self, predicate, idx, negative=False):
        if negative:
            return predicate[(*idx, 0)]
        return predicate[(*idx, 1)]


class ProbConstructor(Constructor):
    def __init__(self, lazy_not=False, self_relation=True):
        super().__init__(lazy_not=lazy_not, self_relation=self_relation)

    def get_predication(self, predicate, idx, negative=False):
        if negative:
            return 1 - predicate[idx]
        return predicate[idx]


class BatchMaskProbConstructor(Constructor):
    def get_predication(self, predicate, idx, negative=False):
        value, mask = predicate
        if negative:
            value = 1 - value
        return value[(slice(value.shape[0]),*idx)], mask[(slice(mask.shape[0]),*idx)]

    def isskip(self, value):
        return False