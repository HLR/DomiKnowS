from enum import Enum
from itertools import product, permutations
import warnings

import numpy as np
from gurobipy import Model, GRB

from regr.utils import isbad
from .gurobi_solver import GurobiSolver


class SolverSession:
    class VTYPE(Enum):
        BIN = bool
        INT = int
        DEC = float

    class CTYPE(Enum):
        EQ = '=='
        LT = '<'
        LE = '<='
        GT = '>'
        GE = '>='

    class OTYPE(Enum):
        MAX = 'max'
        MIN = 'min'

    def var(self, vtype, lb, ub, name=None):
        raise NotImplementedError

    def constr(self, lhs, ctype, rhs, name=None):
        raise NotImplementedError

    def obj(self, otype, expr):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError


class GurobiSession(SolverSession):
    def __init__(self, silence=True):
        self.model = Model('solver')
        self.silence = silence
        if silence:
            self.model.params.outputflag = 0

    def __str__(self):
        return str(self.model)

    VMAP = {
        SolverSession.VTYPE.BIN: GRB.BINARY,
        SolverSession.VTYPE.INT: GRB.INTEGER,
        SolverSession.VTYPE.DEC: GRB.CONTINUOUS}
    def var(self, vtype, lb, ub, name=None):
        var = self.model.addVar(lb, ub, vtype=self.VMAP[vtype], name=name)
        self.model.update()
        return var

    CMAP = {
        SolverSession.CTYPE.EQ: (1, GRB.EQUAL),
        SolverSession.CTYPE.LT: (-1, GRB.GREATER_EQUAL),
        SolverSession.CTYPE.LE: (1, GRB.LESS_EQUAL),
        SolverSession.CTYPE.GT: (-1, GRB.LESS_EQUAL),
        SolverSession.CTYPE.GE: (1, GRB.GREATER_EQUAL)}
    def constr(self, lhs, ctype, rhs, name=None):
        coeff, ctype = self.CMAP[ctype]
        constr = self.model.addConstr(coeff * lhs, ctype, coeff * rhs, name)
        self.model.update()
        return constr

    OMAP = {
        SolverSession.OTYPE.MAX: GRB.MAXIMIZE,
        SolverSession.OTYPE.MIN: GRB.MINIMIZE}
    def obj(self, otype, expr):
        obj = self.model.setObjective(expr, self.OMAP[otype])
        self.model.update()
        return obj

    def optimize(self):
        self.model.optimize()
        if self.model.status != GRB.Status.OPTIMAL:
            warnings.warn('Model did not finish in an optimal status! Status code is {}.'.format(self.model.status), RuntimeWarning)

    def get_value(self, var):
        return var.x

class MiniSolverDebug(GurobiSolver):
    ilpSolver = 'mini_debug'

    def get_predication(self, predicate, idx, negative=False):
        if negative:
            return predicate[(*idx, 0)]
        return predicate[(*idx, 1)]

    def set_predication(self, predicate, idx, value):
        # predicates_result[concept][idx] = session.get_value(variables[concept, x])
        predicate[idx] = value

    def solve_legacy(self, data, *predicates_list):
        #self.myLogger.setLevel(logging.DEBUG)

        # data is a list of objects of the base type
        # predicates_list is a list of predicates
        # predicates_list[i] is the dict of predicates of the objects of the base type to the power of i
        # predicates_list[i][concept] is the prediction result of the predicate for concept
        self.myLogger.debug('Start for data %s', data)
        session = GurobiSession()

        # prepare candidates
        length = len(data)
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

        variables = {} # (concept, (object,...)) -> variable
        predictions = {} # (concept, (object,...)) -> prediction
        constraints = {} # (rel, (object,...)) -> constr

        # add variables
        self.myLogger.debug('add variables')
        for predicates in predicates_list:
            for concept, predicate in predicates.items():
                self.myLogger.debug('for %s', concept.name)
                self.myLogger.debug(predicate)
                for x in candidates[concept]: # flat: C-order -> last dim first!
                    idx, _ = zip(*x)
                    prediction = self.get_predication(predicate, idx)
                    if isbad(prediction): continue
                    var = session.var(
                        session.VTYPE.BIN, 0, 1,
                        name='{}_{}'.format(concept.name, str(x)))
                    self.myLogger.debug(' - add %s', var)
                    variables[concept, x] = var
                    predictions[concept, x] = prediction

        # add constraints
        self.myLogger.debug('add constraints')
        for predicates in predicates_list:
            for concept in predicates:
                self.myLogger.debug('for %s', concept.name)
                self.myLogger.debug(' - is_a')
                for rel in concept.is_a():
                    self.myLogger.debug(' - - %s', rel.name)
                    # A is_a B : A(x) <= B(x)
                    for x in candidates[rel.src]:
                        if (rel.src, x) not in variables: continue
                        if (rel.dst, x) not in variables: continue
                        constr = session.constr(
                            variables[rel.src, x], SolverSession.CTYPE.LE, variables[rel.dst, x],
                            name='{}_{}'.format(rel.name, str(x)))
                        self.myLogger.debug(' - - add %s', constr)
                        constraints[rel, x] = constr
                self.myLogger.debug(' - not_a')
                for rel in concept.not_a():
                    self.myLogger.debug(' - - %s', rel.name)
                    # A not_a B : A(x) + B(x) <= 1
                    for x in candidates[rel.src]:
                        if (rel.src, x) not in variables: continue
                        if (rel.dst, x) not in variables: continue
                        constr = session.constr(
                            variables[rel.src, x] + variables[rel.dst, x], SolverSession.CTYPE.LE, 1,
                            name='{}_{}'.format(rel.name, str(x)))
                        self.myLogger.debug(' - - add %s', constr)
                        constraints[rel, x] = constr
                self.myLogger.debug(' - has_a')
                for arg_id, rel in enumerate(concept.has_a()): # TODO: need to include indirect ones like sp_tr is a tr while tr has a lm
                    self.myLogger.debug(' - - %s', rel.name)
                    # A has_a B : A(x,y,...) <= B(x)
                    for xy in candidates[rel.src]:
                        x = xy[arg_id]
                        if (rel.src, xy) not in variables: continue
                        if (rel.dst, (x,)) not in variables: continue
                        constr = session.constr(
                            variables[rel.src, xy], SolverSession.CTYPE.LE, variables[rel.dst, (x,)],
                            name='{}_{}_{}'.format(rel.name, str(xy), str(x)))
                        self.myLogger.debug(' - - add %s', constr)
                        constraints[rel, xy, x] = constr

        if self.lazy_not:
            self.myLogger.debug('lazy negative')
            variables_not = {} # (concept, (x,...)) -> variable
            predictions_not = {} # (concept, (x,...)) -> prediction
            constraints_not = {} # (rel, (x,...)) -> constr

            # add variables
            self.myLogger.debug('lazy negative add variables')
            for predicates in predicates_list:
                for concept, predicate in predicates.items():
                    self.myLogger.debug('for %s', concept.name)
                    for x in candidates[concept]:
                        idx, _ = zip(*x)
                        prediction_not = self.get_predication(predicate, idx, negative=True)
                        if isbad(prediction_not): continue
                        var = session.var(
                            session.VTYPE.BIN, 0, 1,
                            name='lazy_not_{}_{}'.format(concept.name, str(x)))
                        self.myLogger.debug(' - add %s', var)
                        variables_not[concept, x] = var
                        predictions_not[concept, x] = prediction_not

            # add constraints
            self.myLogger.debug('lazy negative add constraints')
            for predicates in predicates_list:
                for concept in predicates:
                    self.myLogger.debug('for %s', concept.name)
                    for x in candidates[concept]:
                        if (concept, x) not in variables: continue
                        if (concept, x) not in variables_not: continue
                        constr = session.constr(
                            variables[concept, x] + variables_not[concept, x], SolverSession.CTYPE.EQ, 1,
                            name='lazy_not_{}_{}'.format(concept.name, str(x)))
                        self.myLogger.debug(' - add %s', constr)
                        constraints_not[concept, x] = constr

        # Set objective
        self.myLogger.debug('set objective')
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
        session.obj(SolverSession.OTYPE.MAX, objective)
        self.myLogger.debug(' - model %s', session)

        # solve
        self.myLogger.debug('start optimize')
        session.optimize()
        self.myLogger.debug('end optimize')
        self.myLogger.debug(' - model %s', session)

        # collect result
        retval = []
        for arity, predicates in enumerate(predicates_list, 1):
            predicates_result = {}
            retval.append(predicates_result)
            for concept, predicate in predicates.items():
                self.myLogger.debug('for %s', concept.name)
                predicates_result[concept] = np.zeros((length,) * arity)
                for x in candidates[concept]:
                    #import pdb; pdb.set_trace()
                    if (concept, x) not in variables: continue
                    # NB: candidates generated by 'C' order
                    idx, _ = zip(*x)
                    value = session.get_value(variables[concept, x])
                    self.set_predication(predicates_result[concept], idx, value)
                #import pdb;pdb.set_trace()
                self.myLogger.debug(predicates_result[concept])

        if len(retval) == 1:
            return retval[0]

        return retval


class Mini(MiniSolverDebug):
    ilpSolver = 'mini_prob_debug'

    def get_predication(self, predicate, idx, negative=False):
        if negative:
            return 1 - predicate[idx]
        return predicate[idx]
