from gurobipy import Model, GRB
import numpy as np
import torch
from itertools import product, permutations
import warnings
import logging

from regr.graph import Concept
from regr.graph.relation import IsA, HasA, NotA
from .gurobi_solver import GurobiSolver


def isbad(x):
    return (
        x != x or  # nan
        abs(x) == float('inf')  # inf
    )


class GurobiSolverDebug(GurobiSolver):
    ilpSolver = 'gurobi_debug'

    def solve_legacy(self, data, *predicates_list):
        self.myLogger.setLevel(logging.DEBUG)

        # data is a list of objects of the base type
        # predicates_list is a list of predicates
        # predicates_list[i] is the dict of predicates of the objects of the base type to the power of i
        # predicates_list[i][concept] is the prediction result of the predicate for concept
        self.myLogger.info('Start for data {}'.format(data))
        model = Model('solver')
        model.params.outputflag = 0

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
        self.myLogger.info('add variables')
        for predicates in predicates_list:
            for concept, predicate in predicates.items():
                self.myLogger.debug('for {}'.format(concept.name))
                self.myLogger.debug('{}'.format(predicate))
                for x in candidates[concept]: # flat: C-order -> last dim first!
                    idx, _ = zip(*x)
                    if isbad(predicate[idx]): continue
                    var = model.addVar(vtype=GRB.BINARY,
                                       name='{}_{}'.format(concept.name, str(x)))
                    model.update()
                    self.myLogger.debug(' - add {}'.format(var))
                    variables[concept, x] = var
                    predictions[concept, x] = predicate[idx]

        # add constraints
        self.myLogger.info('add constraints')
        for predicates in predicates_list:
            for concept in predicates:
                self.myLogger.debug('for {}'.format(concept.name))
                self.myLogger.debug(' - is_a')
                for rel in concept.is_a():
                    self.myLogger.debug(' - - {}'.format(rel.name))
                    # A is_a B : A(x) <= B(x)
                    for x in candidates[rel.src]:
                        if (rel.src, x) not in variables: continue
                        if (rel.dst, x) not in variables: continue
                        constr = model.addConstr(variables[rel.src, x] <= variables[rel.dst, x],
                                                 name='{}_{}'.format(rel.name, str(x)))
                        model.update()
                        self.myLogger.debug(' - - add {}'.format(constr))
                        constraints[rel, x] = constr
                self.myLogger.debug(' - not_a')
                for rel in concept.not_a():
                    self.myLogger.debug(' - - {}'.format(rel.name))
                    # A not_a B : A(x) + B(x) <= 1
                    for x in candidates[rel.src]:
                        if (rel.src, x) not in variables: continue
                        if (rel.dst, x) not in variables: continue
                        constr = model.addConstr(variables[rel.src, x] + variables[rel.dst, x] <= 1,
                                                 name='{}_{}'.format(rel.name, str(x)))
                        model.update()
                        self.myLogger.debug(' - - add {}'.format(constr))
                        constraints[rel, x] = constr
                self.myLogger.debug(' - has_a')
                for arg_id, rel in enumerate(concept.has_a()): # TODO: need to include indirect ones like sp_tr is a tr while tr has a lm
                    self.myLogger.debug(' - - {}'.format(rel.name))
                    # A has_a B : A(x,y,...) <= B(x)
                    for xy in candidates[rel.src]:
                        x = xy[arg_id]
                        if (rel.src, xy) not in variables: continue
                        if (rel.dst, (x,)) not in variables: continue
                        #import pdb;pdb.set_trace()
                        constr = model.addConstr(variables[rel.src, xy] <= variables[rel.dst, (x,)],
                                                 name='{}_{}_{}'.format(rel.name, str(xy), str(x)))
                        model.update()
                        self.myLogger.debug(' - - add {}'.format(constr))
                        constraints[rel, xy, x] = constr

        if self.lazy_not:
            self.myLogger.info('lazy negative')
            variables_not = {} # (concept, (x,...)) -> variable
            predictions_not = {} # (concept, (x,...)) -> prediction
            constraints_not = {} # (rel, (x,...)) -> constr

            # add variables
            self.myLogger.info('lazy negative add variables')
            for predicates in predicates_list:
                for concept, predicate in predicates.items():
                    self.myLogger.debug('for {}'.format(concept.name))
                    for x in candidates[concept]:
                        idx, _ = zip(*x)
                        if isbad(predicate[idx]): continue
                        var = model.addVar(vtype=GRB.BINARY,
                                           name='lazy_not_{}_{}'.format(concept.name, str(x)))
                        model.update()
                        self.myLogger.debug(' - add {}'.format(var))
                        variables_not[concept, x] = var
                        predictions_not[concept, x] = 1 - predicate[idx]

            # add constraints
            self.myLogger.info('lazy negative add constraints')
            for predicates in predicates_list:
                for concept in predicates:
                    self.myLogger.debug('for {}'.format(concept.name))
                    for x in candidates[concept]:
                        if (concept, x) not in variables: continue
                        if (concept, x) not in variables_not: continue
                        constr = model.addConstr(variables[concept, x] + variables_not[concept, x] == 1,
                                               name='lazy_not_{}_{}'.format(concept.name, str(x)))
                        model.update()
                        self.myLogger.debug(' - add {}'.format(constr))
                        constraints_not[concept, x] = constr

        # Set objective
        self.myLogger.info('set objective')
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
        model.setObjective(objective, GRB.MAXIMIZE)
        model.update()
        self.myLogger.debug(' - model {}'.format(model))

        # solve
        #model.update()
        self.myLogger.info('start optimize')
        model.optimize()
        self.myLogger.info('end optimize')
        self.myLogger.info(' - model {}'.format(model))
        #import pdb;pdb.set_trace()

        if model.status != GRB.Status.OPTIMAL:
            warnings.warn('Model did not finish in an optimal status! Status code is {}.'.format(model.status), RuntimeWarning)

        # collect result
        retval = []
        for arity, predicates in enumerate(predicates_list, 1):
            predicates_result = {}
            retval.append(predicates_result)
            for concept, predicate in predicates.items():
                self.myLogger.debug('for {}'.format(concept.name))
                predicates_result[concept] = np.zeros((length,) * arity)
                for x in candidates[concept]:
                    #import pdb; pdb.set_trace()
                    if (concept, x) not in variables: continue
                    # NB: candidates generated by 'C' order
                    idx, _ = zip(*x)
                    predicates_result[concept][idx] = variables[concept, x].x
                #import pdb;pdb.set_trace()
                self.myLogger.debug('{}'.format(predicates_result[concept]))

        if len(retval) == 1:
            return retval[0]

        return retval

# for (c, x), v in variables.items(): print(c.name,x,v)
# for (c, x), v in predictions.items(): print(c.name,x,v)
# for (c, x), v in constraints.items(): print(c.name,x,v)
# for c in model.getConstrs(): print(c)
