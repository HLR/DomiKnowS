from gurobipy import Model, GRB
import numpy as np
import torch
from itertools import product, permutations
import warnings
import logging

from regr.graph import Concept
from regr.graph.relation import IsA, HasA, NotA
from regr.utils import isbad
from .gurobi_solver import GurobiSolver


class GurobilogSolver(GurobiSolver):
    ilpSolver = 'gurobi_log'

    def solve_legacy(self, data, *predicates_list):
        # data is a list of objects of the base type
        # predicates_list is a list of predicates
        # predicates_list[i] is the dict of predicates of the objects of the base type to the power of i
        # predicates_list[i][concept] is the prediction result of the predicate for concept
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
        for predicates in predicates_list:
            for concept, predicate in predicates.items():
                for x in candidates[concept]: # flat: C-order -> last dim first!
                    idx, _ = zip(*x)
                    if isbad(predicate[(*idx, 1)]): continue
                    var = model.addVar(vtype=GRB.BINARY,
                                       name='{}_{}'.format(concept.name, str(x)))
                    model.update()
                    variables[concept, x] = var
                    predictions[concept, x] = predicate[(*idx, 1)]

        # add constraints
        for predicates in predicates_list:
            for concept in predicates:
                for rel in concept.is_a():
                    # A is_a B : A(x) <= B(x)
                    for x in candidates[rel.src]:
                        if (rel.src, x) not in variables: continue
                        if (rel.dst, x) not in variables: continue
                        constr = model.addConstr(variables[rel.src, x] <= variables[rel.dst, x],
                                                 name='{}_{}'.format(rel.name, str(x)))
                        model.update()
                        constraints[rel, x] = constr
                for rel in concept.not_a():
                    # A not_a B : A(x) + B(x) <= 1
                    for x in candidates[rel.src]:
                        if (rel.src, x) not in variables: continue
                        if (rel.dst, x) not in variables: continue
                        constr = model.addConstr(variables[rel.src, x] + variables[rel.dst, x] <= 1,
                                                 name='{}_{}'.format(rel.name, str(x)))
                        model.update()
                        constraints[rel, x] = constr
                for arg_id, rel in enumerate(concept.has_a()): # TODO: need to include indirect ones like sp_tr is a tr while tr has a lm
                    # A has_a B : A(x,y,...) <= B(x)
                    for xy in candidates[rel.src]:
                        x = xy[arg_id]
                        if (rel.src, xy) not in variables: continue
                        if (rel.dst, (x,)) not in variables: continue
                        #import pdb;pdb.set_trace()
                        constr = model.addConstr(variables[rel.src, xy] <= variables[rel.dst, (x,)],
                                                 name='{}_{}_{}'.format(rel.name, str(xy), str(x)))
                        model.update()
                        constraints[rel, xy, x] = constr

        if self.lazy_not:
            variables_not = {} # (concept, (x,...)) -> variable
            predictions_not = {} # (concept, (x,...)) -> prediction
            constraints_not = {} # (rel, (x,...)) -> constr

            # add variables
            for predicates in predicates_list:
                for concept, predicate in predicates.items():
                    for x in candidates[concept]:
                        idx, _ = zip(*x)
                        if (concept, x) not in variables: continue
                        var = model.addVar(vtype=GRB.BINARY,
                                           name='lazy_not_{}_{}'.format(concept.name, str(x)))
                        model.update()
                        variables_not[concept, x] = var
                        predictions_not[concept, x] = predicate[(*idx, 0)]

            # add constraints
            for predicates in predicates_list:
                for concept in predicates:
                    for x in candidates[concept]:
                        if (concept, x) not in variables: continue
                        if (concept, x) not in variables_not: continue
                        constr = model.addConstr(variables[concept, x] + variables_not[concept, x] == 1,
                                               name='lazy_not_{}_{}'.format(concept.name, str(x)))
                        model.update()
                        constraints_not[concept, x] = constr

        # Set objective
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

        # solve
        #model.update()
        model.optimize()
        #import pdb;pdb.set_trace()

        if model.status != GRB.Status.OPTIMAL:
            warnings.warn('Model did not finish in an optimal status! Status code is {}.'.format(model.status), RuntimeWarning)

        # collect result
        retval = []
        for arity, predicates in enumerate(predicates_list, 1):
            predicates_result = {}
            retval.append(predicates_result)
            for concept, predicate in predicates.items():
                predicates_result[concept] = np.zeros((length,) * arity)
                for x in candidates[concept]:
                    #import pdb; pdb.set_trace()
                    if (concept, x) not in variables: continue
                    # NB: candidates generated by 'C' order
                    idx, _ = zip(*x)
                    predicates_result[concept][idx] = variables[concept, x].x
                #import pdb;pdb.set_trace()

        if len(retval) == 1:
            return retval[0]

        return retval

# for (c, x), v in variables.items(): print(c.name,x,v)
# for (c, x), v in predictions.items(): print(c.name,x,v)
# for (c, x), v in constraints.items(): print(c.name,x,v)
# for c in model.getConstrs(): print(c)
