from gurobipy import Model, GRB
import numpy as np
from itertools import product, permutations

from regr.graph.relation import IsA, HasA, NotA
from .solver import Solver


class GurobiSolver(Solver):
    def __init__(self, lazy_not=True, self_relation=True):
        self.lazy_not = lazy_not
        self.self_relation = self_relation

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
                for obj in candidates[concept]: # flat: C-order -> last dim first!
                    var = model.addVar(vtype=GRB.BINARY,
                                       name='{}_{}'.format(concept.name, str(obj)))
                    variables[concept, obj] = var
                    idx, _ = zip(*obj)
                    predictions[concept, obj] = predicate[idx]

        # add constraints
        for predicates in predicates_list:
            for concept in predicates:
                for rel in concept.is_a():
                    # A is_a B : A(x) <= B(x)
                    for x in candidates[rel.src]:
                        if (rel.dst, x) not in variables: continue
                        constr = model.addConstr(variables[rel.src, x] <= variables[rel.dst, x],
                                                 name='{}_{}'.format(rel.name, str(x)))
                        #model.update()
#                         print(rel.name)
#                         print(variables[rel.src, x], '<=', variables[rel.dst, x])
#                         print(constr)
                        constraints[rel, x] = constr
                for rel in concept.not_a():
                    # A not_a B : A(x) + B(x) <= 1
                    for x in candidates[rel.src]:
                        if (rel.dst, x) not in variables: continue
                        constr = model.addConstr(variables[rel.src, x] + variables[rel.dst, x] <= 1,
                                                 name='{}_{}'.format(rel.name, str(x)))
                        #model.update()
#                         print(rel.name)
#                         print(variables[rel.src, x], '+', variables[rel.dst, x], '<= 1')
#                         print(constr)
                        constraints[rel, x] = constr
                for arg_id, rel in enumerate(concept.has_a()): # TODO: need to include indirect ones like sp_tr is a tr while tr has a lm
                    # A has_a B : A(x,y,...) <= B(x)
                    for xy in candidates[rel.src]:
                        x = xy[arg_id]
                        if (rel.dst, (x,)) not in variables: continue
                        #import pdb;pdb.set_trace()
                        constr = model.addConstr(variables[rel.src, xy] <= variables[rel.dst, (x,)],
                                                 name='{}_{}_{}'.format(rel.name, str(xy), str(x)))
                        #model.update()
#                         print(rel.name)
#                         print(variables[rel.src, xy], '<=', variables[rel.dst, (x,)])
#                         print(constr)
                        constraints[rel, xy, x] = constr

        if self.lazy_not:
            variables_not = {} # (concept, (object,...)) -> variable
            predictions_not = {} # (concept, (object,...)) -> prediction
            constraints_not = {} # (rel, (object,...)) -> constr

            # add variables
            for predicates in predicates_list:
                for concept, predicate in predicates.items():
                    for obj in candidates[concept]:
                        var = model.addVar(vtype=GRB.BINARY,
                                           name='lazy_not_{}_{}'.format(concept.name, str(obj)))
                        variables_not[concept, obj] = var
                        idx, _ = zip(*obj)
                        predictions_not[concept, obj] = 1 - predicate[idx]

            # add constraints
            for predicates in predicates_list:
                for concept in predicates:
                    for x in candidates[concept]:
                        constr = model.addConstr(variables[concept, x] + variables_not[concept, x] == 1,
                                               name='lazy_not_{}_{}'.format(concept.name, str(x)))
                        #model.update()
#                         print(concept.name, 'lazy_not')
#                         print(variables[concept, x], '+', variables_not[concept, x], '<= 1')
#                         print(constr)
                        constraints_not[concept, x] = constr

        # Set objective
        objective = None
        for predicates in predicates_list:
            for concept in predicates:
                for x in candidates[concept]:
                    objective += variables[concept, x] * predictions[concept, x]
        if self.lazy_not:
            for predicates in predicates_list:
                for concept in predicates:
                    for x in candidates[concept]:
                        objective += variables_not[concept, x] * predictions_not[concept, x]
        model.setObjective(objective, GRB.MAXIMIZE)

        # solve
        #model.update()
        model.optimize()
        #import pdb;pdb.set_trace()

        if model.status != GRB.Status.OPTIMAL:
            raise model.status

        # collect result
        retval = []
        for arity, predicates in enumerate(predicates_list, 1):
            predicates_result = {}
            retval.append(predicates_result)
            for concept, predicate in predicates.items():
                predicates_result[concept] = np.zeros((length,) * arity)
                for obj in candidates[concept]:
                    # NB: candidates generated by 'C' order
                    idx, _ = zip(*obj)
                    predicates_result[concept][idx] = variables[concept, obj].x
                #import pdb;pdb.set_trace()

        if len(retval) == 1:
            return retval[0]

        return retval

# for (c, x), v in variables.items(): print(c.name,x,v)
# for (c, x), v in predictions.items(): print(c.name,x,v)
# for (c, x), v in constraints.items(): print(c.name,x,v)
# for c in model.getConstrs(): print(c)
