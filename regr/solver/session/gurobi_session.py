import warnings

from gurobipy import Model, GRB
from .solver_session import SolverSession


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
