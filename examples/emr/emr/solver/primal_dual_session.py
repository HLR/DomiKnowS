import torch
from regr.solver.session.solver_session import SolverSession


class PrimalDualSession(SolverSession):
    def __init__(self):
        self.vars = {}
        self.constrs = {}

    @staticmethod
    def autoname(container, prefix='auto'):
        num = len(container)
        name = '{}_{}'.format(prefix, num)
        while name in container:
            num += 1
            name = '{}_{}'.format(prefix, num)
        return name

    VMAP = {
        SolverSession.VTYPE.BIN: torch.bool,
        SolverSession.VTYPE.INT: torch.int,
        SolverSession.VTYPE.DEC: torch.float}
    def var(self, vtype, lb, ub, name=None):
        name = name or self.autoname(self.vars)
        var = torch.empty((), dtype=self.VMAP[vtype])
        self.vars[name] = var
        # TODO: handle lb and ub
        return var

    def constr(self, lhs, ctype, rhs, name=None):
        name = name or self.autoname(self.constrs)
        if ctype == SolverSession.CTYPE.EQ:
            penalty = max(0, lhs - rhs) + max(0, rhs - lhs)
            constr = penalty
        elif ctype in (SolverSession.CTYPE.GE, SolverSession.CTYPE.GT):
            penalty = max(0, lhs - rhs)
            constr = penalty
        elif ctype in (SolverSession.CTYPE.LE, SolverSession.CTYPE.LT):
            penalty = max(0, rhs - lhs)
            constr = penalty
        self.constrs[name] = penalty
        return constr

    def obj(self, otype, expr):
        pass

    def optimize(self):
        pass

    def get_value(self, var):
        return var
