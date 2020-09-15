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
        self.vars[name] = None
        return None

    def constr(self, lhs, ctype, rhs, name=None):
        name = name or self.autoname(self.constrs)
        if ctype == SolverSession.CTYPE.EQ:
            penalty = (rhs - lhs).clamp(min=0) + (lhs - rhs).clamp(min=0)
            constr = penalty
        elif ctype in (SolverSession.CTYPE.GE, SolverSession.CTYPE.GT):
            penalty = (rhs - lhs).clamp(min=0)
            constr = penalty
        elif ctype in (SolverSession.CTYPE.LE, SolverSession.CTYPE.LT):
            penalty = (lhs - rhs).clamp(min=0)
            constr = penalty
        self.constrs[name] = penalty
        return constr

    def obj(self, otype, expr):
        pass

    def optimize(self):
        pass

    def get_value(self, var):
        return var
