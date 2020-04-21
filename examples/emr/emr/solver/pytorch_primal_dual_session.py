import torch
from regr.solver.solver_session import SolverSession


class PytorchPrimalDualSession(SolverSession):
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
            la = torch.ones((2,), dtype=torch.float)
            penalty = la[0] * max(0, lhs - rhs) + la[1] * max(0, rhs - lhs)
            constr = (la, penalty)
        elif ctype in (SolverSession.CTYPE.GE, SolverSession.CTYPE.GT):
            la = torch.ones((), dtype=torch.float)
            penalty = la * max(0, lhs - rhs)
            constr = (la, penalty)
        elif ctype in (SolverSession.CTYPE.LE, SolverSession.CTYPE.LT):
            la = torch.ones((), dtype=torch.float)
            penalty = la * max(0, rhs - lhs)
            constr = (la, penalty)
        self.constrs[name] = (la, penalty)
        return constr

    def obj(self, otype, expr):
        pass

    def optimize(self):
        pass

    def get_value(self, var):
        return var
