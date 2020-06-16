from enum import Enum


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
