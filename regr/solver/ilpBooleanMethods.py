import abc

if __package__ is None or __package__ == '':
    from regr.solver.ilpConfig import ilpConfig 
else:
    from .ilpConfig import ilpConfig 
    
class ilpBooleanProcessor(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, _ildConfig = ilpConfig) -> None:
        super().__init__()

    # Negation
    @abc.abstractmethod
    def notVar(m, _var): pass
    
    # Conjunction 2 variable
    @abc.abstractmethod
    def and2Var(m, _var1, _var2): pass
        
    # Conjunction
    @abc.abstractmethod
    def andVar(m, *_var): pass
       
    # Disjunction 2 variables
    @abc.abstractmethod
    def or2Var(m, _var1, _var2): pass
        
    # Disjunction
    @abc.abstractmethod
    def orVar(m, *_var): pass

    # Nand (Alternative denial) 2 variables
    @abc.abstractmethod
    def nand2Var(m, _var1, _var2): pass

    # Nand (Alternative denial)
    @abc.abstractmethod
    def nandVar(m, *_var): pass
  
    # Nor (Joint Denial) i2 variables
    @abc.abstractmethod
    def nor2Var(m, _var1, _var2): pass

    # Nor (Joint Denial)
    @abc.abstractmethod
    def norVar(m, *_var): pass

    # Exclusive Disjunction
    @abc.abstractmethod
    def xorVar(m, _var1, _var2): pass

    # Implication
    @abc.abstractmethod
    def ifVar(m, _var1, _var2): pass

    # Equivalence 
    @abc.abstractmethod
    def epqVar(m, _var1, _var2): pass