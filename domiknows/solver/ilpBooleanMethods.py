import abc

if __package__ is None or __package__ == '':
    from domiknows.solver.ilpConfig import ilpConfig 
else:
    from .ilpConfig import ilpConfig 
    
class ilpBooleanProcessor(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, _ildConfig = ilpConfig) -> None:
        super().__init__()

    # NOT Negation
    # Create new variable varNOT, create constrain 1 - var == varNOT, return varNOT
    # 1 - var == varNOT
    # if onlyConstrains then only construct constrain 1 - var >= 0
    @abc.abstractmethod
    def notVar(self, m, _var, onlyConstrains = False): pass
    
    # AND Conjunction with 2 variable
    # Create new variable varAND, create constrains:
    # varAND <= var1
    # varAND <= var2 
    # var1 + var2 <= varAND + 2 - 1
    # return varAND
    # if onlyConstrains then only construct constrain var1 + var2 >= 2
    @abc.abstractmethod
    def and2Var(self, m, _var1, _var2,  onlyConstrains = False): pass
        
    # AND Conjunction
    # Create new variable varAND, create constrains:
    # varAND <= var1
    # varAND <= var2
    # ....
    # varAND <= varN
    # var1 + var2 + .. + varN <= varAND + N - 1
    # return varAND
    # if onlyConstrains then only construct constrain var1 + var2 + .. + varN >= N
    @abc.abstractmethod
    def andVar(self, m, *_var, onlyConstrains = False): pass
       
    # OR Disjunction with 2 variables
    # Create new variable varOR, create constrains:
    # var1 <= varOR
    # var2 <= varOR 
    # var1 + var2 >= varOR 
    # return varOR
    # if onlyConstrains then only construct constrain var1 + var2 >= 1
    #
    # if limit > 1
    # var1 + var2 - (limit - 1) >= varOR 
    # if onlyConstrains then only construct constrain var1 + var2 >= limit
    @abc.abstractmethod
    def or2Var(self, m, _var1, _var2,  onlyConstrains = False): pass
        
    # OR Disjunction
    # Create new variable varOR, create constrains:
    # var1 <= varOR
    # var2 <= varOR 
    # ...
    # varN <= varOR
    # var1 + var2 + ... + varN >= varOR
    # return varOR
    # if onlyConstrains then only construct constrain var1 + var2 + ... + varN >= 1
    # if limit > 1
    # var1 + var2 + ... + varN - (limit - 1) >= varOR 
    # if onlyConstrains then only construct constrain var1 + var2 + ... + varN >= limit
    @abc.abstractmethod
    def orVar(self, m, *_var, onlyConstrains = False): pass

    # NAND (Alternative denial) with 2 variables
    # Create new variable varNAND, create constrains:
    # not(varNAND) <= var1
    # not(varNAND) <= var2 
    # var1 + var2 <= not(varNAND) + 2 - 1
    # return varNAND
    # if onlyConstrains then only construct constrain var1 + var2 <= 1
    @abc.abstractmethod
    def nand2Var(self, m, _var1, _var2): pass

    # NAND (Alternative denial)
    # Create new variable varNAND, create constrains:
    # not(varNAND) <= var1
    # not(varNAND) <= var2 
    # ...
    # not(varNAND <= varN 
    # var1 + var2 + ... + varN <= not(varNAND) + N - 1
    # return varNAND
    # if onlyConstrains then only construct constrain var1 + var2 + ... + varN <= N -1
    @abc.abstractmethod
    def nandVar(self, m, *_var, onlyConstrains = False): pass
  
    # NOR (Joint Denial) 2 variables
    # Create new variable varNOR, create constrains:
    # var1 <= not(varNOR)
    # var2 <= not(varNOR) 
    # var1 + var2 >= not(varNOR)
    # return varNOR
    # if onlyConstrains then only construct constrain var1 + var2 <= 0
    @abc.abstractmethod
    def nor2Var(self, m, _var1, _var2): pass

    # NOR (Joint Denial)
    # Create new variable varNOR, create constrains:
    # var1 <= not(varNOR)
    # var2 <= not(varNOR) 
    # ...
    # varN <= not(varNOR)
    # var1 + var2 + ... + varN >= not(varNOR)
    # return varNOR
    # if onlyConstrains then only construct constrain var1 + var2 + ... + varN <= 0
    @abc.abstractmethod
    def norVar(self, m, *_var, onlyConstrains = False): pass

    # XOR Exclusive Disjunction 
    # Create new variable varXOR, create constrains:
    # var1 + var2 + varXOR <= 2
    # -var1 - var2 + varXOR <= 0
    # var1 - var2 + varXOR >= 0
    # -var1 + var2 + varXOR >= 0
    # return varXOR
    # if onlyConstrains then only construct constrain var1 + var2 <= 1 and var1 + var2 >= 1 
    @abc.abstractmethod
    def xorVar(self, m, _var1, _var2,  onlyConstrains = False): pass

    # IF Implication
    # Create new variable varIF, create constrains:
    # 1 - var1 <= varIF
    # var2 <= varIF
    # 1 - var1 + var2 >= varIF
    # return varIF
    # if onlyConstrains then only construct constrain var1 <= var2
    @abc.abstractmethod
    def ifVar(self, m, _var1, _var2,  onlyConstrains = False): pass

    # XNOR Equivalence (EQ)
    # Create new variable varEQ, create constrains:
    # var1 + var2 - varEQ <= 1
    # var1 + var2 + varEQ >= 1
    # -var1 + var2 + varEQ <= 1
    # var1 - var2 + varEQ <= 1
    # return varEQ
    # if onlyConstrains then only construct constrain var1 >= var2 and var1 <= var2
    @abc.abstractmethod
    def epqVar(self, m, _var1, _var2,  onlyConstrains = False): pass
    
    # Create constrain var == label, return 1
    @abc.abstractmethod
    def fixedVar(self, m, _var, onlyConstrains = False): pass