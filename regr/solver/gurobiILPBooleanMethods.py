if __package__ is None or __package__ == '':
    from regr.solver.ilpConfig import ilpConfig 
    from regr.solver.ilpBooleanMethods import ilpBooleanProcessor 
else:
    from .ilpConfig import ilpConfig 
    from .ilpBooleanMethods import ilpBooleanProcessor 

from gurobipy import *

class gurobiILPBooleanProcessor(ilpBooleanProcessor):
    
    def __init__(self, _ildConfig = ilpConfig) -> None:
        super().__init__()

    # Negation
    # 1 - var == varNOT
    # TODO Without variable return 1 - var
    def notVar(self, m, var):
        varNOT=m.addVar(vtype=GRB.BINARY, name="not_%s"%(var))
        
        m.addConstr(1 - var == varNOT)
    
        return varNOT
    
    # Conjunction 2 variable
    # varAND <= var1
    # varAND <= var2 
    # var1 + var2 <= varAND + 2 - 1
    # TODO Without variable return var1 + var2 >= 2
    def and2Var(self, m, var1, var2):
        varAND=m.addVar(vtype=GRB.BINARY, name="and_%s_%s"%(var1, var2))
            
        m.addConstr(varAND <= var1)
        m.addConstr(varAND <= var2)
        m.addConstr(var1 + var2 <= varAND + 2 - 1)
            
        return varAND
    
    # Conjunction
    # varAND <= var1
    # varAND <= var2
    # ....
    # varAND <= varN
    # var1 + var2 + .. + varN <= varAND + N - 1
    # TODO Without variable return var1 + var2 + .. + varN >= N
    def andVar(self, m, *var):
        N = len(var)
        if N < 1:
            return None
        
        if N == 1:
            return currentVar[0]
        
        andVarName = "and"
        for currentVar in var:
            andVarName += "_%s"%(currentVar)
            
        varAND = m.addVar(vtype=GRB.BINARY, name=andVarName)
        for currentVar in var:
            m.addConstr(varAND <= currentVar)

        varSumLinExpr = LinExpr()
        for currentVar in var:
            varSumLinExpr.addTerms(1.0, currentVar)
        
        m.addConstr(varSumLinExpr <= varAND + N - 1)
    
        return varAND
    
    # Disjunction 2 variables
    # var1 <= varOR
    # var2 <= varOR 
    # var1 + var2 >= varOR 
    # TODO Without variable return var1 + var2 >= 1
    def or2Var(self, m, var1, var2):
        varOR=m.addVar(vtype=GRB.BINARY, name="or_%s_%s"%(var1, var2))
            
        m.addConstr(var1 <= varOR)
        m.addConstr(var2 <= varOR)
            
        m.addConstr(var1 + var2 >= varOR)
    
        return varOR
    
    # Disjunction
    # var1 <= varOR
    # var2 <= varOR 
    # ...
    # varN <= varOR
    # var1 + var2 + ... + varN >= varOR
    # TODO Without variable return var1 + var2 + ... + varN >= 1
    def orVar(self, m, *var):
        N = len(var)
        
        if N < 1:
            return None
        
        if N == 1:
            return currentVar[0]

        orVarName = "or"
        for currentVar in var:
            orVarName += "_%s" % (currentVar)
        
        varOR = m.addVar(vtype=GRB.BINARY, name=orVarName)

        for currentVar in var:
            m.addConstr(currentVar <= varOR)

        varSumLinExpr = LinExpr()
        for currentVar in var:
            varSumLinExpr.addTerms(1.0, currentVar)
        
        m.addConstr(varSumLinExpr >= varOR)
             
        return varOR
    
    # Nand (Alternative denial) 2 variables
    # not(varNAND) <= var1
    # not(varNAND) <= var2 
    # var1 + var2 <= not(varNAND) + 2 - 1
    # TODO Without variable return var1 + var2 <= 1
    def nand2Var(self, m, var1, var2):
        varNAND = m.addVar(vtype=GRB.BINARY, name="nand_%s_%s"%(var1, var2))
            
        m.addConstr(self.notVar(m, varNAND) <= var1)
        m.addConstr(self.notVar(m, varNAND) <= var2)
        
        m.addConstr(var1 + var2 <= self.notVar(m, varNAND) + 2 - 1)
        
        return varNAND
    
    # Nand (Alternative denial)
    # not(varNAND) <= var1
    # not(varNAND) <= var2 
    # ...
    # not(varNAND <= varN 
    # var1 + var2 + ... + varN <= not(varNAND) + N - 1
    # TODO Without variable return var1 + var2 + ... + varN <= N -1
    def nandVar(self, m, *var):
        N = len(var)
        
        if N < 1:
            return None
        
        if N == 1:
            return currentVar[0]
        
        nandVarName = "nand"
        for currentVar in var:
            nandVarName += "_%s"%(currentVar)
            
        varNAND = m.addVar(vtype=GRB.BINARY, name=nandVarName)
        for currentVar in var:
            m.addConstr(self.notVar(m, varNAND) <= currentVar)

        varSumLinExpr = LinExpr()
        for currentVar in var:
            varSumLinExpr.addTerms(1.0, currentVar)
    
        m.addConstr(varSumLinExpr <= self.notVar(m, varNAND) + N - 1)
    
        return varNAND
    
    # Nor (Joint Denial) 2 variables
    # var1 <= not(varNOR)
    # var2 <= not(varNOR) 
    # var1 + var2 >= not(varNOR)
    # TODO Without variable return var1 + var2 <= 0
    def nor2Var(self, m, var1, var2):
        varNOR = m.addVar(vtype=GRB.BINARY, name="nor_%s_%s"%(var1, var2))
            
        m.addConstr(var1 <= self.notVar(m, varNOR))
        m.addConstr(var2 <= self.notVar(m, varNOR))
            
        m.addConstr(var1 + var2 >= self.notVar(m, varNOR))
        
        return varNOR
    
    # Nor (Joint Denial)
    # Nor (Joint Denial) 2 variables
    # var1 <= not(varNOR)
    # var2 <= not(varNOR) 
    # ...
    # varN <= not(varNOR)
    # var1 + var2 + ... + varN >= not(varNOR)
    # TODO Without variable return var1 + var2 + ... + varN <= 0
    def norVar(self, m, *var):
        N = len(var)
        
        if N < 1:
            return None
        
        if N == 1:
            return currentVar[0]
        
        _norVarName = "nor"
        for currentVar in var:
            _norVarName += "_%s"%(currentVar)
           
        varNOR = m.addVar(vtype=GRB.BINARY, name=norVarName)
        for currentVar in var:
            m.addConstr(currentVar <= self.notVar(m, varNOR))
        
        varSumLinExpr = LinExpr()
        for currentVar in var:
            varSumLinExpr.addTerms(1.0, currentVar)
            
            m.addConstr(varSumLinExpr >= self.notVar(m, varNOR))
    
        return varNOR
    
    # Exclusive Disjunction
    # var1 + var2 + varXOR <= 2
    # -var1 - var2 + varXOR <= 0
    # var1 - var2 + varXOR >= 0
    # -var1 + var2 + varXOR >= 0
    # TODO Without variable return var1 + var2 == 1 
    def xorVar(self, m, var1, var2):
        varXOR = m.addVar(vtype=GRB.BINARY, name="xor_%s_%s"%(var1, var2))
            
        m.addConstr(var1 + var2 + varXOR <= 2)
        m.addConstr(-var1 - var2 + varXOR <= 0)
        m.addConstr(var1 - var2 + varXOR >= 0)
        m.addConstr(-var1 + var2 + varXOR >= 0)
            
        return varXOR
    
    # Implication
    # 1 - var1 <= varIF
    # var2 <= varIF
    # 1 - var1 + var2 >= varIF
    # TODO Without variable return var1 <= var2
    def ifVar(self, m, var1, var2):
        varIF = m.addVar(vtype=GRB.BINARY, name="if_%s_%s"%(var1, var2))
            
        m.addConstr(1 - var1 <= varIF)
        m.addConstr(var2 <= varIF)
        m.addConstr(1 - var1 + var2 >= varIF)
            
        return varIF
           
    # Equivalence 
    # var1 + var2 - varEQ <= 1
    # var1 + var2 + varEQ >= 1
    # -var1 + var2 + varEQ <= 1
    # var1- var2 + varEQ <= 1
    # TODO Without variable return var1 == var2
    def eqVar(self, m, var1, var2):
        varEQ = m.addVar(vtype=GRB.BINARY, name="epq_%s_%s"%(var1, var2))
            
        m.addConstr(var1 + var2 - varEQ <= 1)
        m.addConstr(var1 + var2 + varEQ >= 1)
        m.addConstr(-var1 + var2 + varEQ <= 1)
        m.addConstr(var1 - var2 + varEQ <= 1)
        
        return varEQ