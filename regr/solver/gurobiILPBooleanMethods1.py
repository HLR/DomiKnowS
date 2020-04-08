import logging
from itertools import permutations

from regr.solver.ilpBooleanMethods import ilpBooleanProcessor 
from regr.solver.ilpConfig import ilpConfig 

from gurobipy import *

class gurobiILPBooleanProcessor(ilpBooleanProcessor):
    
    def __init__(self, _ildConfig = ilpConfig) -> None:
        super().__init__()
                
        self.myLogger = logging.getLogger(ilpConfig['log_name'])
        self.constrainCaches = {}

    def resetCaches(self):
         self.constrainCaches = {}

    def __addToConstrainCaches(self, lmName, onlyConstrains, var, cachedValue):
        if lmName in self.constrainCaches:
                if onlyConstrains in self.constrainCaches[lmName]:
                    self.constrainCaches[lmName][onlyConstrains][var] = cachedValue
                else:
                    self.constrainCaches[lmName][onlyConstrains] = {}
                    self.constrainCaches[lmName][onlyConstrains][var] = cachedValue
        else:
            self.constrainCaches[lmName] = {}
            self.constrainCaches[lmName][onlyConstrains] = {}

            self.constrainCaches[lmName][onlyConstrains][var] = cachedValue
            
    def __isInConstrainCaches(self, lmName, onlyConstrains, var):
        if lmName in self.constrainCaches:
            if onlyConstrains in self.constrainCaches[lmName]:
                for currentVarPermutation in permutations(var):
                    if currentVarPermutation in self.constrainCaches[lmName][onlyConstrains]:
                        #self.myLogger.debug("%s already created constrain for this variables %s - does nothing"%(lmName, [x.VarName for x in var]))
                        return (True, self.constrainCaches[lmName][onlyConstrains][currentVarPermutation])
                    
        return (False, None)
                
    def notVar(self, m, var, onlyConstrains = False):
        #self.myLogger.debug("NOT called with : %s"%(var))

        cacheResult = self.__isInConstrainCaches('notVar', onlyConstrains, var)
        if cacheResult[0]:
            return cacheResult[1]
            
        if onlyConstrains:
            m.addConstr(1 - var >= 1)

            self.__addToConstrainCaches('notVar', onlyConstrains, (var1, ), None)

            return
        
        varNOT = m.addVar(vtype=GRB.BINARY, name="not_%s"%(var.VarName))
        
        m.addConstr(1 - var == varNOT)
    
        m.update()

        return varNOT
    
    def and2Var(self, m, var1, var2, onlyConstrains = False):
        #self.myLogger.debug("AND called with : %s"%(var1,var2))

        if onlyConstrains:
            m.addConstr(var1 + var2 >= 2)
            return
        
        varAND = m.addVar(vtype=GRB.BINARY, name="and_%s_%s"%(var1, var2))
            
        m.addConstr(varAND <= var1)
        m.addConstr(varAND <= var2)
        m.addConstr(var1 + var2 <= varAND + 2 - 1)
            
        return varAND
    
    def andVar(self, m, *var, onlyConstrains = False):
        N = len(var)
        
        if N <= 1:
            return None
        
        if onlyConstrains:
            varSumLinExpr = LinExpr()
            for currentVar in var:
                varSumLinExpr.addTerms(1.0, currentVar)
        
            m.addConstr(varSumLinExpr >= N)
            return
            
        varAND = m.addVar(vtype=GRB.BINARY)
        for currentVar in var:
            m.addConstr(varAND <= currentVar)

        varSumLinExpr = LinExpr()
        for currentVar in var:
            varSumLinExpr.addTerms(1.0, currentVar)
        
        m.addConstr(varSumLinExpr <= varAND + N - 1)
    
        return varAND
    
    def or2Var(self, m, var1, var2, onlyConstrains = False):
        #self.myLogger.debug("OR called with : %s"%(var1,var2))

        if onlyConstrains:
            m.addConstr(var1 + var2 >= 1)
            return
        
        varOR=m.addVar(vtype=GRB.BINARY, name="or_%s_%s"%(var1, var2))
            
        m.addConstr(var1 <= varOR)
        m.addConstr(var2 <= varOR)
            
        m.addConstr(var1 + var2 >= varOR)
    
        return varOR
    
    def orVar(self, m, *var, onlyConstrains = False):
        #self.myLogger.debug("OR called with : %s"%(var,))
        
        cacheResult = self.__isInConstrainCaches('orVar', onlyConstrains, var)
        if cacheResult[0]:
            return cacheResult[1]

        if onlyConstrains:
            varSumLinExpr = LinExpr()
            for currentVar in var:
                varSumLinExpr.addTerms(1.0, currentVar)
        
            m.addConstr(varSumLinExpr >= 1)
            
            varSumLinExprStr = str(varSumLinExpr)

            self.__addToConstrainCaches('nandVar', onlyConstrains, var, None)           
            return
        
        N = len(var)
        
        if N <= 1:
            return None
        
        orVarName = "OR_VAR-"
        for currentVar in var:
            if currentVar is None:
                continue
            orVarName += "-%s" % (currentVar.VarName)
        
        varOR = m.addVar(vtype=GRB.BINARY, name=orVarName)

        for currentVar in var:
            if currentVar is None:
                continue
            m.addConstr(currentVar <= varOR)

        varSumLinExpr = LinExpr()
        for currentVar in var:
            varSumLinExpr.addTerms(1.0, currentVar)
        
        m.addConstr(varSumLinExpr >= varOR)

        self.__addToConstrainCaches('nandVar', onlyConstrains, var, varOR)

        m.update()
             
        return varOR
    
    def nand2Var(self, m, var1, var2, onlyConstrains = False):
        #self.myLogger.debug("NAND called with : %s"%(var1,var2))

        if onlyConstrains:
            m.addConstr(var1 + var2 <= 1)
            return
        
        varNAND = m.addVar(vtype=GRB.BINARY, name="nand_%s_%s"%(var1, var2))
            
        m.addConstr(self.notVar(m, varNAND) <= var1)
        m.addConstr(self.notVar(m, varNAND) <= var2)
        
        m.addConstr(var1 + var2 <= self.notVar(m, varNAND) + 2 - 1)
        
        return varNAND
    
    def nandVar(self, m, *var, onlyConstrains = False):
        #self.myLogger.debug("NAND called with : %s"%(var,))
        
        cacheResult = self.__isInConstrainCaches('nandVar', onlyConstrains, var)
        if cacheResult[0]:
            return cacheResult[1]
            
        N = len(var)
        
        if N <= 1:
            self.myLogger.debug("NAND returns : %s"%('None'))
            return None
        
        if onlyConstrains:
            varSumLinExpr = LinExpr()
            for currentVar in var:
                try:
                    varSumLinExpr.addTerms(1.0, currentVar)
                except  GurobiError:
                    pass
        
            m.addConstr(varSumLinExpr <= N - 1)
                        
            varSumLinExprStr = str(varSumLinExpr)

            self.__addToConstrainCaches('nandVar', onlyConstrains, var, None)
      
            return
        
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
    
        self.__addToConstrainCaches('nandVar', onlyConstrains, var, varNAND)

        return varNAND
    
    def nor2Var(self, m, var1, var2, onlyConstrains = False):
        if onlyConstrains:
            m.addConstr(var1 + var2 <= 0)
            return
        
        varNOR = m.addVar(vtype=GRB.BINARY, name="nor_%s_%s"%(var1, var2))
            
        m.addConstr(var1 <= self.notVar(m, varNOR))
        m.addConstr(var2 <= self.notVar(m, varNOR))
            
        m.addConstr(var1 + var2 >= self.notVar(m, varNOR))
        
        return varNOR
    
    def norVar(self, m, *var, onlyConstrains = False):
        N = len(var)
        
        if N <= 1:
            return None
        
        if onlyConstrains:
            varSumLinExpr = LinExpr()
            for currentVar in var:
                varSumLinExpr.addTerms(1.0, currentVar)
        
            m.addConstr(varSumLinExpr <= 0)
            return
        
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
    
    def xorVar(self, m, var1, var2, onlyConstrains = False):
        #self.myLogger.debug("XOR called with : %s"%(var1,var2))
        
        cacheResult = self.__isInConstrainCaches('xorVar', onlyConstrains, (var1, var2))
        if cacheResult[0]:
            return cacheResult[1]

        if onlyConstrains:
            m.addConstr(var1 + var2 <= 1)
            m.addConstr(var1 + var2 >= 1)

            self.__addToConstrainCaches('ifVar', onlyConstrains, (var1, var2), None)
            return
        
        varXOR = m.addVar(vtype=GRB.BINARY, name="xor_%s_%s"%(var1, var2))
            
        m.addConstr(var1 + var2 + varXOR <= 2)
        m.addConstr(-var1 - var2 + varXOR <= 0)
        m.addConstr(var1 - var2 + varXOR >= 0)
        m.addConstr(-var1 + var2 + varXOR >= 0)
            
        return varXOR
    
    def ifVar(self, m, var1, var2, onlyConstrains = False):
        #self.myLogger.debug("IF called with : %s"%(var1,var2))

        if (not var1) or (not var2):
            return
    
        cacheResult = self.__isInConstrainCaches('ifVar', onlyConstrains, (var1, var2))
        if cacheResult[0]:
            return cacheResult[1]
            
        if onlyConstrains:
            m.addConstr(var1 <= var2)
            self.__addToConstrainCaches('ifVar', onlyConstrains, (var1, var2), None)
            return
        
        varIF = m.addVar(vtype=GRB.BINARY, name="if_%s_%s"%(var1, var2))
            
        m.addConstr(1 - var1 <= varIF)
        m.addConstr(var2 <= varIF)
        m.addConstr(1 - var1 + var2 >= varIF)
            
        m.update()

        self.__addToConstrainCaches('ifVar', onlyConstrains, (var1, var2), varIF)

        return varIF
           
    def eqVar(self, m, var1, var2, onlyConstrains = False):
        #self.myLogger.debug("EQ called with : %s"%(var1,var2))

        cacheResult = self.__isInConstrainCaches('eqVar', onlyConstrains, (var1, var2))
        if cacheResult[0]:
            return cacheResult[1]
        
        if onlyConstrains:
            m.addConstr(var1 >= var2)

            m.addConstr(var1 <= var2)
            self.__addToConstrainCaches('eqVar', onlyConstrains, (var1, var2), None)
            return
        
        varEQ = m.addVar(vtype=GRB.BINARY, name="epq_%s_%s"%(var1, var2))
            
        m.addConstr(var1 + var2 - varEQ <= 1)
        m.addConstr(var1 + var2 + varEQ >= 1)
        m.addConstr(-var1 + var2 + varEQ <= 1)
        m.addConstr(var1 - var2 + varEQ <= 1)
        
        m.update()
             
        self.__addToConstrainCaches('eqVar', onlyConstrains, (var1, var2), varEQ)

        return varEQ