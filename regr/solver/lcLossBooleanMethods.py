import logging

from regr.solver.ilpBooleanMethods import ilpBooleanProcessor 
from regr.solver.ilpConfig import ilpConfig 

class lcLossBooleanMethods(ilpBooleanProcessor):
    
    def __init__(self, _ildConfig = ilpConfig) -> None:
        super().__init__()
                
        self.myLogger = logging.getLogger(ilpConfig['log_name'])
        self.ifLog =  ilpConfig['ifLog']

    def notVar(self, m, var, onlyConstrains = False):
        methodName = "notVar"
        logicMethodName = "NOT"
                
        if self.ifLog: self.myLogger.debug("%s called with : %s"%(logicMethodName,var))

        notLoss = 1 - var   
        
        return notLoss
    
    def and2Var(self, m, var1, var2, onlyConstrains = False):
        methodName = "and2Var"
        logicMethodName = "AND"
            
        if self.ifLog: self.myLogger.debug("%s called with : var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
        
        and2Loss = 2 - var1 - var2
        
        return and2Loss
        
    def andVar(self, m, *var, onlyConstrains = False):
        methodName = "andVar"
        logicMethodName = "AND"
            
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))
                
        N = len(var)
        
        varSum = 0
        for currentVar in var:
            varSum += currentVar
            
        andLoss = N - varSum
        
        return andLoss       
    
    def or2Var(self, m, var1, var2, onlyConstrains = False):
        methodName = "or2Var"
        logicMethodName = "OR"
       
        if self.ifLog: self.myLogger.debug("%s called with : var1 - %s, var2 - %s"%(logicMethodName,var1,var2))

        or2Loss = 1 - var1 - var2
       
        return or2Loss
    
    def orVar(self, m, *var, onlyConstrains = False):
        methodName = "orVar"
        logicMethodName = "OR"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))
        
        varSum = 0
        for currentVar in var:
            varSum += currentVar
           
        orLoss = 1 - varSum
            
        return orLoss
        
    def nand2Var(self, m, var1, var2, onlyConstrains = False):
        methodName = "nand2Var"
        logicMethodName = "NAND"
        
        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
        
        nand2Loss = var1 + var2 - 1
        
        return nand2Loss
    
    def nandVar(self, m, *var, onlyConstrains = False):
        methodName = "nandVar"
        logicMethodName = "NAND"
       
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))
        
        N = len(var)
        
        varSum = 0
        for currentVar in var:
            varSum += currentVar
        
        nandLoss = varSum - N + 1
                    
        return nandLoss
    
    def ifVar(self, m, var1, var2, onlyConstrains = False):
        methodName = "ifVar"
        logicMethodName = "IF"

        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
     
        ifLoss = var1 - var2
        
        return ifLoss
               
    def norVar(self, m, *var, onlyConstrains = False):
        methodName = "norVar"
        logicMethodName = "NOR"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName,var))

        varSum = 0
        for currentVar in var:
            varSum += currentVar
    
        norLoss = varSum
        
        return norLoss
    
    def xorVar(self, m, var1, var2, onlyConstrains = False):
        methodName = "xorVar"
        logicMethodName = "XOR"
        
        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))

        xorLoss = abs(var1 + var2 - 1)
        
        return xorLoss
    
    def epqVar(self, m, var1, var2, onlyConstrains = False):
        methodName = "epqVar"
        logicMethodName = "EPQ"
        
        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))

        epqLoss = var1 - var2
        
        return epqLoss
     
    def countVar(self, m, *var, onlyConstrains = False, limitOp = 'None', limit = 1):
        methodName = "countVar"
        logicMethodName = "COUNT"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName,var))

        varSum = 0
        for currentVar in var:
            varSum += currentVar
            
        countLoas = 0
        
        if limitOp == '>':     
            countLoas = limit - varSum
            
        elif limitOp == '<':
            countLoas = varSum - limit

        elif limitOp == '=':
            countLoas = abs(varSum - limit)

        return countLoas