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
        
        notSuccess = 1 - var

        if onlyConstrains:
            notLoss = 1 - notSuccess # var   
            
            return notLoss
        else:            
            return notSuccess
    
    def and2Var(self, m, var1, var2, onlyConstrains = False):
        methodName = "and2Var"
        logicMethodName = "AND"
            
        if self.ifLog: self.myLogger.debug("%s called with : var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
        
        and2Success = max(var1 + var2 - 1, 0)
         
        if onlyConstrains:
            and2Loss = 1 - and2Success # min(2 - var1 - var2, 1)
            
            return and2Loss
        else:
            return and2Success
        
    def andVar(self, m, *var, onlyConstrains = False):
        methodName = "andVar"
        logicMethodName = "AND"
            
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))
                
        N = len(var)
        
        varSum = 0
        for currentVar in var:
            varSum += currentVar
            
        andSuccess = max(varSum - N + 1, 0)

        if onlyConstrains:
            andLoss = 1 - andSuccess # min(N - varSum, 1)
        
            return andLoss       
        else:
            return andSuccess
    
    def or2Var(self, m, var1, var2, onlyConstrains = False):
        methodName = "or2Var"
        logicMethodName = "OR"
       
        if self.ifLog: self.myLogger.debug("%s called with : var1 - %s, var2 - %s"%(logicMethodName,var1,var2))

        or2Success = min(var1 + var2, 1)

        if onlyConstrains:
            or2Loss = 1 - or2Success # max(1 - var1 - var2, 0)
           
            return or2Loss
        else:
            return or2Success
    
    def orVar(self, m, *var, onlyConstrains = False):
        methodName = "orVar"
        logicMethodName = "OR"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))
        
        N = len(var)
        
        varSum = 0
        for currentVar in var:            
            varSum += currentVar
           
        orSuccess = min(varSum, 1)

        if onlyConstrains:
            orLoss = 1 - orSuccess # max(N - varSum, 0)
                
            return orLoss
        else:            
            return orSuccess
            
    def nand2Var(self, m, var1, var2, onlyConstrains = False):
        methodName = "nand2Var"
        logicMethodName = "NAND"
        
        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
        
        # nand(var1, var2) = not(and(var1, var2))
        nand2Success = self.notVar(m, self.and2Var(m, var1, var2)) #  1 - max(var1 + var2 - 1, 0)

        if onlyConstrains:
            nand2Loss = 1 - nand2Success # max(var1 + var2 - 1, 0)
                        
            return nand2Loss
        else:
            return nand2Success
    
    def nandVar(self, m, *var, onlyConstrains = False):
        methodName = "nandVar"
        logicMethodName = "NAND"
       
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))
        
        nandLoss = 0
        
        N = len(var)
        
        varSum = 0
        for currentVar in var:
            varSum += currentVar
        
        # nand(var) = not(and(var))
        nandSuccess = self.notVar(m, self.andVar(m, *var)) # 1 - max(varSum - N + 1, 0)

        if onlyConstrains:
            nandLoss = 1 - nandSuccess # max(varSum - N + 1, 0)
                        
            return nandLoss
        else:            
            return nandSuccess
            
    def ifVar(self, m, var1, var2, onlyConstrains = False):
        methodName = "ifVar"
        logicMethodName = "IF"

        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
     
        # if(var1, var2) = or(not(var1), var2)
        ifSuccess = self.or2Var(m, self.notVar(m, var1), var2)  # min(1 - var1 + var2, 1)

        if onlyConstrains:
            ifLoss = 1 - ifSuccess # min(1 - var1 + var2, 1)
            
            return ifLoss
        else:            
            return ifSuccess
               
    def norVar(self, m, *var, onlyConstrains = False):
        methodName = "norVar"
        logicMethodName = "NOR"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName,var))

        varSum = 0
        for currentVar in var:
            if currentVar > 0.55:
                varSum += currentVar
    
        # nor(var) = not(or(var)
        norSucess = self.notVar(m, self.orVar(m, *var))
        
        if onlyConstrains:
            norLoss = 1- norSucess # max(varSum, 1)
            
            return norLoss
        else:
            return norSucess
        
    def xorVar(self, m, var1, var2, onlyConstrains = False):
        methodName = "xorVar"
        logicMethodName = "XOR"
        
        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))

        # xor(var1, var2) = or(and(var1, not(var2)), and(not(var1), var2))
        xorSuccess = self.or2Var(m, self.and2Var(m, var1, self.notVar(m, var2)), self.and2Var(m, self.notVar(m, var1), var2))
        
        if onlyConstrains:
            xorLoss = 1- xorSuccess # 1 - abs(var1 - var2)
            
            return xorLoss
        else:
            return xorSuccess
    
    def epqVar(self, m, var1, var2, onlyConstrains = False):
        methodName = "epqVar"
        logicMethodName = "EPQ"
        
        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))

        # epq(var1, var2) = and(or(var1, not(var2)), or(not(var1), var2)))
        epqSuccess = self.and2Var(m, self.or2Var(m, var1, self.notVar(m, var2)), self.or2Var(m, self.notVar(m, var1), var2))
        
        if onlyConstrains:
            epqLoss = 1 - epqSuccess # abs(var1 - var2)
            
            return epqLoss
        else:
            return epqSuccess
     
    def countVar(self, m, *var, onlyConstrains = False, limitOp = 'None', limit = 1):
        methodName = "countVar"
        logicMethodName = "COUNT"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName,var))

        varSum = 0
        for currentVar in var:
            varSum += currentVar
            
        countSuccess = 0
            
        if limitOp == '>':     
            countSuccess = min(max(varSum - limit, 0), 1)
            
        elif limitOp == '<':
            countSuccess = min(max(limit - varSum, 0), 1)

        elif limitOp == '=':
            countSuccess = min(max(abs(varSum - limit), 0), 1)
                
        if onlyConstrains:
            countLoss = 1 - countSuccess
    
            return countLoss
        else:
            return countSuccess