import logging

import torch

from regr.solver.ilpBooleanMethods import ilpBooleanProcessor 
from regr.solver.lcLossBooleanMethods import lcLossBooleanMethods

from regr.solver.ilpConfig import ilpConfig 

class booleanMethods(ilpBooleanProcessor):
    def notVar(self, _, var, onlyConstrains = False):
        notSuccess = 1 - var

        if onlyConstrains:
            notLoss = 1 - notSuccess
            
            return notLoss
        else:            
            return notSuccess
    
    def and2Var(self, _, var1, var2, onlyConstrains = False):
        and2Success = 0
        if var1 + var2 == 2:
            and2Success = 1
            
        if onlyConstrains:
            and2Loss = 1 - and2Success
            
            return and2Loss
        else:            
            return and2Success    
            
    def andVar(self, _, *var, onlyConstrains = False):
        andSuccess = 0
        if sum(var) == len(var):
            andSuccess = 1
            
        if onlyConstrains:
            andLoss = 1 - andSuccess
            
            return andLoss
        else:            
            return andSuccess    
    
    def or2Var(self, _, var1, var2, onlyConstrains = False):
        or2Success = 0
        if var1 + var2 > 0:
            or2Success = 1
            
        if onlyConstrains:
            or2Loss = 1 - or2Success
            
            return or2Loss
        else:            
            return or2Success   
    
    def orVar(self, _, *var, onlyConstrains = False):
        orSuccess = 0
        if sum(var) > 0:
            orSuccess = 1
            
        if onlyConstrains:
            orLoss = 1 - orSuccess
            
            return orLoss
        else:            
            return orSuccess    
            
    def nand2Var(self, _, var1, var2, onlyConstrains = False):
        #results = self.notVar(_, self.and2Var(_, var1, var2,))
        
        nand2Success = 1
        if var1 + var2 == 2:
            nand2Success = 0
            
        if onlyConstrains:
            nand2Loss = 1 - nand2Success
            
            return nand2Loss
        else:            
            return nand2Success         
         
    def nandVar(self, _, *var, onlyConstrains = False):
        #results = self.notVar(_, self.andVar(_, var))
            
        nandSuccess = 1
        if sum(var) == len(var):
            nandSuccess = 0
            
        if onlyConstrains:
            nandLoss = 1 - nandSuccess
            
            return nandLoss
        else:            
            return nandSuccess     
        
    def ifVar(self, _, var1, var2, onlyConstrains = False):
        #results = self.and2Var(_, self.andVar(_, var1), var2)
        
        ifSuccess = 1
        if var1 - var2 == 1:
            ifSuccess = 0
            
        if onlyConstrains:
            ifLoss = 1 - ifSuccess
            
            return ifLoss
        else:            
            return ifSuccess 
    
    def norVar(self, _, *var, onlyConstrains = False):
        #results = self.notVar(_, self.orVar(_, var))
            
        norSuccess = 1
        if sum(var) > 0:
            norSuccess = 0
            
        if onlyConstrains:
            norLoss = 1 - norSuccess
            
            return norLoss
        else:            
            return norSuccess
           
    def xorVar(self, _, var1, var2, onlyConstrains = False):
        xorSuccess = 0
        if var1 + var2 == 1:
            xorSuccess = 1
            
        if onlyConstrains:
            xorLoss = 1 - xorSuccess
            
            return xorLoss
        else:            
            return xorSuccess
    
    def epqVar(self, _, var1, var2, onlyConstrains = False):
        epqSuccess = 0
        if var1 - var2 == 0:
            epqSuccess = 1
            
        if onlyConstrains:
            epqLoss = 1 - epqSuccess
            
            return epqLoss
        else:       
            return epqSuccess     
    
    def countVar(self, _, *var, onlyConstrains = False, limitOp = '==', limit = 1, logicMethodName = "COUNT"):
        var = var[0]
        countSuccess = 0

        varSum = sum(list(var))

        if limitOp == '>=':
            if varSum >= limit:
                countSuccess = 1
        elif limitOp == '<=':
            if varSum <= limit:
                countSuccess = 1
        elif limitOp == '==':
            if varSum == limit:
                countSuccess = 1
                
        if onlyConstrains:
            countLoss = 1 - countSuccess
            
            return countLoss
        else:       
            return countSuccess    
        
    def fixedVar(self, _, _var, onlyConstrains = False):
        fixedSuccess = 1
        
        if onlyConstrains:
            fixedLoss = 1 - fixedSuccess
    
            return fixedLoss
        else:
            return fixedSuccess
        
class lcLossSampleBooleanMethods(ilpBooleanProcessor):
    def __init__(self, _ildConfig = ilpConfig) -> None:
        super().__init__()
        self.myBooleanMethods = booleanMethods() # lcLossBooleanMethods() # booleanMethods()

        self.myLogger = logging.getLogger(ilpConfig['log_name'])
        self.ifLog = ilpConfig['ifLog']
    
    def logicMany(self, _, var, onlyConstrains = False, logicMethod=None):
        for currentVar in var:
            if currentVar is None:
                results = None
                return results
            
        sampleSize = len(var[0])
        results = torch.zeros(sampleSize, dtype=torch.int)
        
        for i in range(sampleSize-1):
            _var = []
            for v in var:
                currentV = v[i].item()
                _var.append(currentV)
                
            results[i] = logicMethod(_, *_var, onlyConstrains = onlyConstrains)
            
        return results
    
    def notVar(self, _, var, onlyConstrains = False):
        logicMethodName = "NOT"
                
        if self.ifLog: self.myLogger.debug("%s is called with samples"%(logicMethodName))
        
        return self.logicMany(_, [var], onlyConstrains = onlyConstrains, logicMethod = self.myBooleanMethods.notVar)
        
    def and2Var(self, _, var1, var2, onlyConstrains = False):
        logicMethodName = "AND"
            
        if self.ifLog: self.myLogger.debug("%s is called with samples"%(logicMethodName))
        
        return self.logicMany(_, [var1, var2], onlyConstrains = onlyConstrains, logicMethod = self.myBooleanMethods.and2Var)
        
    def andVar(self, _, *var, onlyConstrains = False):
        logicMethodName = "AND"
            
        if self.ifLog: self.myLogger.debug("%s is called with samples"%(logicMethodName))
              
        return self.logicMany(_, var, onlyConstrains = onlyConstrains, logicMethod = self.myBooleanMethods.andVar)
    
    def or2Var(self, _, var1, var2, onlyConstrains = False):
        logicMethodName = "OR"
       
        if self.ifLog: self.myLogger.debug("%s is called with samples"%(logicMethodName))

        return self.logicMany(_, [var1, var2], onlyConstrains = onlyConstrains, logicMethod = self.myBooleanMethods.or2Var)
    
    def orVar(self, _, *var, onlyConstrains = False):
        logicMethodName = "OR"
        
        if self.ifLog: self.myLogger.debug("%s is called with samples"%(logicMethodName))
        
        return self.logicMany(_, var, onlyConstrains = onlyConstrains, logicMethod = self.myBooleanMethods.orVar)
            
    def nand2Var(self, _, var1, var2, onlyConstrains = False):
        logicMethodName = "NAND"
        
        if self.ifLog: self.myLogger.debug("%s is called with samples"%(logicMethodName))
        
        return self.logicMany(_, [var1, var2], onlyConstrains = onlyConstrains, logicMethod = self.myBooleanMethods.nand2Var)
    
    def nandVar(self, _, *var, onlyConstrains = False):
        logicMethodName = "NAND"
       
        if self.ifLog: self.myLogger.debug("%s is called with samples"%(logicMethodName))

        return self.logicMany(_, var, onlyConstrains = onlyConstrains, logicMethod = self.myBooleanMethods.nandVar)
            
    def ifVar(self, _, var1, var2, onlyConstrains = False):
        logicMethodName = "IF"

        if self.ifLog: self.myLogger.debug("%s is called with samples"%(logicMethodName))
     
        return self.logicMany(_, [var1, var2], onlyConstrains = onlyConstrains, logicMethod = self.myBooleanMethods.ifVar)

    def norVar(self, _, *var, onlyConstrains = False):
        logicMethodName = "NOR"
        
        if self.ifLog: self.myLogger.debug("%s is called with samples"%(logicMethodName))

        return self.logicMany(_, var, onlyConstrains = onlyConstrains, logicMethod = self.myBooleanMethods.norVar)
        
    def xorVar(self, _, var1, var2, onlyConstrains = False):
        logicMethodName = "XOR"
        
        if self.ifLog: self.myLogger.debug("%s is called with samples"%(logicMethodName))

        return self.logicMany(_, [var1, var2], onlyConstrains = onlyConstrains, logicMethod = self.myBooleanMethods.xorVar)
    
    def epqVar(self, _, var1, var2, onlyConstrains = False):
        logicMethodName = "EPQ"
        
        if self.ifLog: self.myLogger.debug("%s is called with samples"%(logicMethodName))

        return self.logicMany(_, [var1, var2], onlyConstrains = onlyConstrains, logicMethod = self.myBooleanMethods.epqVar)
     
    def countVar(self, _, *var, onlyConstrains = False, limitOp = '==', limit = 1, logicMethodName = "COUNT"):
        logicMethodName = "COUNT"
        
        if self.ifLog: self.myLogger.debug("%s is called with samples"%(logicMethodName))

        countFun = lambda _, *var, onlyConstrains=onlyConstrains : self.myBooleanMethods.countVar(_, var, onlyConstrains=onlyConstrains, limitOp=limitOp, limit=limit, logicMethodName = "COUNT")
        return self.logicMany(_, var, onlyConstrains = onlyConstrains, logicMethod = countFun)
    
    def fixedVar(self, _, *var, onlyConstrains = False):
        logicMethodName = "FIXED"
        
        if self.ifLog: self.myLogger.debug("%s is called with samples"%(logicMethodName))
        
        # Search for tensor size
        tensorLen = 0
        for v in var:
            if v is not None:
                tensorLen = len(v)
                break
            
        var_update = []
        for v in var:
            if v == None:
                var_update.append(torch.ones(tensorLen))
            else:
                var_update.append(v)


        return self.logicMany(_, var_update, onlyConstrains = onlyConstrains, logicMethod = self.myBooleanMethods.fixedVar)
        