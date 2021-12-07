import logging

import torch

from regr.solver.ilpBooleanMethods import ilpBooleanProcessor 

from regr.solver.ilpConfig import ilpConfig 

class booleanMethods(ilpBooleanProcessor):
    def notVar(self, _, var, onlyConstrains = False):
        results = 1 - var
        
        return results
    
    def and2Var(self, _, var1, var2, onlyConstrains = False):
        results = 0
        if var1 + var2 == 2:
            results = 1
            
        return results
        
    def andVar(self, _, *var, onlyConstrains = False):
        results = 0
        if sum(var) == len(var):
            results = 1
            
        return results
    
    def or2Var(self, _, var1, var2, onlyConstrains = False):
        results = 0
        if var1 + var2 > 0:
            results = 1
            
        return results
    
    def orVar(self, _, *var, onlyConstrains = False):
        results = 0
        if sum(var) > 0:
            results = 1
            
        return results
            
    def nand2Var(self, _, var1, var2, onlyConstrains = False):
        #results = self.notVar(_, self.and2Var(_, var1, var2,))
            
        results = 1
        if var1 + var2 == 2:
            results = 0
            
        return results        
    def nandVar(self, _, *var, onlyConstrains = False):
        #results = self.notVar(_, self.andVar(_, var))
            
        results = 1
        if sum(var) == len(var):
            results = 0
            
        return results 
       
    def ifVar(self, _, var1, var2, onlyConstrains = False):
        #results = self.and2Var(_, self.andVar(_, var1), var2)
        results = 1
        if var1 - var2 == 1:
            results = 0
            
        return results
    
    def norVar(self, _, *var, onlyConstrains = False):
        #results = self.notVar(_, self.orVar(_, var))
            
        results = 1
        if sum(var) > 0:
            results = 0
            
        return results     
  
    def xorVar(self, _, var1, var2, onlyConstrains = False):
        results = 0
        if var1 + var2 == 1:
            results = 1
            
        return results
    
    def epqVar(self, _, var1, var2, onlyConstrains = False):
        results = 0
        if var1 - var2 == 0:
            results = 1
            
        return results
    
    def countVar(self, _, *var, onlyConstrains = False, limitOp = '==', limit = 1, logicMethodName = "COUNT"):
        var = var[0]
        results = 0

        if limitOp == '>=':
            if sum(var) >= limit:
                results = 1
        if limitOp == '<=':
            if sum(var) <= limit:
                results = 1
        if limitOp == '==':
            if sum(var) == limit:
                results = 1
                
        return results
        
class lcLossSampleBooleanMethods(ilpBooleanProcessor):
    def __init__(self, _ildConfig = ilpConfig) -> None:
        super().__init__()
        self.myBooleanMethods = booleanMethods() #gurobiILPBooleanProcessor() #lcLossBooleanMethods()

        self.myLogger = logging.getLogger(ilpConfig['log_name'])
        self.ifLog = ilpConfig['ifLog']
    
    def logicMany(self, _, var, onlyConstrains = False, logicMethod=None):
        for currentVar in var:
            if currentVar is None:
                continue
            if not torch.is_tensor(currentVar):
                raise Exception("Provided variable is not tensor %s"%(type(currentVar)))
                
        if var[0] is None:
            results = None
            return results
        
        results = torch.clone(var[0])
        
        for i, _ in enumerate(var[0]):
            _var = []
            
            if i == 0:
                results[i] = 1
            else:
                for v in var:
                    currentV = v[i].item()
                    if currentV > 0:
                        currentV = 1
                        
                    _var.append(currentV)
                
                results[i] = logicMethod(_, *_var, onlyConstrains = onlyConstrains)
            
            if results[i] == 1:
                for v in var: 
                    if v[i] > 0:
                        results[i] = results[i] * v[i]
                    else:
                        results[i] = results[i] * (1 - v[0])   
                             
        return results
    
    def notVar(self, _, var, onlyConstrains = False):
        logicMethodName = "NOT"
                
        if self.ifLog: self.myLogger.debug("%s called with : %s"%(logicMethodName,var))
        
        return self.logicMany(_, [var], onlyConstrains = onlyConstrains, logicMethod = self.myBooleanMethods.notVar)
        
    def and2Var(self, _, var1, var2, onlyConstrains = False):
        logicMethodName = "AND"
            
        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
        
        return self.logicMany(_, [var1, var2], onlyConstrains = onlyConstrains, logicMethod = self.myBooleanMethods.and2Var)
        
    def andVar(self, _, *var, onlyConstrains = False):
        logicMethodName = "AND"
            
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))
              
        return self.logicMany(_, var, onlyConstrains = onlyConstrains, logicMethod = self.myBooleanMethods.andVar)
    
    def or2Var(self, _, var1, var2, onlyConstrains = False):
        logicMethodName = "OR"
       
        if self.ifLog: self.myLogger.debug("%s called with : var1 - %s, var2 - %s"%(logicMethodName,var1,var2))

        return self.logicMany(_, [var1, var2], onlyConstrains = onlyConstrains, logicMethod = self.myBooleanMethods.or2Var)
    
    def orVar(self, _, *var, onlyConstrains = False):
        logicMethodName = "OR"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))
        
        return self.logicMany(_, var, onlyConstrains = onlyConstrains, logicMethod = self.myBooleanMethods.orVar)
            
    def nand2Var(self, _, var1, var2, onlyConstrains = False):
        logicMethodName = "NAND"
        
        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
        
        return self.logicMany(_, [var1, var2], onlyConstrains = onlyConstrains, logicMethod = self.myBooleanMethods.nand2Var)
    
    def nandVar(self, _, *var, onlyConstrains = False):
        logicMethodName = "NAND"
       
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName,var))

        return self.logicMany(_, var, onlyConstrains = onlyConstrains, logicMethod = self.myBooleanMethods.nandVar)
            
    def ifVar(self, _, var1, var2, onlyConstrains = False):
        logicMethodName = "IF"

        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
     
        return self.logicMany(_, [var1, var2], onlyConstrains = onlyConstrains, logicMethod = self.myBooleanMethods.ifVar)

    def norVar(self, _, *var, onlyConstrains = False):
        logicMethodName = "NOR"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName,var))

        return self.logicMany(_, var, onlyConstrains = onlyConstrains, logicMethod = self.myBooleanMethods.norVar)
        
    def xorVar(self, _, var1, var2, onlyConstrains = False):
        logicMethodName = "XOR"
        
        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))

        return self.logicMany(_, [var1, var2], onlyConstrains = onlyConstrains, logicMethod = self.myBooleanMethods.xorVar)
    
    def epqVar(self, _, var1, var2, onlyConstrains = False):
        logicMethodName = "EPQ"
        
        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))

        return self.logicMany(_, [var1, var2], onlyConstrains = onlyConstrains, logicMethod = self.myBooleanMethods.epqVar)
     
    def countVar(self, _, *var, onlyConstrains = False, limitOp = '==', limit = 1, logicMethodName = "COUNT"):
        logicMethodName = "COUNT"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName,var))

        countFun = lambda _, *var, onlyConstrains=onlyConstrains : self.myBooleanMethods.countVar(_, var, onlyConstrains=onlyConstrains, limitOp=limitOp, limit=limit, logicMethodName = "COUNT")
        return self.logicMany(_, var, onlyConstrains = onlyConstrains, logicMethod = countFun)
    
    def fixedVar(self, _, *var, onlyConstrains = False):
        logicMethodName = "FIXED"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName,var))

        if var[0] == None:
            return None
        else:
            zeros = torch.zeros(len(var))
            return zeros