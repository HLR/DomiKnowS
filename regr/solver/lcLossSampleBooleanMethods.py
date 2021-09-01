import logging

import torch

from regr.solver.ilpBooleanMethods import ilpBooleanProcessor 
from regr.solver.lcLossBooleanMethods import lcLossBooleanMethods

from regr.solver.ilpConfig import ilpConfig 

class lcLossSampleBooleanMethods(ilpBooleanProcessor):
    
    def __init__(self, _ildConfig = ilpConfig) -> None:
        super().__init__()
        self.myLcLossBooleanMethods = lcLossBooleanMethods()

        self.myLogger = logging.getLogger(ilpConfig['log_name'])
        self.ifLog =  ilpConfig['ifLog']

    def setTNorm(self, tnorm='L'):
        if tnorm =='L':
            self.myLcLossBooleanMethods.tnorm(tnorm)
            if self.ifLog: self.myLogger.info("Using Lukasiewicz t-norms Formulation")
        elif tnorm =='G':
            self.myLcLossBooleanMethods.tnorm(tnorm)
            if self.ifLog: self.myLogger.info("Using Godel t-norms Formulation")
        elif tnorm =='P':
            self.myLcLossBooleanMethods.tnorm(tnorm)
            if self.ifLog: self.myLogger.info("Using Product t-norms Formulation")
        else:
            raise Exception('Unknown type of t-norms formulation - %s'%(tnorm))

        self.tnorm = tnorm
        
    def notVar(self, _, var, onlyConstrains = False, tnome = 'L'):
        methodName = "notVar"
        logicMethodName = "NOT"
                
        if self.ifLog: self.myLogger.debug("%s called with : %s"%(logicMethodName,var))
        
        if not torch.is_tensor(var):
            raise Exception("Provided variable is not tensor %s"%(type(var)))
            
        results = torch.as_tensor(var)
        
        for i, v in enumerate(var):
            results[i] = self.myLcLossBooleanMethods.notVar(_, v.item(), onlyConstrains = onlyConstrains)

        return results
    
    def logic2(self, _, var1, var2, onlyConstrains = False, logicMethod=None):
        if not torch.is_tensor(var1):
            raise Exception("Provided variable is not tensor %s"%(type(var1)))
        
        if not torch.is_tensor(var2):
            raise Exception("Provided variable is not tensor %s"%(type(var2)))
        
        results = torch.as_tensor(var1)
        
        for i, v in enumerate(var1):
            results[i] = logicMethod(_, var1[i].item(), var2[i].item(), onlyConstrains = onlyConstrains)
        
        return results
    
    def logicMany(self, _, var, onlyConstrains = False, logicMethod=None):
        for currentVar in var:
            if not torch.is_tensor(currentVar):
                raise Exception("Provided variable is not tensor %s"%(type(currentVar)))
                
        results = torch.as_tensor(var[0])
        
        for i, v in enumerate(var[0]):
            _var = []
            
            for v in var:
                _var.append(v[i].item())
            
            results[i] = logicMethod(_, *_var, onlyConstrains = onlyConstrains)
        
        return results
    
    def and2Var(self, _, var1, var2, onlyConstrains = False):
        methodName = "and2Var"
        logicMethodName = "AND"
            
        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
        
        return self.logic2(_, var1, var2, onlyConstrains, self.myLcLossBooleanMethods.and2Var)
        
    def andVar(self, _, *var, onlyConstrains = False):
        methodName = "andVar"
        logicMethodName = "AND"
            
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))
              
        return self.logicMany(_, var, onlyConstrains, self.myLcLossBooleanMethods.andVar)
    
    def or2Var(self, _, var1, var2, onlyConstrains = False):
        methodName = "or2Var"
        logicMethodName = "OR"
       
        if self.ifLog: self.myLogger.debug("%s called with : var1 - %s, var2 - %s"%(logicMethodName,var1,var2))

        return self.logic2(_, var1, var2, onlyConstrains, self.myLcLossBooleanMethods.or2Var)
    
    def orVar(self, _, *var, onlyConstrains = False):
        methodName = "orVar"
        logicMethodName = "OR"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))
        
        return self.logicMany(_, var, onlyConstrains, self.myLcLossBooleanMethods.ordVar)
            
    def nand2Var(self, _, var1, var2, onlyConstrains = False):
        methodName = "nand2Var"
        logicMethodName = "NAND"
        
        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
        
        return self.logic2(_, var1, var2, onlyConstrains, self.myLcLossBooleanMethods.nand2Var)

    
    def nandVar(self, _, *var, onlyConstrains = False):
        methodName = "nandVar"
        logicMethodName = "NAND"
       
        return self.logicMany(_, var, onlyConstrains, self.myLcLossBooleanMethods.nandVar)
            
    def ifVar(self, _, var1, var2, onlyConstrains = False):
        methodName = "ifVar"
        logicMethodName = "IF"

        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
     
        return self.logic2(_, var1, var2, onlyConstrains, self.myLcLossBooleanMethods.ifVar)

    def norVar(self, _, *var, onlyConstrains = False):
        methodName = "norVar"
        logicMethodName = "NOR"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName,var))

        var = self._fixVar(var)
                
        varSum = 0
        for currentVar in var:
            varSum += currentVar
    
        # nor(var) = not(or(var)
        norSucess = self.notVar(_, self.orVar(_, *var))
        
        if onlyConstrains:
            norLoss = 1 - norSucess
            
            return norLoss
        else:
            return norSucess
        
    def xorVar(self, _, var1, var2, onlyConstrains = False):
        methodName = "xorVar"
        logicMethodName = "XOR"
        
        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))

        return self.logic2(_, var1, var2, onlyConstrains, self.myLcLossBooleanMethods.xorVar)
    
    def epqVar(self, _, var1, var2, onlyConstrains = False):
        methodName = "epqVar"
        logicMethodName = "EPQ"
        
        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))

        return self.logic2(_, var1, var2, onlyConstrains, self.myLcLossBooleanMethods.epqVar)
     
    def countVar(self, _, *var, onlyConstrains = False, limitOp = '=', limit = 1, logicMethodName = "COUNT"):
        methodName = "countVar"
        logicMethodName = "COUNT"
        
        return self.logicMany(_, var, onlyConstrains, self.myLcLossBooleanMethods.countVar)