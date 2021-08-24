import logging

from regr.solver.ilpBooleanMethods import ilpBooleanProcessor 
from regr.solver.ilpConfig import ilpConfig 

class lcLossBooleanMethods(ilpBooleanProcessor):
    
    def __init__(self, _ildConfig = ilpConfig) -> None:
        super().__init__()
                
        self.myLogger = logging.getLogger(ilpConfig['log_name'])
        self.ifLog =  ilpConfig['ifLog']

    def setTNorm(self, tnorm='L'):
        if tnorm =='L':
            if self.ifLog: self.myLogger.info("Using Lukasiewicz t-norms Formulation")
        elif tnorm =='G':
            if self.ifLog: self.myLogger.info("Using Godel t-norms Formulation")
        elif tnorm =='P':
            if self.ifLog: self.myLogger.info("Using Product t-norms Formulation")
        else:
            raise Exception('Unknown type of t-norms formulation - %s'%(tnorm))

        self.tnorm = tnorm
        
    
    def _isNumber(self, v):
        if v is None:
            return  False
        elif isinstance(v, (int, float, complex)):
            return True
        else:
            return False
        
    def _fixVar(self, var):
        varFixed = []  
        for v in var:
            if not self._isNumber(v):
                varFixed.append(0)
            else:
                varFixed.append(v)
        
        return varFixed
            
    def notVar(self, _, var, onlyConstrains = False, tnome = 'L'):
        methodName = "notVar"
        logicMethodName = "NOT"
                
        if self.ifLog: self.myLogger.debug("%s called with : %s"%(logicMethodName,var))
        
        if not self._isNumber(var):
            var = 0
            
        notSuccess = 1 - var

        if onlyConstrains:
            notLoss = 1 - notSuccess
            
            return notLoss
        else:            
            return notSuccess
    
    def and2Var(self, _, var1, var2, onlyConstrains = False):
        methodName = "and2Var"
        logicMethodName = "AND"
            
        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
        
        if not self._isNumber(var1):
            var1 = 0
            
        if not self._isNumber(var2):
            var2 = 0
            
        if self.tnorm =='L':
            and2Success = max(0, var1 + var2 - 1)
        elif self.tnorm =='G':
            and2Success = min(var1, var2)
        elif self.tnorm =='P':
            and2Success = var1*var2
         
        if onlyConstrains:
            and2Loss = 1 - and2Success
            
            return and2Loss
        else:
            return and2Success
        
    def andVar(self, _, *var, onlyConstrains = False):
        methodName = "andVar"
        logicMethodName = "AND"
            
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))
              
        var = self._fixVar(var)
                
        N = len(var)
        
        varSum = 0
        for currentVar in var:
            varSum += currentVar 
            
        if self.tnorm =='L':
            andSuccess = max(varSum - N + 1, 0)
        elif self.tnorm =='G':
            andSuccess = min(var)
        elif self.tnorm =='P':
            import math
            andSuccess = math.prod(var)

        if onlyConstrains:
            andLoss = 1 - andSuccess
        
            return andLoss       
        else:
            return andSuccess
    
    def or2Var(self, _, var1, var2, onlyConstrains = False):
        methodName = "or2Var"
        logicMethodName = "OR"
       
        if self.ifLog: self.myLogger.debug("%s called with : var1 - %s, var2 - %s"%(logicMethodName,var1,var2))

        if not self._isNumber(var1):
            var1 = 0
            
        if not self._isNumber(var2):
            var2 = 0
            
        if self.tnorm =='L':
            or2Success = min(var1 + var2, 1)
        elif self.tnorm =='G':
            or2Success = max(var1, var2)
        elif self.tnorm =='P':
            or2Success = var1+var2 - var1*var2

        if onlyConstrains:
            or2Loss = 1 - or2Success
           
            return or2Loss
        else:
            return or2Success
    
    def orVar(self, _, *var, onlyConstrains = False):
        methodName = "orVar"
        logicMethodName = "OR"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))
        
        var = self._fixVar(var)
        
        N = len(var)
        
        varSum = 0
        for currentVar in var:            
            varSum += currentVar
           
        if self.tnorm =='L':
            orSuccess = min(varSum, 1)
        elif self.tnorm =='G':
            orSuccess = max(var)
        elif self.tnorm =='P':
            import math
            orSuccess = varSum - math.prod(var)
            
        if onlyConstrains:
            orLoss = 1 - orSuccess
                
            return orLoss
        else:            
            return orSuccess
            
    def nand2Var(self, _, var1, var2, onlyConstrains = False):
        methodName = "nand2Var"
        logicMethodName = "NAND"
        
        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
        
        if not self._isNumber(var1):
            var1 = 0
            
        if not self._isNumber(var2):
            var2 = 0
            
        # nand(var1, var2) = not(and(var1, var2))
        nand2Success = self.notVar(_, self.and2Var(_, var1, var2))

        if onlyConstrains:
            nand2Loss = 1 - nand2Success
                        
            return nand2Loss
        else:
            return nand2Success
    
    def nandVar(self, _, *var, onlyConstrains = False):
        methodName = "nandVar"
        logicMethodName = "NAND"
       
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))
        
        var = self._fixVar(var)
                
        nandLoss = 0
        
        N = len(var)
        
        varSum = 0
        for currentVar in var:
            varSum += currentVar
        
        # nand(var) = not(and(var))
        nandSuccess = self.notVar(_, self.andVar(_, *var))

        if onlyConstrains:
            nandLoss = 1 - nandSuccess
                        
            return nandLoss
        else:            
            return nandSuccess
            
    def ifVar(self, _, var1, var2, onlyConstrains = False):
        methodName = "ifVar"
        logicMethodName = "IF"

        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
     
        if not self._isNumber(var1):
            var1 = 0
            
        if not self._isNumber(var2):
            var2 = 0
          
        if self.tnorm =='L':
            ifSuccess = min(1, 1 - var1 + var2)
        elif self.tnorm =='G':
            if var2 > var1: 
                ifSuccess = 1
            else: 
                ifSuccess = var2
        elif self.tnorm =='P':
            if var1 != 0:
                ifSuccess = min(1, var2/var1)
            else:
                ifSuccess = 1

              
        # if(var1, var2) = or(not(var1), var2)
        #ifSuccess = self.or2Var(_, self.notVar(_, var1), var2)

        if onlyConstrains:
            ifLoss = 1 - ifSuccess
            
            return ifLoss
        else:            
            return ifSuccess
               
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

        if not self._isNumber(var1):
            var1 = 0
            
        if not self._isNumber(var2):
            var2 = 0
            
        # xor(var1, var2) = or(and(var1, not(var2)), and(not(var1), var2))
        xorSuccess = self.or2Var(_, self.and2Var(_, var1, self.notVar(_, var2)), self.and2Var(_, self.notVar(_, var1), var2))
        
        if onlyConstrains:
            xorLoss = 1 - xorSuccess
            
            return xorLoss
        else:
            return xorSuccess
    
    def epqVar(self, _, var1, var2, onlyConstrains = False):
        methodName = "epqVar"
        logicMethodName = "EPQ"
        
        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))

        if not self._isNumber(var1):
            var1 = 0
            
        if not self._isNumber(var2):
            var2 = 0
            
        # epq(var1, var2) = and(or(var1, not(var2)), or(not(var1), var2)))
        epqSuccess = self.and2Var(_, self.or2Var(_, var1, self.notVar(_, var2)), self.or2Var(_, self.notVar(_, var1), var2))
        
        if onlyConstrains:
            epqLoss = 1 - epqSuccess
            
            return epqLoss
        else:
            return epqSuccess
     
    def countVar(self, _, *var, onlyConstrains = False, limitOp = '=', limit = 1, logicMethodName = "COUNT"):
        methodName = "countVar"
        logicMethodName = "COUNT"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName,var))

        var = self._fixVar(var)
                
        varSum = 0
        for currentVar in var:
            varSum += currentVar
            
        countSuccess = 0
            
        if limitOp == '>': # > limit
            countSuccess = min(max(varSum - limit, 0), 1)
            
        elif limitOp == '<': # < limit
            countSuccess = min(max(limit - varSum, 0), 1)

        elif limitOp == '=': # == limit
            countSuccess = min(max(abs(varSum - limit), 0), 1)
                
        if onlyConstrains:
            countLoss = 1 - countSuccess
    
            return countLoss
        else:
            return countSuccess