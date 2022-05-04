import logging
import torch

from regr.graph import DataNode

from regr.solver.ilpBooleanMethods import ilpBooleanProcessor 
from regr.solver.ilpConfig import ilpConfig 

class lcLossBooleanMethods(ilpBooleanProcessor):
    
    def __init__(self, _ildConfig = ilpConfig) -> None:
        super().__init__()
        self.tnorm = DataNode.tnormsDefault
        self.grad = True
        
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
        
    def _isTensor(self, v):
        if v is None:
            return  None
        elif torch.is_tensor(v):
            if len(v.shape) == 0 or len(v.shape) == 1 and v.shape[0] == 1:
                return v.item()
            else:
                None
        elif isinstance(v, (int, float, complex)):
            return None
        else:
            return None
        
    def _fixVar(self, var):
        varFixed = []  
        for v in var:
            if not self._isTensor(v):
                varFixed.append(torch.zeros(1, device=self.current_device, requires_grad=True))
            else:
                varFixed.append(v)
        
        return varFixed
            
    def notVar(self, _, var, onlyConstrains = False):
        methodName = "notVar"
        logicMethodName = "NOT"
                
        if self.ifLog: self.myLogger.debug("%s called with : %s"%(logicMethodName,var))
        
        if not self._isTensor(var):
            var = torch.zeros(1, device=self.current_device, requires_grad=True)
            
        notSuccess = torch.sub(1, var)

        if onlyConstrains:
            notLoss = torch.sub(1, notSuccess)
            
            return notLoss
        else:            
            return notSuccess
    
    def and2Var(self, _, var1, var2, onlyConstrains = False):
        methodName = "and2Var"
        logicMethodName = "AND"
            
        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
        
        if not self._isTensor(var1):
            var1 = 0
            
        if not self._isTensor(var2):
            var2 = 0
            
        if self.tnorm =='L':
            and2Success = torch.maximum(torch.zeros(1, device=self.current_device, requires_grad=True), torch.sub(torch.add(var1, var2), 1))   #max(0, var1 + var2 - 1)
        elif self.tnorm =='G':
            and2Success = torch.minimum(var1, var2) #min(var1, var2)
        elif self.tnorm =='P':
            and2Success = torch.mul(var1, var2) #var1*var2
         
        if onlyConstrains:
            and2Loss = torch.sub(1, and2Success)
            
            return and2Loss
        else:
            return and2Success
        
    def andVar(self, _, *var, onlyConstrains = False):
        methodName = "andVar"
        logicMethodName = "AND"
            
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))
              
        var = self._fixVar(var)
                    
        if self.tnorm =='L':
            N = len(var)
        
            varSum = var[0]
            for v in var[1:]:
                varSum.add(v)
                
            andSuccess = torch.maximum(torch.add(torch.sub(varSum,N), 1), torch.zeros(1, device=self.current_device, requires_grad=True))
        elif self.tnorm =='G':
            andSuccess = var[0]
            for v in var[1:]:
                andSuccess.minimum(v)
        elif self.tnorm =='P':
            andSuccess = var[0]
            for v in var[1:]:
                andSuccess.mul(v)

        if onlyConstrains:
            andLoss = 1 - andSuccess
        
            return andLoss       
        else:
            return andSuccess
    
    def or2Var(self, _, var1, var2, onlyConstrains = False):
        methodName = "or2Var"
        logicMethodName = "OR"
       
        if self.ifLog: self.myLogger.debug("%s called with : var1 - %s, var2 - %s"%(logicMethodName,var1,var2))

        if not self._isTensor(var1):
            var1 = 0
            
        if not self._isTensor(var2):
            var2 = 0
            
        if self.tnorm =='L':
            or2Success = torch.minimum(torch.add(var1, var2), torch.ones(1, device=self.current_device, requires_grad=True))  #min(var1 + var2, 1)
        elif self.tnorm =='G':
            or2Success = torch.maximum(var1, var2)#max(var1, var2)
        elif self.tnorm =='P':
            or2Success = torch.sub(torch.add(var1, var2), torch.mul(var1, var2))  #var1+var2 - var1*var2

        if onlyConstrains:
            or2Loss = torch.sub(1, or2Success) # 1 - or2Success
           
            return or2Loss
        else:
            return or2Success
    
    def orVar(self, _, *var, onlyConstrains = False):
        methodName = "orVar"
        logicMethodName = "OR"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))
        
        var = self._fixVar(var)
        
        varSum = var[0]
        for v in var[1:]:
            varSum.add(v)
           
        if self.tnorm =='L':
            orSuccess = torch.minimum(varSum, torch.ones(1, device=self.current_device, requires_grad=True))  #min(varSum, 1)
        elif self.tnorm =='G':
            orSuccess = var[0]
            for v in var[1:]:
                orSuccess.maximum(v)
        elif self.tnorm =='P':
            varPod = var[0]
            for v in var[1:]:
                varPod.mul(v)
            orSuccess = torch.sub(varSum, varPod)   #varSum - math.prod(var)
            
        if onlyConstrains:
            orLoss = torch.sub(1, orSuccess) #1 - orSuccess
                
            return orLoss
        else:            
            return orSuccess
            
    def nand2Var(self, _, var1, var2, onlyConstrains = False):
        methodName = "nand2Var"
        logicMethodName = "NAND"
        
        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
            
        # nand(var1, var2) = not(and(var1, var2))
        nand2Success = self.notVar(_, self.and2Var(_, var1, var2))

        if onlyConstrains:
            nand2Loss = torch.sub(1, nand2Success)
                        
            return nand2Loss
        else:
            return nand2Success
    
    def nandVar(self, _, *var, onlyConstrains = False):
        methodName = "nandVar"
        logicMethodName = "NAND"
       
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))
        
        # nand(var) = not(and(var))
        nandSuccess = self.notVar(_, self.andVar(_, *var))

        if onlyConstrains:
            nandLoss = torch.sub(1, nandSuccess)
                        
            return nandLoss
        else:            
            return nandSuccess
            
    def ifVar(self, _, var1, var2, onlyConstrains = False):
        methodName = "ifVar"
        logicMethodName = "IF"

        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
     
        var1Item = self._isTensor(var1)
        if var1Item == None:
            var1 = 0
            
        var2Item = self._isTensor(var2)
        if var2Item == None:
            var2 = 0
          
        if self.tnorm =='L':
            ifSuccess = torch.minimum(torch.ones(1, device=self.current_device, requires_grad=True), torch.sub(1, torch.add(var1, var2))) #min(1, 1 - var1 + var2) #torch.sub
        elif self.tnorm =='G':
            if var2Item > var1Item: 
                ifSuccess = torch.mul(var2, torch.div(1, var2)) #1
            else: 
                ifSuccess = var2
        elif self.tnorm =='P':
            if var1Item != 0:
                ifSuccess = torch.minimum(torch.ones(1, device=self.current_device, requires_grad=True), torch.div(var2, var1)) # min(1, var2/var1) # 
            else:
                ifSuccess = torch.mul(var2, torch.div(1, var2)) #1

        # if(var1, var2) = or(not(var1), var2)
        #ifSuccess = self.or2Var(_, self.notVar(_, var1), var2)

        if onlyConstrains:
            ifLoss = torch.sub(1, ifSuccess) # 1 - ifSuccess
            
            return ifLoss
        else:            
            return ifSuccess
               
    def norVar(self, _, *var, onlyConstrains = False):
        methodName = "norVar"
        logicMethodName = "NOR"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName,var))

        # nor(var) = not(or(var)
        norSucess = self.notVar(_, self.orVar(_, *var))
        
        if onlyConstrains:
            norLoss = torch.sub(1, norSucess)
            
            return norLoss
        else:
            return norSucess
        
    def xorVar(self, _, var1, var2, onlyConstrains = False):
        methodName = "xorVar"
        logicMethodName = "XOR"
        
        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))

        # xor(var1, var2) = or(and(var1, not(var2)), and(not(var1), var2))
        xorSuccess = self.or2Var(_, self.and2Var(_, var1, self.notVar(_, var2)), self.and2Var(_, self.notVar(_, var1), var2))
        
        if onlyConstrains:
            xorLoss = torch.sub(1, xorSuccess)
            
            return xorLoss
        else:
            return xorSuccess
    
    def epqVar(self, _, var1, var2, onlyConstrains = False):
        methodName = "epqVar"
        logicMethodName = "EPQ"
        
        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
            
        # epq(var1, var2) = and(or(var1, not(var2)), or(not(var1), var2)))
        epqSuccess = self.and2Var(_, self.or2Var(_, var1, self.notVar(_, var2)), self.or2Var(_, self.notVar(_, var1), var2))
        
        if onlyConstrains:
            epqLoss = torch.sub(1, epqSuccess)
            
            return epqLoss
        else:
            return epqSuccess
     
    def countVar(self, _, *var, onlyConstrains = False, limitOp = '==', limit = 1, logicMethodName = "COUNT"):
        methodName = "countVar"
        logicMethodName = "COUNT"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName,var))

        var = self._fixVar(var)
                
        varSum = var[0]
        for v in var[1:]:
            varSum.add(v)
                        
        if limitOp == '>=': # > limit
            countSuccess = torch.minimum(torch.maximum(torch.sub(varSum, limit), torch.zeros(1, device=self.current_device, requires_grad=True)), torch.ones(1, device=self.current_device, requires_grad=True)) #min(max(varSum - limit, 0), 1)
            
        elif limitOp == '<=': # < limit
            countSuccess = torch.minimum(torch.maximum(torch.sub(limit, varSum), torch.zeros(1, device=self.current_device, requires_grad=True)), torch.ones(1, device=self.current_device, requires_grad=True)) #min(max(limit - varSum, 0), 1)

        elif limitOp == '==': # == limit
            countSuccess = torch.minimum(torch.maximum(torch.abs(torch.sub(limit, varSum)), torch.zeros(1, device=self.current_device, requires_grad=True)), torch.ones(1, device=self.current_device, requires_grad=True)) #min(max(abs(varSum - limit), 0), 1)
                
        if onlyConstrains:
            countLoss = torch.sub(1, countSuccess)
    
            return countLoss
        else:
            return countSuccess
        
    def fixedVar(self, _, _var, onlyConstrains = False):
        logicMethodName = "FIXED"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName,_var))
        
        fixedSuccess = torch.ones(1, device=self.current_device, requires_grad=True)
        
        if onlyConstrains:
            fixedLoss = torch.sub(1,  fixedSuccess)
    
            return fixedLoss
        else:
            return fixedSuccess