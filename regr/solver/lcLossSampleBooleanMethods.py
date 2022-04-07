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

        self.myLogger = logging.getLogger(ilpConfig['log_name'])
        self.ifLog = ilpConfig['ifLog']
    
    def ifNone(self, var):
        for v in var:
            if not torch.is_tensor(v):
                return True
        
        return False
    
    def notVar(self, _, var, onlyConstrains = False):
        if self.ifNone([var]):
            return None
        
        if onlyConstrains:
            return var # notLoss
        else:
            notSuccess = torch.logical_not(var)
            return notSuccess
    
    def and2Var(self, _, var1, var2, onlyConstrains = False):
        if self.ifNone([var1, var2]):
            return None
        
        and2Success = torch.logical_and(var1,var2)
       
        if onlyConstrains:
            and2Loss = torch.logical_not(and2Success)
            
            return and2Loss
        else:            
            return and2Success    
            
    def andVar(self, _, *var, onlyConstrains = False): 
        if self.ifNone(var):
            return None
        
        andSuccess = var[0]
        for i in range(1, len(var)):
            andSuccess = torch.mul(andSuccess.float(), var[i].float()) #torch.logical_and(andSuccess.float(), var[i].float())
            
        if onlyConstrains:
            andLoss = torch.logical_not(andSuccess)
            
            return andLoss
        else:            
            return andSuccess    
    
    def or2Var(self, _, var1, var2, onlyConstrains = False):
        if self.ifNone([var1, var2]):
            return None
        
        or2Success = torch.logical_or(var1,var2)
            
        if onlyConstrains:
            or2Loss = torch.logical_not(or2Success)
            
            return or2Loss
        else:            
            return or2Success   
    
    def orVar(self, _, *var, onlyConstrains = False):
        if self.ifNone(var):
            return None
        
        orSuccess = var[0]
        for i in range(1, len(var)):
            orSuccess = torch.logical_or(orSuccess, var[i])
            
        if onlyConstrains:
            orLoss = torch.logical_not(orSuccess)
            
            return orLoss
        else:            
            return orSuccess    
            
    def nand2Var(self, _, var1, var2, onlyConstrains = False):
        #results = self.notVar(_, self.and2Var(_, var1, var2,))
        if self.ifNone([var1, var2]):
            return None
        
        nand2Success = torch.logical_not(torch.logical_and(var1, var2))

        if onlyConstrains:
            nand2Loss =  torch.logical_not(nand2Success)
            
            return nand2Loss
        else:            
            return nand2Success         
         
    def nandVar(self, _, *var, onlyConstrains = False):
        #results = self.notVar(_, self.andVar(_, var))
        if self.ifNone(var):
            return None
            
        nandSuccess = var[0]
        for i in range(1, len(var)):
            nandSuccess = torch.logical_and(nandSuccess, var[i])
            
        # nand is reversed to and
        if onlyConstrains:
            nandLoss = nandSuccess
            return nandLoss
        else:            
            nandSuccess = torch.logical_not(nandSuccess)
            return nandSuccess     
        
    def ifVar(self, _, var1, var2, onlyConstrains = False):
        if self.ifNone([var1, var2]):
            return None
                
        notVar2 = torch.sub(torch.ones(len(var2), device=var2.device), var2.float())

        andV = torch.mul(var1, notVar2)
        ifSuccess =  torch.sub(torch.ones(len(andV), device=andV.device), andV)
        #with torch.no_grad():
        #   ifSuccessO = torch.logical_or(torch.logical_not(var1), var2)
    
        if onlyConstrains:
            ifLoss = torch.sub(torch.ones(len(ifSuccess), device=ifSuccess.device), ifSuccess) #torch.logical_not(ifSuccess)
            
            return ifLoss
        else:            
            return ifSuccess 
    
    def norVar(self, _, *var, onlyConstrains = False):
        #results = self.notVar(_, self.orVar(_, var))
        if self.ifNone(var):
            return None
            
        norSuccess = var[0]
        for i in range(1, len(var)):
            norSuccess = torch.logical_or(norSuccess, var[i])
            
        # nor is reversed to or
        if onlyConstrains:
            norLoss = norSuccess
            return norLoss
        else:            
            norSuccess = torch.logical_not(norSuccess)
            return norSuccess     
           
    def xorVar(self, _, var1, var2, onlyConstrains = False):
        if self.ifNone([var1, var2]):
            return None
        
        xorSuccess = torch.logical_or(torch.logical_and(torch.logical_not(var1), var2), torch.logical_and(var1, torch.logical_not((var2))))
            
        if onlyConstrains:
            xorLoss = torch.logical_not(xorSuccess)
            
            return xorLoss
        else:            
            return xorSuccess
    
    def epqVar(self, _, var1, var2, onlyConstrains = False):
        if self.ifNone([var1, var2]):
            return None
        
        epqSuccess = torch.eq(var1,var2)
            
        if onlyConstrains:
            epqLoss = torch.logical_not(epqSuccess)
            
            return epqLoss
        else:       
            return epqSuccess     
    
    def countVar(self, _, *var, onlyConstrains = False, limitOp = '==', limit = 1, logicMethodName = "COUNT"):
        if self.ifNone(var):
            return None
        
        limitTensor = torch.full([len(var[0])], limit, device = var[0].device)
       
        # Calculate sum 
        varSum = torch.clone(var[0])
        for i in range(1, len(var)):
            varSum = torch.add(varSum, var[i])

        # Check condition
        if limitOp == '>=':
            #if varSum >= limit:
            countSuccess = torch.ge(varSum, limitTensor)
        elif limitOp == '<=':
            #if varSum <= limit:
            countSuccess = torch.sub(limitTensor, varSum)
            countSuccess = torch.add(torch.ones(len(countSuccess), device=countSuccess.device), countSuccess)
            countSuccess = torch.clamp(countSuccess, min=0, max=1)

            #countSuccess = torch.le(varSum, limitTensor)
        elif limitOp == '==':
            #if varSum == limit:
            countSuccess = torch.eq(varSum, limitTensor)
                
        if onlyConstrains:
            countLoss = torch.logical_not(countSuccess)
            
            return countLoss
        else:       
            return countSuccess    
        
    def fixedVar(self, _, var, onlyConstrains = False):
        if self.ifNone([var]):
            return None
        
        fixedSuccess = 1
        
        if onlyConstrains:
            fixedLoss = 1 - fixedSuccess
    
            return fixedLoss
        else:
            return fixedSuccess