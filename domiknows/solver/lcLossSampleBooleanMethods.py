import logging

import torch

from domiknows.solver.ilpBooleanMethods import ilpBooleanProcessor 

from domiknows.solver.ilpConfig import ilpConfig 
   
class lcLossSampleBooleanMethods(ilpBooleanProcessor):
    def __init__(self, _ildConfig = ilpConfig) -> None:
        super().__init__()
        self.grad = False
        
        self.myLogger = logging.getLogger(ilpConfig['log_name'])
        self.ifLog = ilpConfig['ifLog']
    
    # -- Consider None
    def ifNone(self, var): # Used in all except countVar
        for v in var:
            if not torch.is_tensor(v):
                return True
        
        return False
    #--
    
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
            andSuccess = torch.logical_and(andSuccess, var[i])
            
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
                
        ifSuccess = torch.logical_or(torch.logical_not(var1), var2)
    
        if onlyConstrains:
            ifLoss = torch.logical_not(ifSuccess)
            
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
        # -- Consider None
        fixedVar = []
        for v in var:
            if torch.is_tensor(v):
                fixedVar.append(v)
            else:
                if limitOp == '>=':
                    fixedVar.append(torch.zeros([self.sampleSize], device=self.current_device))
                elif limitOp == '<=':
                    fixedVar.append(torch.ones([self.sampleSize], device=self.current_device))
                elif limitOp == '==':
                    fixedVar.append(torch.zeros([self.sampleSize], device=self.current_device))
        # --
         
        limitTensor = torch.full([self.sampleSize], limit, device = self.current_device)
       
        # Calculate sum 
        varSum = torch.zeros([self.sampleSize], device=self.current_device)
        if fixedVar:
            varSum = fixedVar[0].int()
            
        for i in range(1, len(fixedVar)):
            varSum.add_(fixedVar[i].int())

        # Check condition
        if limitOp == '>=':
            #if varSum >= limit:
            countSuccess = torch.ge(varSum, limitTensor)
        elif limitOp == '<=':
            #if varSum <= limit:
            countSuccess = torch.le(varSum, limitTensor)
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