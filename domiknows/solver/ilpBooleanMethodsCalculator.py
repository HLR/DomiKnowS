import logging

import torch
from domiknows.solver.ilpBooleanMethods import ilpBooleanProcessor 
from domiknows.solver.ilpConfig import ilpConfig 

class booleanMethodsCalculator(ilpBooleanProcessor):
    
    def __init__(self, _ildConfig = ilpConfig) -> None:
        super().__init__()
                
        self.grad = False
        
        self.myLogger = logging.getLogger(ilpConfig['log_name'])
        self.ifLog =  ilpConfig['ifLog']
        
    def notVar(self, _, var, onlyConstrains = False):
        # -- Consider None
        if var is None:
            var = 0 # when None
        # --
        
        notSuccess = 1 - var
        
        return notSuccess
    
    def and2Var(self, _, var1, var2, onlyConstrains = False):
        and2Success = 0
        if var1 + var2 == 2:
            and2Success = 1
            
        return and2Success    
            
    def andVar(self, _, *var, onlyConstrains = False):
        # -- Consider None
        varFixed = []  
        for v in var:
            if v is None:
                tOnes = torch.ones(1, device=self.current_device, requires_grad=False)
                tOnesSqueezed = torch.squeeze(tOnes)
                varFixed.append(tOnesSqueezed) # when None
            else:
                varFixed.append(v)
        var = varFixed
        # --
        
        andSuccess = 0
        if sum(var) == len(var):
            andSuccess = 1
            
        return andSuccess    
    
    def or2Var(self, _, var1, var2, onlyConstrains = False):
        or2Success = 0
        if var1 + var2 > 0:
            or2Success = 1
            
        return or2Success   
    
    def orVar(self, _, *var, onlyConstrains = False):
        # -- Consider None
        varFixed = []  
        for v in var:
            if v is None:
                tZeros = torch.zeros(1, device=self.current_device, requires_grad=False)
                tZerosSqueezed = torch.squeeze(tZeros)
                varFixed.append(tZerosSqueezed)  # when None
            else:
                varFixed.append(v)
        var = varFixed
        # --
        
        orSuccess = 0
        if sum(var) > 0:
            orSuccess = 1
            
        return orSuccess    
            
    def nand2Var(self, _, var1, var2, onlyConstrains = False):
        #results = self.notVar(_, self.and2Var(_, var1, var2,))
        
        nand2Success = 1
        if var1 + var2 == 2:
            nand2Success = 0
            
        return nand2Success         
         
    def nandVar(self, _, *var, onlyConstrains = False):
        #results = self.notVar(_, self.andVar(_, var))
        
        # -- Consider None
        varFixed = []  
        for v in var:
            if v is None:
                tZeros = torch.zeros(1, device=self.current_device, requires_grad=False)
                tZerosSqueezed = torch.squeeze(tZeros)
                varFixed.append(tZerosSqueezed)  # when None
            else:
                varFixed.append(v)
        
        var = varFixed
        # --
            
        nandSuccess = 1
        if sum(var) == len(var):
            nandSuccess = 0
            
        return nandSuccess     
        
    def ifVar(self, _, var1, var2, onlyConstrains = False):
        #results = self.and2Var(_, self.andVar(_, var1), var2)
        
        # -- Consider None
        if var1 is None: # antecedent 
            var1 = 1 # when None

        if var2 is None: # consequent
            var2 = 0 # when None
        # --
        
        ifSuccess = 1
        if var1 - var2 == 1:
            ifSuccess = 0
            
        return ifSuccess 
    
    def norVar(self, _, *var, onlyConstrains = False):
        #results = self.notVar(_, self.orVar(_, var))
        # -- Consider None
        varFixed = []  
        for v in var:
            if v is None:
                tOnes = torch.ones(1, device=self.current_device, requires_grad=False)
                tOnesSqueezed = torch.squeeze(tOnes)
                varFixed.append(tOnesSqueezed) # when None
            else:
                varFixed.append(v)
        var = varFixed
        # --
         
        norSuccess = 1
        if sum(var) > 0:
            norSuccess = 0
            
        return norSuccess
           
    def xorVar(self, _, var1, var2, onlyConstrains = False):
        xorSuccess = 0
        if var1 + var2 == 1:
            xorSuccess = 1
            
        return xorSuccess
    
    def epqVar(self, _, var1, var2, onlyConstrains = False):
        # -- Consider None
        if var1 is None:
            var1 = 0
            
        if var2 is None:
            var2 = 0
        # --
        
        epqSuccess = 0
        if var1 - var2 == 0:
            epqSuccess = 1
            
        return epqSuccess     
    
    def countVar(self, _, *var, onlyConstrains = False, limitOp = '==', limit = 1, logicMethodName = "COUNT"):
        # -- Consider None
        varFixed = []  
        for v in var:
            if v is None:
                tZeros = torch.zeros(1, device=self.current_device, requires_grad=False)
                tZerosSqueezed = torch.squeeze(tZeros)
                varFixed.append(tZerosSqueezed)  # when None
            else:
                varFixed.append(v)
        
        var = varFixed
        # --
        
        countSuccess = 0

        varSum = sum(list(var)).item()

        if limitOp == '>=':
            if varSum >= limit:
                countSuccess = 1
        elif limitOp == '<=':
            if varSum <= limit:
                countSuccess = 1
        elif limitOp == '==':
            if varSum == limit:
                countSuccess = 1
                
        return countSuccess    
        
    def fixedVar(self, _, _var, onlyConstrains = False):
        fixedSuccess = 1
        
        return fixedSuccess