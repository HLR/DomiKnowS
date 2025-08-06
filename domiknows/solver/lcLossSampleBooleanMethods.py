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
           
    def xorVar(self, _, *var, onlyConstrains = False):
        if self.ifNone(var):
            return None
        
        if len(var) == 0:
            # XOR of no variables is False
            return torch.zeros([self.sampleSize], device=self.current_device, dtype=torch.bool)
        elif len(var) == 1:
            # XOR of single variable is the variable itself
            return var[0]
        else:
            # Multi-variable XOR: iteratively apply binary XOR
            xorSuccess = var[0]
            
            for v in var[1:]:
                # XOR(a, b) = (a AND NOT b) OR (NOT a AND b)
                xorSuccess = torch.logical_or(
                    torch.logical_and(xorSuccess, torch.logical_not(v)),
                    torch.logical_and(torch.logical_not(xorSuccess), v)
                )
            
            if onlyConstrains:
                xorLoss = torch.logical_not(xorSuccess)
                return xorLoss
            else:            
                return xorSuccess

    def equivalenceVar(self, _, *var, onlyConstrains = False):
        if self.ifNone(var):
            return None
        
        if len(var) == 0:
            # Equivalence of no variables is True (vacuous truth)
            return torch.ones([self.sampleSize], device=self.current_device, dtype=torch.bool)
        elif len(var) == 1:
            # Equivalence of single variable is True (always equivalent to itself)
            return torch.ones([self.sampleSize], device=self.current_device, dtype=torch.bool)
        else:
            # Multi-variable equivalence: all variables have same truth value
            # equiv(a, b, c, ...) = (all true) OR (all false)
            
            # All true case: AND of all variables
            all_true = var[0]
            for v in var[1:]:
                all_true = torch.logical_and(all_true, v)
            
            # All false case: AND of all negated variables
            all_false = torch.logical_not(var[0])
            for v in var[1:]:
                all_false = torch.logical_and(all_false, torch.logical_not(v))
            
            # Equivalence = (all true) OR (all false)
            equivSuccess = torch.logical_or(all_true, all_false)
            
            if onlyConstrains:
                equivLoss = torch.logical_not(equivSuccess)
                return equivLoss
            else:       
                return equivSuccess
            
    def countVar(self, _, *var, onlyConstrains=False, limitOp='==', limit=1, logicMethodName="COUNT"):
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
            varSum = fixedVar[0].clone()

        for i in range(1, len(fixedVar)):
            varSum.add_(fixedVar[i])

        # Check condition
        if limitOp == '>=':
            # if varSum >= limit:
            countSuccess = torch.ge(varSum, limitTensor)
        elif limitOp == '<=':
            # if varSum <= limit:
            countSuccess = torch.le(varSum, limitTensor)
        elif limitOp == '==':
            # if varSum == limit:
            countSuccess = torch.eq(varSum, limitTensor)

        if onlyConstrains:
            countLoss = torch.logical_not(countSuccess)

            return countLoss
        else:
            return countSuccess

    def compareCountsVar(
        self,
        _,                    # no ILP model object in the PyTorch version
        varsA,                # iterable of literals forming the “left” count
        varsB,                # iterable of literals forming the “right” count
        *,                    # force keyword-only for clarity
        compareOp='>',        # one of '>', '>=', '<', '<=', '==', '!='
        diff: int = 0,        # optional offset:  count(A) − count(B) ∘ diff
        onlyConstrains=False,
        logicMethodName="COUNT_CMP",
    ):
        """
        Compare two literal-sets by their per-sample counts.

        Returns
        -------
        torch.BoolTensor  (shape = [sampleSize])
            • if onlyConstrains is False “success” mask  
            • if onlyConstrains is True  “loss”  mask  (¬success)
        """

        if compareOp not in ('>', '>=', '<', '<=', '==', '!='):
            raise ValueError(f"{logicMethodName}: unsupported operator {compareOp}")

        # ---------- helper to normalise missing literals ----------------------
        def _to_tensor_list(iterable):
            tensors = []
            for v in iterable:
                if torch.is_tensor(v):
                    tensors.append(v)
                else:                       # treat None / scalars as 0-tensor
                    tensors.append(torch.zeros(
                        [self.sampleSize], device=self.current_device))
            return tensors

        tensorsA = _to_tensor_list(varsA)
        tensorsB = _to_tensor_list(varsB)

        # ---------- count “True” literals per sample --------------------------
        countA = torch.zeros([self.sampleSize], device=self.current_device)
        for t in tensorsA:
            countA.add_(t)

        countB = torch.zeros_like(countA)
        for t in tensorsB:
            countB.add_(t)

        diffTensor = torch.full([self.sampleSize], diff, device=self.current_device)
        delta = countA - countB

        # ---------- evaluate relation ----------------------------------------
        if   compareOp == '>':
            success = torch.gt(delta,  diffTensor)
        elif compareOp == '>=':
            success = torch.ge(delta,  diffTensor)
        elif compareOp == '<':
            success = torch.lt(delta,  diffTensor)
        elif compareOp == '<=':
            success = torch.le(delta,  diffTensor)
        elif compareOp == '==':
            success = torch.eq(delta,  diffTensor)
        elif compareOp == '!=':
            success = torch.ne(delta,  diffTensor)

        return torch.logical_not(success) if onlyConstrains else success

    def fixedVar(self, _, var, onlyConstrains = False):
        if self.ifNone([var]):
            return None
        
        fixedSuccess = 1
        
        if onlyConstrains:
            fixedLoss = 1 - fixedSuccess
    
            return fixedLoss
        else:
            return fixedSuccess