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
        #results = self.andVar(_, self.andVar(_, var1), var2)
        
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
           
    def xorVar(self, _, *var, onlyConstrains = False):
        # -- Consider None
        varFixed = []  
        for v in var:
            if v is None:
                varFixed.append(0) # when None - XOR treats None as False
            else:
                varFixed.append(v)
        var = varFixed
        # --
        
        if len(var) == 0:
            return 0  # XOR of no variables is False
        elif len(var) == 1:
            return int(var[0])  # XOR of single variable is the variable itself
        else:
            # Multi-variable XOR: true when odd number of variables are true
            true_count = 0
            for v in var:
                if torch.is_tensor(v):
                    true_count += int(v.item())
                else:
                    true_count += int(v)
            
            # XOR is true when odd number of inputs are true
            xorSuccess = 1 if (true_count % 2 == 1) else 0
            return xorSuccess

    def equivalenceVar(self, _, *var, onlyConstrains = False):
        # -- Consider None
        varFixed = []  
        for v in var:
            if v is None:
                varFixed.append(0) # when None - treat as False
            else:
                varFixed.append(v)
        var = varFixed
        # --
        
        if len(var) == 0:
            return 1  # Equivalence of no variables is True (vacuous truth)
        elif len(var) == 1:
            return 1  # Equivalence of single variable is True (always equivalent to itself)
        else:
            # Multi-variable equivalence: true when all variables have same truth value
            
            # Convert all to integer values for comparison
            values = []
            for v in var:
                if torch.is_tensor(v):
                    values.append(int(v.item()))
                else:
                    values.append(int(v))
            
            # Check if all values are the same (all 0 or all 1)
            first_value = values[0]
            all_same = all(val == first_value for val in values)
            
            equivSuccess = 1 if all_same else 0
            return equivSuccess
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
        
    def countVar(
        self,
        _,
        *var,
        onlyConstrains: bool = False,
        limitOp: str = "==",
        limit: int = 1,
        logicMethodName: str = "COUNT",
    ) -> int:
        """
        Return 1 when the number of truthy literals in *var satisfies
            sum(var)  limitOp  limit
        else return 0.

        Parameters
        ----------
        _ : Any
            Ignored (kept for signature compatibility with ILP-based subclasses).
        *var : Iterable[Union[int, bool, torch.Tensor, None]]
            Binary literals to count. `None` values are skipped.
        onlyConstrains : bool
            Unused here; present only for API compatibility.
        limitOp : str
            Comparison operator: one of '>=', '<=', '=='.
        limit : int
            Threshold on the count.
        logicMethodName : str
            Label used in error messages (not used here).

        Returns
        -------
        int
            1 if the comparison is satisfied, otherwise 0.
        """
        # --- robust scalar summation (supports tensors and scalars) ------------
        varSum = 0
        for v in var:
            if v is None:
                continue
            if torch.is_tensor(v):
                varSum += int(v.item())
            else:
                varSum += int(v)

        # --- evaluate the comparison ------------------------------------------
        if limitOp == ">=" and varSum >= limit:
            return 1
        elif limitOp == "<=" and varSum <= limit:
            return 1
        elif limitOp == "==" and varSum == limit:
            return 1
        else:
            return 0
    
    def compareCountsVar(
        self,
        _,  
        varsA,
        varsB,
        *,
        compareOp='>',
        diff=0,
        onlyConstrains=False,          # kept for signature compatibility
        logicMethodName="COUNT_CMP",
    ):
        """
        Compare sizes of two sets of binary literals.

            result = 1  iff   count(varsA)  compareOp  ( count(varsB) + diff )

        Supported operators: '>', '>=', '<', '<=', '==', '!='
        Each literal may be 1/0, torch scalar tensor, or None (ignored).
        """

        if compareOp not in ('>', '>=', '<', '<=', '==', '!='):
            raise ValueError(f"{logicMethodName}: unsupported operator {compareOp}")

        def _count(seq):
            total = 0
            for v in seq:
                if v is None:
                    continue
                total += v.item() if torch.is_tensor(v) else v
            return total

        lhs = _count(varsA) - _count(varsB)           # count(A) - count(B)

        if   compareOp == '>':   res = lhs >  diff
        elif compareOp == '>=':  res = lhs >= diff
        elif compareOp == '<':   res = lhs <  diff
        elif compareOp == '<=':  res = lhs <= diff
        elif compareOp == '==':  res = lhs == diff
        elif compareOp == '!=':  res = lhs != diff

        return 1 if res else 0

    def fixedVar(self, _, _var, onlyConstrains = False):
        fixedSuccess = 1
        
        return fixedSuccess