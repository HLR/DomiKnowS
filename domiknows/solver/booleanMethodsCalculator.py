import logging

import torch
from domiknows.solver.constraintsProcessorInterface import constraintsProcessor 
from domiknows.solver.ilpConfig import ilpConfig 

class booleanMethodsCalculator(constraintsProcessor):
    
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
    
    def summationVar(self, m, *var, onlyConstrains=False, label=None, logicMethodName="SUMMATION"):
        """
        Sums up a list of binary literals to an integer literal.
        
        Parameters:
        - m: Model context (not used in this method, but included for signature compatibility)
        - *var: Variable number of binary literals (int, bool, torch.Tensor, or None)
        - onlyConstrains: Not used for summation (kept for signature consistency)
        - logicMethodName: Name for logging purposes
        
        Returns:
        - Integer sum of all truthy values
        """
        if onlyConstrains:
            if label is None:
                return None
            return self.countVar(m, *var, onlyConstrains=onlyConstrains, limitOp=">=", limit=label, logicMethodName=logicMethodName)
        
        varSum = 0
        for v in var:
            if v is None:
                continue
            if torch.is_tensor(v):
                varSum += int(v.item())
            else:
                varSum += int(v)
        
        return varSum
    

    def iotaVar(self, _, *var, onlyConstrains=False, temperature=1.0, logicMethodName="IOTA"):
        """
        Verification of definite description: checks if exactly one entity satisfies
        and returns which one.
        
        In verification mode, variables are discrete 0/1 values. We check:
        1. Existence: at least one entity has value 1
        2. Uniqueness: exactly one entity has value 1
        
        Args:
            _: Model context (unused)
            *var: Discrete binary values (0 or 1) for each entity
            onlyConstrains: If True, return 1 if satisfied, 0 if violated
            temperature: Not used in verification
            logicMethodName: Name for logging
        
        Returns:
            - If onlyConstrains=True: 1 if exactly one entity is 1, else 0
            - If onlyConstrains=False: Index of the selected entity (0-indexed),
            or -1 if constraint violated (zero or multiple satisfy)
        """
        # Convert to list of values
        values = []
        for v in var:
            if v is None:
                values.append(0)
            elif torch.is_tensor(v):
                # Handle tensor - flatten and extract values
                flat = v.flatten()
                for i in range(flat.numel()):
                    values.append(int(flat[i].item() > 0.5))
            elif hasattr(v, 'item'):
                values.append(int(v.item() > 0.5))
            else:
                values.append(int(float(v) > 0.5))
        
        if len(values) == 0:
            return -1 if not onlyConstrains else 0
        
        # Find indices where value is 1 (satisfied)
        satisfied_indices = [i for i, v in enumerate(values) if v == 1]
        count = len(satisfied_indices)
        
        if onlyConstrains:
            # Return 1 if exactly one, 0 otherwise
            return 1 if count == 1 else 0
        else:
            # Return index of selected entity
            if count == 1:
                return satisfied_indices[0]
            else:
                # Violation: zero or multiple satisfy
                return -1

    def queryVar(self, _, concept, subclasses, selection_vars, *, onlyConstrains=False, temperature=1.0, logicMethodName="QUERY"):
        """
        Query operator for multiclass attribute selection in verification mode.
        
        Given entity selection (e.g. from iotaL) and a multiclass concept with subclasses,
        returns indicators for which subclass the selected entity belongs to.
        
        In verification mode, all inputs are discrete 0/1 values from argmax predictions.
        This method evaluates whether the query constraint is satisfied.
        
        Verification Logic:
            Given:
            - s_i: selection indicator for entity i (from iotaL, exactly one should be 1)
            - Subclasses: {subclass_0, subclass_1, ..., subclass_k}
            
            Returns:
            - r_j: result indicator for subclass j (one-hot among subclasses)
            
            Verification checks:
            1. Exactly one entity is selected (Σ s_i = 1)
            2. Returns one-hot indicators based on selection
        
        Args:
            _: Model context (unused in verification)
            concept: Parent multiclass concept (e.g., material)
            subclasses: List of (subclass_concept, name, index) tuples
            selection_vars: Entity selection variables from iotaL (list of 0/1 values)
            onlyConstrains: If True, return verification result (1=satisfied, 0=violated)
            temperature: Not used in verification (for interface compatibility)
            logicMethodName: Name for logging
        
        Returns:
            - If onlyConstrains=True: 1 if valid selection exists, 0 otherwise
            - If onlyConstrains=False: List of 0/1 indicators [r_0, r_1, ..., r_k]
        """
        if not subclasses:
            if onlyConstrains:
                return 0
            return None
        
        num_subclasses = len(subclasses)
        
        # Handle None values and convert tensors to numeric
        sel_vars_fixed = []
        for v in selection_vars:
            if v is None:
                sel_vars_fixed.append(0)
            elif torch.is_tensor(v):
                if v.numel() == 1:
                    sel_vars_fixed.append(int(v.item() > 0.5))
                else:
                    # Multi-element tensor - flatten and check if any is 1
                    sel_vars_fixed.append(int(v.sum().item() > 0.5))
            elif hasattr(v, 'item'):
                sel_vars_fixed.append(int(v.item() > 0.5))
            else:
                sel_vars_fixed.append(int(float(v) > 0.5) if v else 0)
        
        if len(sel_vars_fixed) == 0:
            if onlyConstrains:
                return 0
            return [0] * num_subclasses
        
        # Find which entity is selected (has value 1)
        selected_idx = -1
        selected_count = 0
        for i, v in enumerate(sel_vars_fixed):
            if v == 1:
                if selected_idx == -1:
                    selected_idx = i
                selected_count += 1
        
        if onlyConstrains:
            # Return 1 if exactly one entity selected, 0 otherwise
            return 1 if selected_count == 1 else 0
        
        if selected_idx == -1:
            # No entity selected - constraint violated, return zeros
            return [0] * num_subclasses
        
        # For verification, return one-hot with first subclass selected
        # The actual subclass determination happens at a higher level
        # based on the model's predictions for that entity
        result = [0] * num_subclasses
        if num_subclasses > 0:
            result[0] = 1
        
        return result
