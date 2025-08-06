import logging
import torch

from domiknows.graph import DataNode

from domiknows.solver.ilpBooleanMethods import ilpBooleanProcessor 
from domiknows.solver.ilpConfig import ilpConfig 

class lcLossBooleanMethods(ilpBooleanProcessor):
    
    def __init__(self, _ildConfig = ilpConfig) -> None:
        super().__init__()
        self.tnorm = 'P'
        self.counting_tnorm = None
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

    def setCountingTNorm(self, tnorm='L'):
        if tnorm =='L':
            if self.ifLog: self.myLogger.info("Using Lukasiewicz t-norms Formulation")
        elif tnorm =='G':
            if self.ifLog: self.myLogger.info("Using Godel t-norms Formulation")
        elif tnorm =='P':
            if self.ifLog: self.myLogger.info("Using Product t-norms Formulation")
        elif tnorm =='SP':
            if self.ifLog: self.myLogger.info("Using Simplified Product t-norms Formulation")
        #elif tnorm =='LSE':
        #    if self.ifLog: self.myLogger.info("Using Log Sum Exp Formulation")
        else:
            raise Exception('Unknown type of t-norms formulation - %s'%(tnorm))

        self.counting_tnorm = tnorm
        
    def _isTensor(self, v):
        if v is None:
            return -100
        elif torch.is_tensor(v):
            if len(v.shape) == 0 or (len(v.shape) == 1 and v.shape[0] == 1):
                return v.item()
            else:
                return None
        elif isinstance(v, (int, float, complex)):
            return -100
        else:
            return -100
        
    # -- Consider None
    def _fixVar(self, var):
        varFixed = []  
        for v in var:
            if v == None or self._isTensor(v) == -100:
                varFixed.append(torch.tensor([0], device=self.current_device, requires_grad=True, dtype=torch.float64)) # Uses 0 for None
            else:
                varFixed.append(v)
        
        return varFixed
    # -- 

    def notVar(self, _, var, onlyConstrains = False):
        logicMethodName = "NOT"
                
        if self.ifLog: self.myLogger.debug("%s called with : %s"%(logicMethodName,var))
        
        var, = self._fixVar((var,))
            
        tOne = torch.ones(1, device=self.current_device, requires_grad=True, dtype=torch.float64)
        notSuccess = torch.sub(tOne, var)

        if onlyConstrains:
            notLoss = torch.sub(tOne, notSuccess)
            
            return notLoss
        else:            
            return notSuccess
        
    def andVar(self, _, *var, onlyConstrains = False):
        logicMethodName = "AND"
            
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))
              
        var = self._fixVar(var)
                    
        if self.tnorm =='L':
            N = len(var)
            nTorch = torch.tensor([N], device=self.current_device, requires_grad=True, dtype=torch.float64)
            varSum = torch.clone(var[0])
            for v in var[1:]:
                varSum.add_(v)
            
            tZero = torch.zeros(1, device=self.current_device, requires_grad=True, dtype=torch.float64)
            tOne = torch.ones(1, device=self.current_device, requires_grad=True, dtype=torch.float64)

            andSuccess = torch.maximum(torch.add(torch.sub(varSum, nTorch), tOne), tZero) # max(varSum - N + 1, 0)
        elif self.tnorm =='G':
            andSuccess = torch.clone(var[0])
            for v in var[1:]:
                andSuccess = torch.minimum(andSuccess, v)
        elif self.tnorm =='P':
            andSuccess = torch.clone(var[0])
            for v in var[1:]:
                andSuccess.mul_(v)

        if onlyConstrains:
            andLoss = 1 - andSuccess
        
            return andLoss       
        else:
            return andSuccess
    
    def orVar(self, _, *var, onlyConstrains = False):
        logicMethodName = "OR"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))

        var = self._fixVar(var)

        varSum = torch.clone(var[0])
        for v in var[1:]:
            varSum.add_(v)
        
        tOne = torch.ones(1, device=self.current_device, requires_grad=True, dtype=torch.float64)
        if self.tnorm =='L':
            orSuccess = torch.minimum(varSum, tOne) # min(varSum, 1)
        elif self.tnorm =='G':
            orSuccess = torch.clone(var[0])
            for v in var[1:]:
                orSuccess.maximum(v)
        elif self.tnorm =='P':
            varPod = torch.clone(var[0])
            for v in var[1:]:
                varPod.mul_(v)
            orSuccess = torch.sub(varSum, varPod) # varSum - math.prod(var)
            
        if onlyConstrains:
            orLoss = torch.sub(tOne, orSuccess) # 1 - orSuccess
                
            return orLoss
        else:            
            return orSuccess
            
    def nandVar(self, _, *var, onlyConstrains = False):
        logicMethodName = "NAND"
       
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))
        
        # nand(var) = not(and(var))
        nandSuccess = self.notVar(_, self.andVar(_, *var))
        
        tOne = torch.ones(1, device=self.current_device, requires_grad=True, dtype=torch.float64)
        if onlyConstrains:
            nandLoss = torch.sub(tOne, nandSuccess)
                        
            return nandLoss
        else:            
            return nandSuccess
            
    # Singleton tensors 
    def ifVarS(self, _, var1, var2, *, onlyConstrains=False):
        logicMethodName = "IF"
        if self.ifLog:
            self.myLogger.debug("%s called with: var1=%s, var2=%s",
                                logicMethodName, var1, var2)

        # convert None / plain numbers → 0-tensors
        var1, var2 = self._fixVar((var1, var2))

        # ── element-wise implementation (works for scalars **and** vectors) ──
        if self.tnorm == 'L':                        # Łukasiewicz
            ifSuccess = torch.minimum(torch.ones_like(var1),
                                    1 - var1 + var2)

        elif self.tnorm == 'G':                      # Gödel
            ifSuccess = torch.where(var2 >= var1,        # ¬a ∨ b
                                    torch.ones_like(var1),
                                    var2)

        else:                                        # Product (‘P’)
            safe_ratio = torch.where(var1 != 0, var2 / var1, 1)
            ifSuccess  = torch.minimum(torch.ones_like(var1), safe_ratio)

        return 1 - ifSuccess if onlyConstrains else ifSuccess

    def ifVar(self, _, var1, var2, onlyConstrains = False):
        logicMethodName = "IF"

        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
                
        # check if separate tensors used
        if torch.is_tensor(var1) and (len(var1.shape) == 0 or len(var1.shape) == 1 and var1.shape[0] == 1):
            return self.ifVarS(_, var1, var2, onlyConstrains = onlyConstrains)
        
        var1, var2 = self._fixVar((var1,var2))
        tSize = var1.size(dim=0)
        tOneSize = torch.ones(tSize, device=self.current_device, requires_grad=True, dtype=torch.float64)

        var1, var2 = self._fixVar((var1,var2))
        zeroInVar1 = 0 in var1
        zeroInVar2 = 0 in var2
        
        if self.tnorm =='L':
            # min(1, 1 - var1 + var2) #torch.sub
            oneMinusVar1 = torch.sub(tOneSize, var1)
            sumOneMinusVar1AndVar2 = torch.add(oneMinusVar1, var2)
            
            ifSuccess = torch.minimum(tOneSize, sumOneMinusVar1AndVar2) 
        elif self.tnorm =='G':
            #if var2Item > var1Item: 
            #   ifSuccess = torch.mul(var2, torch.div(1, var2)) # 1
            #else: 
            #   ifSuccess = var2
            
            if not zeroInVar2:
                var2Larger = torch.gt(var2, var1)
                var1Larger = torch.ge(var1, var2)
                
                var2Inverse = torch.div(tOneSize, var2)
                var2LargerSuccess = torch.mul(torch.mul(var2Larger, var2), var2Inverse)
                var1LargerSuccess = torch.mul(var1Larger, var2)
                
                ifSuccess = var2LargerSuccess + var1LargerSuccess
            else:
                ifSuccessList = []
                for v1, v2 in zip(var1,var2):
                    ifSuccessList.append(self.ifVarS(_, v1, v2, onlyConstrains = onlyConstrains))
                    
                ifSuccess = torch.stack(ifSuccessList, dim=0)            
        elif self.tnorm =='P':
            #if var1Item != 0:
            #   tOne = torch.ones(tSize, device=self.current_device, requires_grad=True, dtype=torch.float64)
            #   ifSuccess = torch.minimum(tOne, torch.div(var2, var1)) # min(1, var2/var1) # 
            #else:
            #   ifSuccess = torch.mul(var2, torch.div(1, var2)) # 1
            
            if not zeroInVar1:
                div1DivisionDiv2 = torch.div(var2, var1)
                
                ifSuccess = torch.minimum(tOneSize, div1DivisionDiv2)
            else:
                ifSuccessList = []
                for v1, v2 in zip(var1,var2):
                    ifSuccessList.append(self.ifVarS(_, v1, v2, onlyConstrains = onlyConstrains))
                    
                ifSuccess = torch.stack(ifSuccessList, dim=0)     
                            
        tOne = torch.ones(1, device=self.current_device, requires_grad=True, dtype=torch.float64)
        if onlyConstrains:
            ifLoss = torch.sub(tOne, ifSuccess) # 1 - ifSuccess
            
            return ifLoss
        else:            
            return ifSuccess
               
    def norVar(self, _, *var, onlyConstrains = False):
        logicMethodName = "NOR"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName,var))

        # nor(var) = not(or(var)
        norSucess = self.notVar(_, self.orVar(_, *var))
        
        tOne = torch.ones(1, device=self.current_device, requires_grad=True, dtype=torch.float64)
        if onlyConstrains:
            norLoss = torch.sub(1, norSucess)
            
            return norLoss
        else:
            return norSucess
        
    def xorVar(self, _, var1, var2, onlyConstrains = False):
        logicMethodName = "XOR"
        
        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))

        # xor(var1, var2) = or(and(var1, not(var2)), and(not(var1), var2))
        xorSuccess = self.orVar(_, self.andVar(_, var1, self.notVar(_, var2)), self.andVar(_, self.notVar(_, var1), var2))
        
        tOne = torch.ones(1, device=self.current_device, requires_grad=True, dtype=torch.float64)
        if onlyConstrains:
            xorLoss = torch.sub(tOne, xorSuccess)
            
            return xorLoss
        else:
            return xorSuccess
    
    def epqVar(self, _, var1, var2, onlyConstrains = False):
        logicMethodName = "EPQ"
        
        if self.ifLog: self.myLogger.debug("%s called with: var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
            
        # epq(var1, var2) = and(or(var1, not(var2)), or(not(var1), var2)))
        epqSuccess = self.andVar(_, self.orVar(_, var1, self.notVar(_, var2)), self.orVar(_, self.notVar(_, var1), var2))
        
        tOne = torch.ones(1, device=self.current_device, requires_grad=True, dtype=torch.float64)
        if onlyConstrains:
            epqLoss = torch.sub(tOne, epqSuccess)
            
            return epqLoss
        else:
            return epqSuccess

    def calc_probabilities(self, t, s):

        n = len(t)
        dp = torch.zeros(s + 1, device=self.current_device, dtype=torch.float64)
        dp[0] = 1.0
        dp.requires_grad_()
        for i in range(n):
            dp_new = dp.clone()
            dp_new[1:min(s, i + 1) + 1] = dp[1:min(s, i + 1) + 1] * (1 - t[i]) + dp[:min(s, i + 1)] * t[i]
            dp_new[0] = dp[0] * (1 - t[i])
            dp = dp_new
        return dp

    def countVar(self, _, *var, onlyConstrains = False, limitOp = '==', limit = 1, logicMethodName = "COUNT"):
        logicMethodName = "COUNT"

        method=self.counting_tnorm if self.counting_tnorm else self.tnorm
        #if method=="LSE": # log sum exp
        #    exists_at_least_one = lambda t, beta=100.0: torch.clamp(-torch.log((1 / beta) * torch.log(torch.sum(torch.exp(beta * t)))), min=0,max=1)
        #    exists_at_least_s = lambda t, s, beta=10.0: torch.clamp(torch.relu(s - torch.sum(torch.sigmoid(beta * (t - 0.5)))),max=1)
        #    exists_at_most_s = lambda t, s, beta=10.0: torch.clamp(torch.relu(torch.sum(torch.sigmoid(beta * (t - 0.5))) - s),max=1)
        #    exists_exactly_s = lambda t, s, beta=10.0: torch.clamp(torch.abs(s - torch.sum(torch.sigmoid(beta * (t - 0.5)))),max=1)
        if method=="G": # Godel logic
            exists_at_least_one = lambda t: 1 - torch.max(t)
            exists_at_least_s = lambda t, s: 1- torch.min(torch.sort(t, descending=True)[0][:s])
            exists_at_most_s = lambda t, s: 1 - torch.min(torch.sort(1 - t, descending=True)[0][:len(t)-s])
            exists_exactly_s = lambda t, s: 1 - torch.min(torch.min(torch.sort(t, descending=True)[0][:s]) , torch.min(torch.sort(1 - t, descending=True)[0][:len(t)-s]))
        elif method == "L": # Łukasiewicz logic
            exists_at_least_one = lambda t: 1 - torch.min(torch.sum(t), torch.ones(1, device=self.current_device, requires_grad=True, dtype=torch.float64))
            exists_at_least_s = lambda t, s: 1 - torch.max(torch.sum(torch.sort(t, descending=True)[0][:s])-(s-1), torch.zeros(1, device=self.current_device, requires_grad=True, dtype=torch.float64))
            exists_at_most_s = lambda t, s: 1 - torch.max(torch.sum(torch.sort(1 - t, descending=True)[0][:len(t)-s])-(len(t)-s-1), torch.zeros(1, device=self.current_device, requires_grad=True, dtype=torch.float64))
            exists_exactly_s = lambda t, s: 1 - torch.max(torch.sum(torch.sort(t, descending=True)[0][:s])-(s-1)+torch.sum(torch.sort(1 - t, descending=True)[0][:len(t)-s])-(len(t)-s-1), torch.zeros(1, device=self.current_device, requires_grad=True, dtype=torch.float64))

        elif method == "P":  #  Product logic

            exists_at_least_one = lambda t: torch.prod(1 - t)
            exists_at_least_s = lambda t, s: 1 - torch.sum(self.calc_probabilities(t, len(t))[s:])
            exists_at_most_s = lambda t, s: 1 - torch.sum(self.calc_probabilities(t, s))
            exists_exactly_s = lambda t, s: 1 - self.calc_probabilities(t, s)[s]

        else: # "SP" # Simplified product logic
            exists_at_least_one = lambda t: torch.prod(1 - t)
            exists_at_least_s = lambda t, s: 1 - torch.prod(torch.sort(t, descending=True)[0][:s])
            exists_at_most_s = lambda t, s: 1 - torch.prod(torch.sort(1 - t, descending=True)[0][:len(t) - s])
            exists_exactly_s = lambda t, s: 1 - self.calc_probabilities(t, s)[s]


        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName,var))

        var = self._fixVar(var)
                
        varSum = torch.clone(var[0])
        for v in var[1:]:
            varSum.add_(v)
                        
        tZero = torch.zeros(1, device=self.current_device, requires_grad=True, dtype=torch.float64)
        tOne = torch.ones(1, device=self.current_device, requires_grad=True, dtype=torch.float64)
        
        if  limitOp == '==': # == limit
            #countLoss = torch.minimum(torch.maximum(torch.abs(torch.sub(limit, varSum)), tZero), tOne) # min(max(abs(varSum - limit), 0), 1)
            countLoss=exists_exactly_s(varSum, limit)
            if onlyConstrains:
                return countLoss
            else:
                countSuccess = torch.sub(tOne, countLoss)
                return countSuccess
        else:
            if limitOp == '>=': # > limit
                #Existsl
                if onlyConstrains:
                    if limit ==1:return exists_at_least_one(varSum)
                    else: return exists_at_least_s(varSum, limit)
                else:
                    if limit == 1:
                        countSuccess = 1 - exists_at_least_one(varSum)
                    else:
                        countSuccess = 1 - exists_at_least_s(varSum, limit)

            elif limitOp == '<=': # < limit
                #atmostL

                if onlyConstrains:
                    return exists_at_most_s(varSum, limit)
                else:
                    countSuccess = 1 - exists_at_most_s(varSum, limit)

            if onlyConstrains:
                countLoss = torch.sub(tOne, countSuccess)
                return countLoss
            else:
                return countSuccess
            
    def compareCountsVar(
        self,
        _,
        varsA, varsB,
        *,                         # kwargs only
        compareOp: str = '>',
        diff: int | float = 0,
        onlyConstrains: bool = False,
        logicMethodName: str = "COUNT_CMP",
    ):
        """
        Truth / loss for  count(varsA)  compareOp  count(varsB) + diff

        compareOp ∈ {'>', '>=', '<', '<=', '==', '!='}
        diff       constant offset (can be negative)
        onlyConstrains
            • True  → return loss  (1-truth degree)
            • False → return success (truth degree)
        """
        method = self.counting_tnorm if self.counting_tnorm else self.tnorm

        # ── build the two counts ─────────────────────────────────────────────
        varsA = self._fixVar(tuple(varsA))
        varsB = self._fixVar(tuple(varsB))

        sumA = torch.clone(varsA[0])
        for v in varsA[1:]:
            sumA.add_(v)

        sumB = torch.clone(varsB[0])
        for v in varsB[1:]:
            sumB.add_(v)

        expr = sumA - sumB - diff          # Δ = count(A) − count(B) − diff
        tZero = torch.zeros_like(expr)
        tOne  = torch.ones_like(expr)

        # ── Gödel logic ─────────────────────────────────────────────────────
        if method == "G":
            if   compareOp == '>' : success = torch.where(expr >  0, tOne, tZero)
            elif compareOp == '>=': success = torch.where(expr >= 0, tOne, tZero)
            elif compareOp == '<' : success = torch.where(expr <  0, tOne, tZero)
            elif compareOp == '<=': success = torch.where(expr <= 0, tOne, tZero)
            elif compareOp == '==': success = torch.where(expr == 0, tOne, tZero)
            else:                   success = torch.where(expr != 0, tOne, tZero)

        # ── Product logic (smooth) ──────────────────────────────────────────
        elif method == "P":
            β = 10.0  # steepness factor
            if   compareOp in ('>', '>='):
                k = (0.0 if compareOp == '>=' else 1e-6)
                success = torch.sigmoid(β * (expr - k))
            elif compareOp in ('<', '<='):
                k = (0.0 if compareOp == '<=' else -1e-6)
                success = torch.sigmoid(β * (-expr + k))
            elif compareOp == '==':
                success = 1.0 - torch.tanh(β * torch.abs(expr))
            else:  # '!='
                success = torch.tanh(β * torch.abs(expr))

        # ── Łukasiewicz logic (piece-wise linear) ───────────────────────────
        elif method == "L":
            if   compareOp == '>':
                success = torch.clamp(expr, min=0.0, max=1.0)
            elif compareOp == '>=':
                success = torch.clamp(expr + 1.0, min=0.0, max=1.0)
            elif compareOp == '<':
                success = torch.clamp(-expr, min=0.0, max=1.0)
            elif compareOp == '<=':
                success = torch.clamp(1.0 - expr, min=0.0, max=1.0)
            elif compareOp == '==':
                success = torch.clamp(1.0 - torch.abs(expr), min=0.0, max=1.0)
            else:  # '!='
                success_eq = torch.clamp(1.0 - torch.abs(expr), min=0.0, max=1.0)
                success = 1.0 - success_eq

        # ── Simplified-product logic (fallback / default) ───────────────────
        else:  # "SP"
            if   compareOp == '>':
                success = torch.clamp(expr, min=0.0, max=1.0)
            elif compareOp == '>=':
                success = torch.clamp(expr + 1.0, min=0.0, max=1.0)
            elif compareOp == '<':
                success = torch.clamp(-expr, min=0.0, max=1.0)
            elif compareOp == '<=':
                success = torch.clamp(1.0 - expr, min=0.0, max=1.0)
            elif compareOp == '==':
                success = torch.clamp(1.0 - torch.abs(expr), min=0.0, max=1.0)
            else:  # '!='
                success_eq = torch.clamp(1.0 - torch.abs(expr), min=0.0, max=1.0)
                success = 1.0 - success_eq

        # ── return loss or success ──────────────────────────────────────────
        return 1.0 - success if onlyConstrains else success

    def fixedVar(self, _, _var, onlyConstrains = False):
        logicMethodName = "FIXED"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName,_var))
        
        fixedSuccess = torch.ones(1, device=self.current_device, requires_grad=True, dtype=torch.float64)
        
        tOne = torch.ones(1, device=self.current_device, requires_grad=True, dtype=torch.float64)
        if onlyConstrains:
            fixedLoss = torch.sub(tOne,  fixedSuccess)
    
            return fixedLoss
        else:
            return fixedSuccess