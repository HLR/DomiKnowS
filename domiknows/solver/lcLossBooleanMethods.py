import logging
import torch

from domiknows.graph import DataNode

from domiknows.solver.ilpBooleanMethods import ilpBooleanProcessor 
from domiknows.solver.ilpConfig import ilpConfig 
from utils import setup_logger, getProductionModeStatus  # Import the logging utility and production mode checker

class lcLossBooleanMethods(ilpBooleanProcessor):
    
    def __init__(self, _ildConfig = ilpConfig) -> None:
        super().__init__()
        self.tnorm = 'P'
        self.counting_tnorm = None
        self.grad = True
        
        self.myLogger = logging.getLogger(ilpConfig['log_name'])
        self.ifLog =  ilpConfig['ifLog']
        
        # Set up dedicated logger for count operations
        self._setup_count_logger(_ildConfig)

    def _setup_count_logger(self, config):
        """Set up dedicated logger for count operations."""
        count_log_config = {
            'log_name': 'lcLossCountOperations',
            'log_level': logging.DEBUG,
            'log_filename': 'lc_loss_count_operations.log',
            'log_filesize': 50*1024*1024,  # 50MB
            'log_backupCount': 5,
            'log_fileMode': 'a',
            'log_dir': 'logs',
            'timestamp_backup_count': 10
        }
        
        # Override with provided config if available
        if config and isinstance(config, dict):
            if 'count_log_level' in config:
                count_log_config['log_level'] = config['count_log_level']
            if 'count_log_filename' in config:
                count_log_config['log_filename'] = config['count_log_filename']
            if 'count_log_dir' in config:
                count_log_config['log_dir'] = config['count_log_dir']
        
        self.countLogger = setup_logger(count_log_config)
        
        # Disable logger if in production mode
        if getProductionModeStatus():
            self.countLogger.addFilter(lambda record: False)
            self.countLogger.info("Count operations logger disabled due to production mode")
        else:
            self.countLogger.info("=== lcLossBooleanMethods Count Operations Logger Initialized ===")

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

        else:                                        # Product ('P')
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
        
    def xorVar(self, _, *var, onlyConstrains = False):
        logicMethodName = "XOR"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))

        if len(var) == 0:
            # XOR of no variables is False
            xorSuccess = torch.zeros(1, device=self.current_device, requires_grad=True, dtype=torch.float64)
        elif len(var) == 1:
            # XOR of single variable is the variable itself
            var = self._fixVar(var)
            xorSuccess = var[0]
        else:
            # Multi-variable XOR: iteratively apply binary XOR using t-norms
            var = self._fixVar(var)
            xorSuccess = torch.clone(var[0])
            
            for v in var[1:]:
                if self.tnorm == 'L':  # Łukasiewicz
                    # XOR(a, b) = min(1, a + b) - max(0, a + b - 1)
                    tOne = torch.ones_like(xorSuccess)
                    tZero = torch.zeros_like(xorSuccess)
                    sum_ab = torch.add(xorSuccess, v)
                    xorSuccess = torch.minimum(tOne, sum_ab) - torch.maximum(tZero, sum_ab - tOne)
                    
                elif self.tnorm == 'G':  # Gödel
                    # XOR(a, b) = max(min(a, 1-b), min(1-a, b))
                    tOne = torch.ones_like(xorSuccess)
                    not_a = torch.sub(tOne, xorSuccess)
                    not_b = torch.sub(tOne, v)
                    xorSuccess = torch.maximum(torch.minimum(xorSuccess, not_b), 
                                            torch.minimum(not_a, v))
                    
                elif self.tnorm == 'P':  # Product
                    # XOR(a, b) = a*(1-b) + b*(1-a) = a + b - 2*a*b
                    xorSuccess = torch.add(xorSuccess, v) - 2 * torch.mul(xorSuccess, v)
        
        tOne = torch.ones(1, device=self.current_device, requires_grad=True, dtype=torch.float64)
        if onlyConstrains:
            xorLoss = torch.sub(tOne, xorSuccess)
            
            return xorLoss
        else:
            return xorSuccess
    
    def equivalenceVar(self, _, *var, onlyConstrains = False):
        logicMethodName = "EQUIVALENCE"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))

        if len(var) == 0:
            # Equivalence of no variables is True (vacuous truth)
            equivSuccess = torch.ones(1, device=self.current_device, requires_grad=True, dtype=torch.float64)
        elif len(var) == 1:
            # Equivalence of single variable is True (always equivalent to itself)
            equivSuccess = torch.ones(1, device=self.current_device, requires_grad=True, dtype=torch.float64)
        else:
            # Multi-variable equivalence using t-norms: all variables have same truth value
            var = self._fixVar(var)
            
            if self.tnorm == 'L':  # Łukasiewicz
                # AND(a,b,c,...) = max(0, sum(vars) - (n-1))
                # AND(¬a,¬b,¬c,...) = max(0, sum(1-vars) - (n-1)) = max(0, n - sum(vars) - (n-1)) = max(0, 1 - sum(vars))
                n = len(var)
                tZero = torch.zeros(1, device=self.current_device, requires_grad=True, dtype=torch.float64)
                tOne = torch.ones(1, device=self.current_device, requires_grad=True, dtype=torch.float64)
                
                var_sum = torch.clone(var[0])
                for v in var[1:]:
                    var_sum.add_(v)
                
                # All true case: max(0, sum - (n-1))
                all_true = torch.maximum(tZero, var_sum - (n-1))
                
                # All false case: max(0, 1 - sum)  
                all_false = torch.maximum(tZero, tOne - var_sum)
                
                # OR: min(1, all_true + all_false)
                equivSuccess = torch.minimum(tOne, all_true + all_false)
                
            elif self.tnorm == 'G':  # Gödel
                # AND = min, OR = max
                # All true case: min(all vars)
                all_true = torch.clone(var[0])
                for v in var[1:]:
                    all_true = torch.minimum(all_true, v)
                
                # All false case: min(all 1-vars)
                tOne = torch.ones_like(var[0])
                all_false = torch.sub(tOne, var[0])  # 1 - var[0]
                for v in var[1:]:
                    all_false = torch.minimum(all_false, torch.sub(tOne, v))
                
                # OR: max(all_true, all_false)
                equivSuccess = torch.maximum(all_true, all_false)
                
            elif self.tnorm == 'P':  # Product
                # AND = product, OR = sum - product
                # All true case: product(all vars)
                all_true = torch.clone(var[0])
                for v in var[1:]:
                    all_true.mul_(v)
                
                # All false case: product(all 1-vars)
                tOne = torch.ones_like(var[0])
                all_false = torch.sub(tOne, var[0])  # 1 - var[0]
                for v in var[1:]:
                    all_false.mul_(torch.sub(tOne, v))
                
                # OR: all_true + all_false - all_true * all_false
                equivSuccess = all_true + all_false - torch.mul(all_true, all_false)
        
        tOne = torch.ones(1, device=self.current_device, requires_grad=True, dtype=torch.float64)
        if onlyConstrains:
            equivLoss = torch.sub(tOne, equivSuccess)
            
            return equivLoss
        else:
            return equivSuccess

    def calc_probabilities(self, t: torch.Tensor, n: int | None = None) -> torch.Tensor:
        """
        Poisson–binomial PMF over counts 0..n for independent Bernoulli probs in t.
        Returns a length-(n+1) vector where pmf[k] = P(K==k).
        Differentiable w.r.t. t. t is 1-D with entries in [0,1].
        """
        if t.ndim != 1:
            t = t.view(-1)
        if n is None:
            n = t.numel()
        # Use only the first n probs if more are given
        p = t[:n]

        # PMF DP: start with P(K=0)=1
        pmf = torch.zeros(n + 1, dtype=p.dtype, device=p.device)
        pmf[0] = 1.0

        # For each probability p_i, convolve with [1-p_i, p_i]
        # Reverse k loop to avoid overwriting dependencies.
        for pi in p:
            # pmf_new[k] = pmf[k]*(1-pi) + pmf[k-1]*pi
            # Do it in-place with careful ordering (descending k)
            prev = pmf.clone()
            pmf[1:] = prev[1:] * (1 - pi) + prev[:-1] * pi
            pmf[0]  = prev[0]  * (1 - pi)

        # Numerical guard
        pmf = torch.clamp(pmf, 0.0, 1.0)
        # renormalize small drift
        s = pmf.sum()
        if torch.isfinite(s) and s > 0:
            pmf = pmf / s
        return pmf

    def countVar(
        self,
        _,
        *var,
        onlyConstrains: bool = False,
        limitOp: str = "==",
        limit: int = 1,
        logicMethodName: str = "COUNT",
    ):
        logicMethodName = "COUNT"

        # Enhanced logging for count operations
        self.countLogger.info(f"=== {logicMethodName} Operation Started ===")
        self.countLogger.info(f"Input parameters: limitOp='{limitOp}', limit={limit}, onlyConstrains={onlyConstrains}")
        self.countLogger.info(f"Number of input variables: {len(var)}")
        self.countLogger.info(f"T-norm method: {self.counting_tnorm if getattr(self, 'counting_tnorm', None) else self.tnorm}")

        # ---- Normalize inputs: skip None, move to correct device/dtype, clamp to [0,1]
        vals = []
        for i, v in enumerate(var):
            self.countLogger.debug(f"Processing variable {i}: {v} (type: {type(v)})")
            if v is None:
                self.countLogger.debug(f"Variable {i} is None, skipping")
                continue
            tv = self._fixVar((v,))[0] if not isinstance(v, torch.Tensor) else v
            tv = tv.to(device=self.current_device, dtype=torch.float64)
            tv = torch.clamp(tv, 0.0, 1.0)
            self.countLogger.debug(f"Variable {i} after processing: {tv.item() if tv.numel() == 1 else tv}")
            vals.append(tv)
            
        for i, v in enumerate(vals):
            if v.numel() != 1:
               self.countLogger.warning(f"Variable {i}: countVar expects scalar literals; got shape {tuple(v.shape)}: {v}")

        if len(vals) == 0:
            self.countLogger.info("No valid variables found, using zero tensor")
            t = torch.zeros(1, device=self.current_device, dtype=torch.float64)
        else:
            # If scalar -> keep as scalar; if vector/tensor -> flatten.
            parts = [x.reshape(()) if x.numel() == 1 else x.flatten() for x in vals]
            t = torch.cat([p.view(-1) for p in parts])  # t is 1-D, length n = total literals

        n = t.numel()
        s = int(limit)  # ensure Python int
        
        self.countLogger.info(f"Final tensor t: {t}")
        self.countLogger.info(f"Tensor length n: {n}, target count s: {s}")

        # ---- Choose t-norm family for counting
        method = self.counting_tnorm if getattr(self, "counting_tnorm", None) else self.tnorm
        self.countLogger.info(f"Using method: {method}")

        # Helpers return a **loss in [0,1]** (0 = satisfied, 1 = maximally violated).
        if method == "G":  # Gödel
            self.countLogger.debug("Defining Gödel t-norm helper functions")
            exists_at_least_one = lambda t: 1 - torch.max(t)  # loss is 0 when any literal is 1
            exists_at_least_s = lambda t, s: 1 - torch.min(torch.sort(t, descending=True)[0][: max(min(s, n), 0)])
            exists_at_most_s = lambda t, s: 1 - torch.min(torch.sort(1 - t, descending=True)[0][: max(n - max(min(s, n), 0), 0)])
            exists_exactly_s = lambda t, s: 1 - torch.min(
                torch.min(torch.sort(t, descending=True)[0][: max(min(s, n), 0)]),
                torch.min(torch.sort(1 - t, descending=True)[0][: max(n - max(min(s, n), 0), 0)]),
            )

        elif method == "L":  # Łukasiewicz
            self.countLogger.debug("Defining Łukasiewicz t-norm helper functions")
            one = torch.tensor(1.0, device=self.current_device, dtype=torch.float64)
            zero = torch.tensor(0.0, device=self.current_device, dtype=torch.float64)
            exists_at_least_one = lambda t: 1 - torch.minimum(torch.sum(t), one)
            exists_at_least_s = lambda t, s: 1 - torch.maximum(
                torch.sum(torch.sort(t, descending=True)[0][: max(min(s, n), 0)]) - (max(min(s, n), 0) - 1),
                zero,
            )
            exists_at_most_s = lambda t, s: 1 - torch.maximum(
                torch.sum(torch.sort(1 - t, descending=True)[0][: max(n - max(min(s, n), 0), 0)]) - (max(n - max(min(s, n), 0), 0) - 1),
                zero,
            )
            exists_exactly_s = lambda t, s: 1 - torch.maximum(
                torch.sum(torch.sort(t, descending=True)[0][: max(min(s, n), 0)]) - (max(min(s, n), 0) - 1)
                + torch.sum(torch.sort(1 - t, descending=True)[0][: max(n - max(min(s, n), 0), 0)]) - (max(n - max(min(s, n), 0), 0) - 1),
                zero,
            )

        elif method == "P":  # Product
            self.countLogger.debug("Defining Product t-norm helper functions")
            exists_at_least_one = lambda t: torch.prod(1 - t)  # loss small if there exists a high literal
            exists_at_least_s = lambda t, s: 1 - torch.sum(self.calc_probabilities(t, t.numel())[max(min(s, n), 0) :])
            exists_at_most_s = lambda t, s: 1 - torch.sum(self.calc_probabilities(t, t.numel())[: max(min(s, n), 0) + 1])
            exists_exactly_s = lambda t, s: 1 - self.calc_probabilities(t, t.numel())[max(min(s, n), 0)]

        else:  # "SP" Simplified Product
            self.countLogger.debug("Defining Simplified Product t-norm helper functions")
            exists_at_least_one = lambda t: torch.prod(1 - t)
            exists_at_least_s = lambda t, s: 1 - torch.prod(torch.sort(t, descending=True)[0][: max(min(s, n), 0)]) if max(min(s, n), 0) > 0 else 1.0
            exists_at_most_s = lambda t, s: 1 - torch.prod(torch.sort(1 - t, descending=True)[0][: max(n - max(min(s, n), 0), 0)]) if max(n - max(min(s, n), 0), 0) > 0 else 1.0
            exists_exactly_s = lambda t, s: 1 - self.calc_probabilities(t, t.numel())[max(min(s, n), 0)]

        # ---- Compute loss or success
        self.countLogger.info(f"Computing result for operation '{limitOp}' with limit {s}")
        
        if limitOp == "==":
            self.countLogger.debug("Using exists_exactly_s function")
            loss = exists_exactly_s(t, s)
        elif limitOp == ">=":
            if s <= 1:
                self.countLogger.debug("Using exists_at_least_one function (s <= 1)")
                loss = exists_at_least_one(t)
            else:
                self.countLogger.debug(f"Using exists_at_least_s function with s={s}")
                loss = exists_at_least_s(t, s)
        elif limitOp == "<=":
            self.countLogger.debug("Using exists_at_most_s function")
            loss = exists_at_most_s(t, s)
        else:
            self.countLogger.error(f"Unsupported limitOp: {limitOp}")
            raise ValueError(f"Unsupported limitOp: {limitOp}")

        self.countLogger.info(f"Computed loss: {loss.item() if hasattr(loss, 'item') else loss}")

        # Check if loss is in valid range [0,1]
        loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
        if not (0 <= loss_val <= 1):
            self.countLogger.warning(f"Loss out of bounds [0,1]: {loss_val}")
            # clamp
            loss = torch.clamp(loss, 0.0, 1.0)
            self.countLogger.info(f"Loss clamped to: {loss.item() if hasattr(loss, 'item') else loss}")

        if onlyConstrains:
            result = loss  # loss in [0,1]
            self.countLogger.info(f"Returning loss (onlyConstrains=True): {result.item() if hasattr(result, 'item') else result}")
        else:
            success = 1.0 - loss
            result = torch.clamp(success, 0.0, 1.0)
            self.countLogger.info(f"Returning success (onlyConstrains=False): {result.item() if hasattr(result, 'item') else result}")
            
        self.countLogger.info(f"=== {logicMethodName} Operation Completed ===\n")
        return result
            
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
        
        # Enhanced logging for compareCountsVar operations
        self.countLogger.info(f"=== {logicMethodName} Operation Started ===")
        self.countLogger.info(f"Input parameters: compareOp='{compareOp}', diff={diff}, onlyConstrains={onlyConstrains}")
        self.countLogger.info(f"Number of varsA: {len(varsA)}, Number of varsB: {len(varsB)}")
        
        method = self.counting_tnorm if self.counting_tnorm else self.tnorm
        self.countLogger.info(f"Using method: {method}")

        # ── build the two counts ─────────────────────────────────────────────
        self.countLogger.debug("Processing varsA...")
        varsA = self._fixVar(tuple(varsA))
        for i, v in enumerate(varsA):
            self.countLogger.debug(f"varsA[{i}]: {v.item() if v.numel() == 1 else v}")

        self.countLogger.debug("Processing varsB...")
        varsB = self._fixVar(tuple(varsB))
        for i, v in enumerate(varsB):
            self.countLogger.debug(f"varsB[{i}]: {v.item() if v.numel() == 1 else v}")

        sumA = torch.clone(varsA[0])
        for v in varsA[1:]:
            sumA.add_(v)
        self.countLogger.info(f"Sum of varsA: {sumA.item() if sumA.numel() == 1 else sumA}")

        sumB = torch.clone(varsB[0])
        for v in varsB[1:]:
            sumB.add_(v)
        self.countLogger.info(f"Sum of varsB: {sumB.item() if sumB.numel() == 1 else sumB}")

        expr = sumA - sumB - diff          # Δ = count(A) − count(B) − diff
        self.countLogger.info(f"Expression (countA - countB - diff): {expr.item() if expr.numel() == 1 else expr}")
        
        tZero = torch.zeros_like(expr)
        tOne  = torch.ones_like(expr)

        # ── Gödel logic ─────────────────────────────────────────────────────
        if method == "G":
            self.countLogger.debug("Using Gödel logic")
            if   compareOp == '>' : 
                success = torch.where(expr >  0, tOne, tZero)
                self.countLogger.debug(f"Gödel '>' condition: expr > 0 → {success.item() if success.numel() == 1 else success}")
            elif compareOp == '>=': 
                success = torch.where(expr >= 0, tOne, tZero)
                self.countLogger.debug(f"Gödel '>=' condition: expr >= 0 → {success.item() if success.numel() == 1 else success}")
            elif compareOp == '<' : 
                success = torch.where(expr <  0, tOne, tZero)
                self.countLogger.debug(f"Gödel '<' condition: expr < 0 → {success.item() if success.numel() == 1 else success}")
            elif compareOp == '<=': 
                success = torch.where(expr <= 0, tOne, tZero)
                self.countLogger.debug(f"Gödel '<=' condition: expr <= 0 → {success.item() if success.numel() == 1 else success}")
            elif compareOp == '==': 
                success = torch.where(expr == 0, tOne, tZero)
                self.countLogger.debug(f"Gödel '==' condition: expr == 0 → {success.item() if success.numel() == 1 else success}")
            else:                   
                success = torch.where(expr != 0, tOne, tZero)
                self.countLogger.debug(f"Gödel '!=' condition: expr != 0 → {success.item() if success.numel() == 1 else success}")

        # ── Product logic (smooth) ──────────────────────────────────────────
        elif method == "P":
            β = 10.0  # steepness factor
            self.countLogger.debug(f"Using Product logic with steepness β={β}")
            if   compareOp in ('>', '>='):
                k = (0.0 if compareOp == '>=' else 1e-6)
                sigmoid_input = β * (expr - k)
                success = torch.sigmoid(sigmoid_input)
                self.countLogger.debug(f"Product '{compareOp}' sigmoid input: {sigmoid_input.item() if sigmoid_input.numel() == 1 else sigmoid_input}")
                self.countLogger.debug(f"Product '{compareOp}' result: {success.item() if success.numel() == 1 else success}")
            elif compareOp in ('<', '<='):
                k = (0.0 if compareOp == '<=' else -1e-6)
                sigmoid_input = β * (-expr + k)
                success = torch.sigmoid(sigmoid_input)
                self.countLogger.debug(f"Product '{compareOp}' sigmoid input: {sigmoid_input.item() if sigmoid_input.numel() == 1 else sigmoid_input}")
                self.countLogger.debug(f"Product '{compareOp}' result: {success.item() if success.numel() == 1 else success}")
            elif compareOp == '==':
                tanh_input = β * torch.abs(expr)
                success = 1.0 - torch.tanh(tanh_input)
                self.countLogger.debug(f"Product '==' tanh input: {tanh_input.item() if tanh_input.numel() == 1 else tanh_input}")
                self.countLogger.debug(f"Product '==' result: {success.item() if success.numel() == 1 else success}")
            else:  # '!='
                tanh_input = β * torch.abs(expr)
                success = torch.tanh(tanh_input)
                self.countLogger.debug(f"Product '!=' tanh input: {tanh_input.item() if tanh_input.numel() == 1 else tanh_input}")
                self.countLogger.debug(f"Product '!=' result: {success.item() if success.numel() == 1 else success}")

        # ── Łukasiewicz logic (piece-wise linear) ───────────────────────────
        elif method == "L":
            self.countLogger.debug("Using Łukasiewicz logic")
            if   compareOp == '>':
                success = torch.clamp(expr, min=0.0, max=1.0)
                self.countLogger.debug(f"Łukasiewicz '>' clamp(expr, 0, 1): {success.item() if success.numel() == 1 else success}")
            elif compareOp == '>=':
                success = torch.clamp(expr + 1.0, min=0.0, max=1.0)
                self.countLogger.debug(f"Łukasiewicz '>=' clamp(expr + 1, 0, 1): {success.item() if success.numel() == 1 else success}")
            elif compareOp == '<':
                success = torch.clamp(-expr, min=0.0, max=1.0)
                self.countLogger.debug(f"Łukasiewicz '<' clamp(-expr, 0, 1): {success.item() if success.numel() == 1 else success}")
            elif compareOp == '<=':
                success = torch.clamp(1.0 - expr, min=0.0, max=1.0)
                self.countLogger.debug(f"Łukasiewicz '<=' clamp(1 - expr, 0, 1): {success.item() if success.numel() == 1 else success}")
            elif compareOp == '==':
                abs_expr = torch.abs(expr)
                success = torch.clamp(1.0 - abs_expr, min=0.0, max=1.0)
                self.countLogger.debug(f"Łukasiewicz '==' abs(expr): {abs_expr.item() if abs_expr.numel() == 1 else abs_expr}")
                self.countLogger.debug(f"Łukasiewicz '==' clamp(1 - |expr|, 0, 1): {success.item() if success.numel() == 1 else success}")
            else:  # '!='
                abs_expr = torch.abs(expr)
                success_eq = torch.clamp(1.0 - abs_expr, min=0.0, max=1.0)
                success = 1.0 - success_eq
                self.countLogger.debug(f"Łukasiewicz '!=' success_eq: {success_eq.item() if success_eq.numel() == 1 else success_eq}")
                self.countLogger.debug(f"Łukasiewicz '!=' final: {success.item() if success.numel() == 1 else success}")

        # ── Simplified-product logic (fallback / default) ───────────────────
        else:  # "SP"
            self.countLogger.debug("Using Simplified Product logic")
            if   compareOp == '>':
                success = torch.clamp(expr, min=0.0, max=1.0)
                self.countLogger.debug(f"SP '>' clamp(expr, 0, 1): {success.item() if success.numel() == 1 else success}")
            elif compareOp == '>=':
                success = torch.clamp(expr + 1.0, min=0.0, max=1.0)
                self.countLogger.debug(f"SP '>=' clamp(expr + 1, 0, 1): {success.item() if success.numel() == 1 else success}")
            elif compareOp == '<':
                success = torch.clamp(-expr, min=0.0, max=1.0)
                self.countLogger.debug(f"SP '<' clamp(-expr, 0, 1): {success.item() if success.numel() == 1 else success}")
            elif compareOp == '<=':
                success = torch.clamp(1.0 - expr, min=0.0, max=1.0)
                self.countLogger.debug(f"SP '<=' clamp(1 - expr, 0, 1): {success.item() if success.numel() == 1 else success}")
            elif compareOp == '==':
                abs_expr = torch.abs(expr)
                success = torch.clamp(1.0 - abs_expr, min=0.0, max=1.0)
                self.countLogger.debug(f"SP '==' abs(expr): {abs_expr.item() if abs_expr.numel() == 1 else abs_expr}")
                self.countLogger.debug(f"SP '==' clamp(1 - |expr|, 0, 1): {success.item() if success.numel() == 1 else success}")
            else:  # '!='
                abs_expr = torch.abs(expr)
                success_eq = torch.clamp(1.0 - abs_expr, min=0.0, max=1.0)
                success = 1.0 - success_eq
                self.countLogger.debug(f"SP '!=' success_eq: {success_eq.item() if success_eq.numel() == 1 else success_eq}")
                self.countLogger.debug(f"SP '!=' final: {success.item() if success.numel() == 1 else success}")

        # ── return loss or success ──────────────────────────────────────────
        if onlyConstrains:
            result = 1.0 - success
            self.countLogger.info(f"Returning loss (onlyConstrains=True): {result.item() if result.numel() == 1 else result}")
        else:
            result = success
            self.countLogger.info(f"Returning success (onlyConstrains=False): {result.item() if result.numel() == 1 else result}")
            
        self.countLogger.info(f"=== {logicMethodName} Operation Completed ===\n")
        return result

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