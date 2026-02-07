"""
Loss Calculator with Per-Constraint T-Norm Selection.

Drop-in replacement for domiknows/solver/ilpOntSolverTools/lossCalculator.py

Automatically selects optimal t-norm based on constraint type:
- Counting (sumL, atLeastAL): L (Łukasiewicz) - linear gradients
- Upper bounds (atMostAL): G (Gödel) - min-based  
- Boolean (andL, orL): SP (Simplified Product) - fast multiplicative
"""

from time import perf_counter_ns
import torch
from typing import Dict, Optional

from domiknows.graph import fixedL
from domiknows.graph.logicalConstrain import sumL


# Optimal t-norm mapping based on mathematical properties
DEFAULT_TNORM_BY_TYPE = {
    # Counting - Łukasiewicz gives linear gradients to ALL elements
    'sumL': 'L',
    'atLeastL': 'L',
    'atLeastAL': 'L',
    
    # Upper bounds - Gödel works well for "all must be below threshold"
    'atMostL': 'G',
    'atMostAL': 'G',
    
    # Exact count - needs both directions, L is most stable
    'exactL': 'L',
    'exactAL': 'L',
    
    # Boolean logic - Simplified Product for fast multiplicative gradients
    'andL': 'SP',
    'orL': 'SP',
    'nandL': 'SP',
    'norL': 'SP',
    'ifL': 'P',
    'existsL': 'L',
    'notL': 'SP',
    
    # Default fallback
    'default': 'L',
}


def _get_constraint_type(lc) -> str:
    """Extract constraint type name from LC object."""
    if lc is None:
        return 'default'
    
    # Handle executable constraints (wrapped)
    if hasattr(lc, 'innerLC') and lc.innerLC is not None:
        lc = lc.innerLC
    
    return type(lc).__name__


class LossCalculator:
    """
    Loss calculator with per-constraint t-norm selection.
    
    Automatically selects optimal t-norm for each constraint type,
    improving gradient flow compared to using a single global t-norm.
    """
    
    # Class-level config - can be modified before instantiation
    TNORM_CONFIG = DEFAULT_TNORM_BY_TYPE.copy()
    
    def __init__(self, solver):
        """
        Initialize loss calculator.
        
        Args:
            solver: Reference to gurobiILPOntSolver instance
        """
        self.solver = solver
        self.tnorm_config = self.TNORM_CONFIG.copy()
    
    @classmethod
    def set_global_config(cls, config: Dict[str, str]):
        """Set t-norm config for all future instances."""
        cls.TNORM_CONFIG = config.copy()
    
    @classmethod  
    def set_tnorm_for_type(cls, constraint_type: str, tnorm: str):
        """Set t-norm for a specific constraint type globally."""
        if tnorm not in ['L', 'P', 'SP', 'G']:
            raise ValueError(f"Invalid t-norm: {tnorm}")
        cls.TNORM_CONFIG[constraint_type] = tnorm
    
    def _get_tnorm_for_constraint(self, lc, fallback: str, counting_tnorm: Optional[str]) -> str:
        """Determine optimal t-norm for a constraint."""
        ctype = _get_constraint_type(lc)
        
        # Check if counting_tnorm override applies
        is_counting = ctype in ['sumL', 'atLeastL', 'atLeastAL', 
                                 'atMostL', 'atMostAL', 'exactL', 'exactAL']
        
        if counting_tnorm and is_counting:
            return counting_tnorm
        
        return self.tnorm_config.get(ctype, self.tnorm_config.get('default', fallback))
    
    def calculateLoss(self, dn, tnorm='L', counting_tnorm=None):
        """
        Calculate loss with per-constraint optimal t-norm selection.
        
        Args:
            dn: Data node
            tnorm: Fallback t-norm for unknown constraint types
            counting_tnorm: If provided, override t-norm for ALL counting constraints
            
        Returns:
            Dictionary of loss values per logical constraint
        """
        myBooleanMethods = self.solver.myLcLossBooleanMethods
        myBooleanMethods.current_device = dn.current_device
        myBooleanMethods.current_dtype = self.solver.constraintConstructor.current_dtype

        self.solver.myLogger.info('Calculating loss with per-constraint t-norms')
        self.solver.myLoggerTime.info('Calculating loss with per-constraint t-norms')
        
        key = "/local/softmax"
        lcLosses = {}
        
        dn.setActiveLCs()

        # First pass: construct logical constraints with per-constraint t-norms
        for graph in self.solver.myGraph:
            for _, lc in graph.allLogicalConstrains:
                startLC = perf_counter_ns()
                
                if not lc.headLC or not lc.active:
                    continue
                
                if type(lc) is fixedL:
                    continue
                
                lcName = lc.lcName
                lcLosses[lcName] = {}
                current_lcLosses = lcLosses[lcName]
                
                current_lcLosses['lc'] = lc
                
                # Select optimal t-norm for this constraint
                selected_tnorm = self._get_tnorm_for_constraint(lc, tnorm, counting_tnorm)
                current_lcLosses['tnorm_used'] = selected_tnorm
                current_lcLosses['constraint_type'] = _get_constraint_type(lc)
                
                # Set t-norm for this constraint
                myBooleanMethods.setTNorm(selected_tnorm)
                
                # Also set counting t-norm if this is a counting constraint
                is_counting = current_lcLosses['constraint_type'] in [
                    'sumL', 'atLeastL', 'atLeastAL', 'atMostL', 'atMostAL', 'exactL', 'exactAL'
                ]
                if is_counting:
                    myBooleanMethods.setCountingTNorm(selected_tnorm)
                
                self.solver.myLogger.info(f'Processing {lc} with t-norm {selected_tnorm}')
                # Construct constraint
                self.solver.constraintConstructor.current_device = self.solver.current_device
                self.solver.constraintConstructor.myGraph = self.solver.myGraph
                lossList = self.solver.constraintConstructor.constructLogicalConstrains(
                    lc, myBooleanMethods, None, dn, 0, 
                    key=key, headLC=True, loss=True, sample=False)
                current_lcLosses['lossList'] = lossList
                
                endLC = perf_counter_ns()
                current_lcLosses['elapsedInMsLC'] = (endLC - startLC) / 1000000
        
        # Second pass: calculate final loss values
        for currentLcName in lcLosses:
            startLC = perf_counter_ns()
            
            current_lcLosses = lcLosses[currentLcName]
            lossList = current_lcLosses['lossList']
            lossTensor = None
            
            seperateTensorsUsed = False
            if lossList and lossList[0]:
                if torch.is_tensor(lossList[0][0]) and (len(lossList[0][0].shape) == 0 or 
                                                        len(lossList[0][0].shape) == 1 and lossList[0][0].shape[0] == 1):
                    seperateTensorsUsed = True
                
                if seperateTensorsUsed:
                    tensors = []
                    for lossItem in lossList:
                        for item in lossItem:
                            if torch.is_tensor(item):
                                tensors.append(item)
                    if tensors:
                        lossTensor = torch.stack(tensors).mean()
                else:
                    # Handle case where lossList[0][0] might be a list or other structure
                    candidate = lossList[0][0]
                    if torch.is_tensor(candidate):
                        lossTensor = candidate
                    elif isinstance(candidate, (list, tuple)):
                        # Flatten and collect tensors
                        tensors = []
                        def collect_tensors(obj):
                            if torch.is_tensor(obj):
                                tensors.append(obj)
                            elif isinstance(obj, (list, tuple)):
                                for item in obj:
                                    collect_tensors(item)
                        collect_tensors(candidate)
                        if tensors:
                            lossTensor = torch.stack(tensors).mean()
            
            current_lcLosses['loss'] = lossTensor
            
            if lossTensor is not None:
                current_lcLosses['conversionSigmoid'] = 1.0 - lossTensor
                
                # For counting constraints, also compute expected count
                lc = current_lcLosses.get('lc')
                if isinstance(lc, sumL) or (hasattr(lc, 'innerLC') and isinstance(lc.innerLC, sumL)):
                    current_lcLosses['expectedCount'] = lossTensor  # Already represents count
            else:
                current_lcLosses['conversionSigmoid'] = None
            
            endLC = perf_counter_ns()
            current_lcLosses['elapsedInMsLC'] = current_lcLosses.get('elapsedInMsLC', 0) + (endLC - startLC) / 1000000
        
        return lcLosses