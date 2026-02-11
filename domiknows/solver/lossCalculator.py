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


def _collect_tensors(obj):
    """Recursively collect tensors from nested lists/tuples."""
    tensors = []
    if torch.is_tensor(obj):
        tensors.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            tensors.extend(_collect_tensors(item))
    return tensors


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

    def _compute_loss_tensor(self, lossList):
        """
        Reduce a lossList (from constructLogicalConstrains) into a single loss tensor.
        
        Args:
            lossList: Nested list of tensors from constraint construction
            
        Returns:
            Single loss tensor or None
        """
        if not lossList or not lossList[0]:
            return None

        separate_tensors = (
            torch.is_tensor(lossList[0][0])
            and (lossList[0][0].dim() == 0
                 or (lossList[0][0].dim() == 1 and lossList[0][0].shape[0] == 1))
        )

        if separate_tensors:
            tensors = []
            for lossItem in lossList:
                for item in lossItem:
                    if torch.is_tensor(item):
                        tensors.append(item)
            return torch.stack(tensors).mean() if tensors else None

        candidate = lossList[0][0]
        if torch.is_tensor(candidate):
            return candidate

        tensors = _collect_tensors(candidate)
        return torch.stack(tensors).mean() if tensors else None

    def calculate_single_lc_loss(self, lc, dn, key, tnorm, counting_tnorm, label=None):
        """
        Calculate loss for a single logical constraint.
        
        Args:
            lc: The logical constraint
            dn: Data node
            key: Softmax key
            tnorm: Fallback t-norm
            counting_tnorm: Override t-norm for counting constraints (or None)
            label: Optional label for the sumL constraint
        Returns:
            Dict with loss info for this constraint, or None if constraint should be skipped
        """
        if not lc.headLC or not lc.active:
            return None

        if type(lc) is fixedL:
            return None

        start = perf_counter_ns()

        myBooleanMethods = self.solver.myLcLossBooleanMethods
        result = {'lc': lc}

        # Select optimal t-norm
        selected_tnorm = self._get_tnorm_for_constraint(lc, tnorm, counting_tnorm)
        constraint_type = _get_constraint_type(lc)
        result['tnorm_used'] = selected_tnorm
        result['constraint_type'] = constraint_type

        myBooleanMethods.setTNorm(selected_tnorm)

        is_counting = constraint_type in [
            'sumL', 'atLeastL', 'atLeastAL', 'atMostL', 'atMostAL', 'exactL', 'exactAL'
        ]
        if isinstance(lc, sumL) and label is None:
            myBooleanMethods.setCountingTNorm(selected_tnorm)
            if label is None:
                # Use datanode method to get the label
                label = dn.getExecutableConstraintLabel(lc.lcName).float()
                if label is None:
                    return None
        
        self.solver.myLogger.info(f'Processing {lc} with t-norm {selected_tnorm}')

        # Construct constraint
        self.solver.constraintConstructor.current_device = self.solver.current_device
        self.solver.constraintConstructor.myGraph = self.solver.myGraph
        
        if isinstance(lc, sumL) and label is None:
             result = None
        else:
            lossList = self.solver.constraintConstructor.constructLogicalConstrains(
                lc, myBooleanMethods, None, dn, 0,
                key=key, headLC=True, loss=True, sample=False, **({'label': int(label)} if label is not None else {}))
        
        result['lossList'] = lossList

        # Reduce to single tensor
        lossTensor = self._compute_loss_tensor(lossList)
        result['loss'] = lossTensor

        if lossTensor is not None:
            result['conversionSigmoid'] = 1.0 - lossTensor
            if isinstance(lc, sumL) or (hasattr(lc, 'innerLC') and isinstance(lc.innerLC, sumL)):
                result['expectedCount'] = lossTensor
        else:
            result['conversionSigmoid'] = None

        result['elapsedInMsLC'] = (perf_counter_ns() - start) / 1_000_000
        return result

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

        dn.setActiveExecutableLCs()

        for graph in self.solver.myGraph:
            for _, lc in graph.allLogicalConstrains:
                result = self.calculate_single_lc_loss(lc, dn, key, tnorm, counting_tnorm, label=None)
                if result is not None:
                    lcLosses[lc.lcName] = result

        return lcLosses