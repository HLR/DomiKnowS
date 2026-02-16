"""
Loss Calculator with unified T-Norm selection via TNormSelector.

Uses TNormSelector from adaptiveTNormLossCalculator for all t-norm decisions.
Supports three modes via the tnorm parameter:
  - Specific ("G","P","L","SP"): Use that t-norm for all constraints.
  - "default": Per-type optimal defaults.
  - "auto": Dynamic adaptation via AdaptiveTNormLossCalculator.
"""

from time import perf_counter_ns
import torch
from typing import Dict, Optional

from domiknows.graph import fixedL
from domiknows.graph.logicalConstrain import sumL

from domiknows.solver.adaptiveTNormLossCalculator import (TNormSelector, get_constraint_type)

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
    Loss calculator with per-constraint t-norm selection via TNormSelector.
    """

    def __init__(self, solver, tnorm_selector: Optional[TNormSelector] = None):
        """
        Args:
            solver: Reference to gurobiILPOntSolver instance.
            tnorm_selector: Optional pre-configured TNormSelector. If None, one is created based tnorm and counting_tnorm parameters in calculateLoss.
        """
        self.solver = solver
        self._external_selector = tnorm_selector

    def _compute_loss_tensor(self, lossList):
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

    def calculate_single_lc_loss(self, lc, dn, key, tnorm='L', counting_tnorm=None, label=None):
        """
        Calculate loss for a single logical constraint.

        Args:
            lc: The logical constraint.
            dn: Data node.
            key: Softmax key.
            tnorm: A t-norm string ("L","P","SP","G","default", "auto").
            counting_tnorm: Deprecated —  but stil accepted for backward compatibility. Ignored if tnorm is a auto.
            label: Optional label for sumL constraints.
        """
        if not lc.headLC or not lc.active:
            return None
        if type(lc) is fixedL:
            return None

        selector =  self._external_selector if self._external_selector is not None else TNormSelector(tnorm, counting_tnorm)

        start = perf_counter_ns()
        myBooleanMethods = self.solver.myLcLossBooleanMethods
        result = {'lc': lc}

        selected_tnorm = selector.select(lc=lc)
        constraint_type = get_constraint_type(lc)
        result['tnorm_used'] = selected_tnorm
        result['constraint_type'] = constraint_type

        myBooleanMethods.setTNorm(selected_tnorm)

        if isinstance(lc, sumL) and label is None:
            label = dn.getExecutableConstraintLabel(lc.lcName).float()
            if label is None:
                return None

        self.solver.myLogger.info(f'Processing {lc} with t-norm {selected_tnorm}')

        self.solver.constraintConstructor.current_device = self.solver.current_device
        self.solver.constraintConstructor.myGraph = self.solver.myGraph

        if isinstance(lc, sumL) and label is None:
            result = None
        else:
            lossList = self.solver.constraintConstructor.constructLogicalConstrains(
                lc, myBooleanMethods, None, dn, 0,
                key=key, headLC=True, loss=True, sample=False,
                **({'label': int(label)} if label is not None else {}))

        result['lossList'] = lossList

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
        Calculate loss with per-constraint t-norm selection.

        Args:
            dn: Data node.
            tnorm: T-norm mode — "L","P","SP","G","default", or "auto".
            counting_tnorm: Deprecated — depricated but still accepted for backward compatibility. Ignored if tnorm is "auto".
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
                result = self.calculate_single_lc_loss(lc, dn, key, tnorm=tnorm, counting_tnorm=counting_tnorm, label=None)
                if result is not None:
                    lcLosses[lc.lcName] = result

        return lcLosses