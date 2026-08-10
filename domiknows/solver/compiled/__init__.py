"""Compiled (batched-gather) evaluation of logical-constraint losses.

Drop-in alternative to the per-datanode interpreter used by
``DataNode.calculateLcLoss``; enable with ``compile_lc=True`` on
``PrimalDualProgram`` (or any ``LossProgram`` whose CModel forwards it).
"""

from .grounding import ProbabilityStore
from .formula import (
    CompiledConstraintEvaluator,
    CompiledLossCalculator,
    lc_tree_supported,
    SUPPORTED_LC_TYPES,
)

__all__ = [
    'ProbabilityStore',
    'CompiledConstraintEvaluator',
    'CompiledLossCalculator',
    'lc_tree_supported',
    'SUPPORTED_LC_TYPES',
]
