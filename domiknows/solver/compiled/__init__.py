"""Ahead-of-time planned logical-constraint evaluation.

Formula and candidate-path plans persist on the solver across batches, while a
lightweight binding context supplies each DataNode's current topology and
prediction tensors. The plans are shared by fuzzy loss, sampling, circuit/WMC,
verification, ILP, and executable inference. Enable with ``compile_lc=True``
on a program, or ``compiled=True`` on the corresponding DataNode API.
"""

from .grounding import ProbabilityStore
from .formula import (
    CompiledConstraintEvaluator,
    CompiledLossCalculator,
    CompiledModeExecutor,
    lc_tree_supported,
    SUPPORTED_LC_TYPES,
)
from .plan import (
    BatchedUnaryImplicationPlan,
    CandidatePlan,
    CompiledFormulaPlan,
    CompiledPlanCache,
    TensorizedCandidateResolver,
)

__all__ = [
    'ProbabilityStore',
    'CompiledConstraintEvaluator',
    'CompiledLossCalculator',
    'CompiledModeExecutor',
    'lc_tree_supported',
    'SUPPORTED_LC_TYPES',
    'BatchedUnaryImplicationPlan',
    'CandidatePlan',
    'CompiledFormulaPlan',
    'CompiledPlanCache',
    'TensorizedCandidateResolver',
]
