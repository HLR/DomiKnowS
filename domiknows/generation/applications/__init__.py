"""Application-level generation controllers and domain adapters."""

from .hybrid import (
    CandidateScore,
    CompactConstraintSelector,
    ConstraintBundle,
    GenerationCandidate,
    HMMDFADecodeResult,
    HybridController,
    HybridScoreWeights,
    ManualConstraintSelector,
    ScoredCandidate,
    preference_pair_ranking_loss,
)
from .hmm_dfa_decoder import HMMDFADecoder
from .adapters import GenerationResult, HuggingFaceGenerationAdapter, OpenAIResponsesAdapter
from .planning import (
    PlanningBundle,
    decode_plan,
    encode_plan,
    planning_bundle_from_graph,
    planning_dfa_from_graph,
    planning_hmm_masks_from_graph,
    reference_plans_from_graph,
)
from .inference import (
    beam_label_inference,
    greedy_label_inference,
    sample_label_inference,
)

__all__ = [
    "CandidateScore",
    "CompactConstraintSelector",
    "ConstraintBundle",
    "GenerationCandidate",
    "GenerationResult",
    "HMMDFADecoder",
    "HMMDFADecodeResult",
    "HybridController",
    "HybridScoreWeights",
    "HuggingFaceGenerationAdapter",
    "ManualConstraintSelector",
    "PlanningBundle",
    "OpenAIResponsesAdapter",
    "ScoredCandidate",
    "beam_label_inference",
    "decode_plan",
    "encode_plan",
    "greedy_label_inference",
    "planning_bundle_from_graph",
    "planning_dfa_from_graph",
    "planning_hmm_masks_from_graph",
    "preference_pair_ranking_loss",
    "reference_plans_from_graph",
    "sample_label_inference",
]
