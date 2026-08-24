"""DomiKnowS baseline for the Embodied Agent Interface benchmark."""

from .vlabench_auxiliary import (
    AuxiliaryTrainingResult,
    prepare_vlabench_text_examples,
    train_vlabench_text_auxiliary,
)

__all__ = [
    "AuxiliaryTrainingResult",
    "prepare_vlabench_text_examples",
    "train_vlabench_text_auxiliary",
]

