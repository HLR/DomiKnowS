from .program import LearningBasedProgram
from .model_program import POIProgram, IMLProgram, POILossProgram, SolverPOIProgram, SolverPOIDictLossProgram
from .callbackprogram import CallbackProgram
from .lossprogram import (
    PrimalDualModel,
    SampleLossModel,
    SemanticLossModel,
    SemanticLossProgram,
    GBIProgram,
)


def __getattr__(name):
    # Lazy import to avoid a circular import: domiknows.reinforcement imports
    # LearningBasedProgram from this package, so it must be importable as
    # `from domiknows.program import ReinforcementProgram` without pulling the
    # reinforcement package in at program-package load time.
    if name == "ReinforcementProgram":
        from ..reinforcement import ReinforcementProgram
        return ReinforcementProgram
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
