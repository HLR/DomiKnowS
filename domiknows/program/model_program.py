from domiknows.utils import setDnSkeletonMode
from .program import LearningBasedProgram
from .model.pytorch import PoiModel, PoiModelToWorkWithLearnerWithLoss, SolverModel, SolverModelDictLoss
from .model.iml import IMLModel

class POIProgram(LearningBasedProgram):
    def __init__(self, graph, **kwargs):
        super().__init__(graph, PoiModel, **kwargs)


class SolverPOIProgram(LearningBasedProgram):
    def __init__(self, graph, **kwargs):
        super().__init__(graph, SolverModel, **kwargs)
        # Check if the 'inferTypes' is in the kwargs and has GBI then setDnSkeletonMode(True)
        if 'inferTypes' in kwargs:         
            if 'GBI' in kwargs['inferTypes']:
                setDnSkeletonMode(True)
        
        
class SolverPOIDictLossProgram(LearningBasedProgram):
    def __init__(self, graph, **kwargs):
        super().__init__(graph, SolverModelDictLoss, **kwargs)


class IMLProgram(LearningBasedProgram):
    def __init__(self, graph, **kwargs):
        super().__init__(graph, IMLModel, **kwargs)


class POILossProgram(LearningBasedProgram):
    def __init__(self, graph, poi=None):
        super().__init__(graph, PoiModelToWorkWithLearnerWithLoss, poi=poi)
