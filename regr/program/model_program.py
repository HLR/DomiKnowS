from .program import LearningBasedProgram
from .model.pytorch import PoiModel, IMLModel, PoiModelToWorkWithLearnerWithLoss


class POIProgram(LearningBasedProgram):
    def __init__(self, graph, **kwargs):
        super().__init__(graph, PoiModel, **kwargs)


class IMLProgram(LearningBasedProgram):
    def __init__(self, graph, **kwargs):
        super().__init__(graph, IMLModel, **kwargs)


class POILossProgram(LearningBasedProgram):
    def __init__(self, graph, poi=None):
        super().__init__(graph, PoiModelToWorkWithLearnerWithLoss, poi=poi)
