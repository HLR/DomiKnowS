from .learningbaseprogram import Program
from .model.pytorch import PoiModel, IMLModel, PoiModelToWorkWithLearnerWithLoss


class POIProgram(Program):
    def __init__(self, graph, **kwargs):
        super().__init__(graph, PoiModel, **kwargs)


class IMLProgram(Program):
    def __init__(self, graph, **kwargs):
        super().__init__(graph, IMLModel, **kwargs)


class POILossProgram(Program):
    def __init__(self, graph, poi=None):
        super().__init__(graph, PoiModelToWorkWithLearnerWithLoss, poi=poi)
