from regr.program.model.pytorch import PoiModel, IMLModel
from regr.program.loss import BCEWithLogitsLoss, BCEWithLogitsIMLoss
from regr.program.metric import MacroAverageTracker, ValueTracker
from regr.solver.ilpOntSolverFactory import ilpOntSolverFactory


class MyIMLModel(IMLModel):
    def __init__(self, graph):
        super().__init__(
            graph, 
            loss=MacroAverageTracker(BCEWithLogitsIMLoss(0.5)),
            metric=ValueTracker(lambda pr, gt: pr.data),
            Solver=ilpOntSolverFactory.getOntSolverInstance)

class MyModel(PoiModel):
    def __init__(self, graph):
        super().__init__(
            graph, 
            loss=MacroAverageTracker(BCEWithLogitsLoss),
            metric=ValueTracker(lambda pr, gt: pr.data))
