from regr.program.model.pytorch import PoiModel, IMLModel
from regr.program.loss import BCEWithLogitsLoss, BCEWithLogitsIMLoss
from regr.program.metric import MacroAverageTracker, ValueTracker
from regr.solver.ilpOntSolverFactory import ilpOntSolverFactory


class Model(IMLModel):
    def __init__(self, graph):
        super().__init__(
            graph, 
            loss=MacroAverageTracker(BCEWithLogitsIMLoss(0.5)),
            metric=ValueTracker(lambda pr, gt: pr.data),
            Solver=ilpOntSolverFactory.getOntSolverInstance)
