import torch

from regr.solver.ilpOntSolverFactory import ilpOntSolverFactory

from regr.utils import Namespace, caller_source
from regr.program import LearningBasedProgram
from regr.program.model.pytorch import PoiModel


config = {
    'Data': {
    },
    'Model': {
        'program': {
            'Type': LearningBasedProgram,
            'Model': lambda graph: PoiModel(
                graph,
                Solver=lambda graph: ilpOntSolverFactory.getOntSolverInstance(graph)
                )
        }
    },
    'Source': {
        'emr': caller_source(),
    }
}

CONFIG = Namespace(config)
