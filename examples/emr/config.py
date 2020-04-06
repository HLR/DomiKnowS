import torch

from regr.solver.ilpOntSolverFactory import ilpOntSolverFactory

from emr.utils import Namespace, caller_source
from emr.graph.metric import BCEFocalLoss, MacroAverageTracker, PRF1Tracker
from emr.graph.solver import Solver


config = {
    'Data': {
        'train_path': "data/EntityMentionRelation/conll04.corp_1_train.corp",
        'valid_path': "data/EntityMentionRelation/conll04.corp_1_test.corp",
        'skip_none': False,
    },
    'Model': {
        'loss': MacroAverageTracker(BCEFocalLoss(gamma=2, alpha=1)),
        'metric': PRF1Tracker(),
        'solver_fn': lambda graph: ilpOntSolverFactory.getOntSolverInstance(graph, Solver)
    },
    'Train': {
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'seed': 1,
        'opt': torch.optim.Adam,
        'batch_size': 8,
        'epoch': 10,
        'train_inference': False,
        'valid_inference': True
    },
    'Source': {
        'emr': caller_source(),
    }
}

CONFIG = Namespace(config)
