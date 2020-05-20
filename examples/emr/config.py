import torch

from regr.solver.ilpOntSolverFactory import ilpOntSolverFactory
from regr.program import LearningBasedProgram
from regr.utils import Namespace, caller_source

from emr.program import PrimalDualLearningBasedProgram
from emr.graph.torch import SolverModel, IMLModel
from emr.graph.loss import BWithLogitsIMLoss, BCEFocalLoss, BCEWithLogitsLoss, BCEWithLogitsFocalLoss
from emr.graph.metric import MacroAverageTracker, PRF1Tracker
from emr.solver.solver import Solver


lbps = {
    'nll': {
        'type': LearningBasedProgram,
        'model': lambda graph: SolverModel(
            graph,
            loss=MacroAverageTracker(BCEWithLogitsLoss()),
            metric=PRF1Tracker(),
            Solver=lambda graph: ilpOntSolverFactory.getOntSolverInstance(graph, Solver))},
    'iml': {
        'type': LearningBasedProgram,
        'model': lambda graph: IMLModel(
            graph,
            loss=MacroAverageTracker(BWithLogitsIMLoss(0)),
            metric=PRF1Tracker(),
            Solver=lambda graph: ilpOntSolverFactory.getOntSolverInstance(graph, Solver))},
    'primal-dual': {
        'type': PrimalDualLearningBasedProgram,
        'model': lambda graph: SolverModel(
            graph,
            loss=MacroAverageTracker(BCEWithLogitsLoss()),
            metric=PRF1Tracker(),
            Solver=lambda graph: ilpOntSolverFactory.getOntSolverInstance(graph, Solver))},
}

config = {
    'Data': {
        'train_path': "data/EntityMentionRelation/conll04.corp_1_train.corp",
        'valid_path': "data/EntityMentionRelation/conll04.corp_1_test.corp",
        'skip_none': False,
    },
    'Model': {
        'word': {
            'emb': {
                'embedding_dim': 50
            },
            'ctx_emb': {
                'input_size': 50,
                'hidden_size': 100,
                'num_layers': 2,
                'batch_first': True,
                'bidirectional': True
            },
            'feature': {
                'in_features': 200,
                'out_features': 200
            },
            'lr': {
                'in_features': 200,
            }
        },
        'pair': {
            'feature': {
                'in_features': 400,
                'out_features': 200
            },
            'lr': {
                'in_features': 200,
            }
        },
        'lbp': lbps['nll']
    },
    'Train': {
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'seed': 1,
        'opt': torch.optim.Adam,
        'copt': torch.optim.Adam,
        'batch_size': 8,
        'epoch': 10,
        'train_inference': True,
        'valid_inference': True
    },
    'Source': {
        'emr': caller_source(),
    }
}

CONFIG = Namespace(config)
