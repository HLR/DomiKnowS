import torch

from regr.solver.ilpOntSolverFactory import ilpOntSolverFactory
from regr.utils import Namespace, caller_source
from regr.program.metric import MacroAverageTracker, BinaryPRF1Tracker
from regr.program.loss import BCEWithLogitsIMLoss, BCEFocalLoss, BCEWithLogitsLoss, BCEWithLogitsFocalLoss
from regr.program.program import LearningBasedProgram
from regr.program.primaldual import PrimalDualLearningBasedProgram
from regr.program.model.torch import SolverModel
from regr.program.model.torch.iml_model import IMLModel
from regr.solver.contextsolver.torch import Solver


lbps = {
    'nll': {
        'type': LearningBasedProgram,
        'model': lambda graph: SolverModel(
            graph,
            loss=MacroAverageTracker(BCEWithLogitsLoss()),
            metric=BinaryPRF1Tracker(),
            Solver=lambda graph: ilpOntSolverFactory.getOntSolverInstance(graph, Solver))},
    'iml': {
        'type': LearningBasedProgram,
        'model': lambda graph: IMLModel(
            graph,
            loss=MacroAverageTracker(BCEWithLogitsIMLoss(0)),
            metric=BinaryPRF1Tracker(),
            Solver=lambda graph: ilpOntSolverFactory.getOntSolverInstance(graph, Solver))},
    'primal-dual': {
        'type': PrimalDualLearningBasedProgram,
        'model': lambda graph: SolverModel(
            graph,
            loss=MacroAverageTracker(BCEWithLogitsLoss()),
            metric=BinaryPRF1Tracker(),
            Solver=lambda graph: ilpOntSolverFactory.getOntSolverInstance(graph, Solver))},
}

config = {
    'seed': 1,
    'Data': {
        'train_path': "data/EntityMentionRelation/conll04.corp_1_train.corp",
        'valid_path': "data/EntityMentionRelation/conll04.corp_1_test.corp",
        'skip_none': False,
        'batch_size': 8,
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
        'Optim': torch.optim.Adam,
        # 'COptim': torch.optim.Adam,
        'train_epoch_num': 100,
        # 'train_inference': False,
        # 'valid_inference': True
    },
    'Source': {
        'emr': caller_source(),
    }
}

CONFIG = Namespace(config)
