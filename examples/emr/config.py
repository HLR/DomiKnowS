import torch

from regr.solver.ilpOntSolverFactory import ilpOntSolverFactory

from emr.utils import Namespace, caller_source
from emr.graph.torch import TorchModel, PoiModel, IMLModel
from emr.graph.loss import BWithLogitsIMLoss, BCEFocalLoss, BCEWithLogitsLoss, BCEWithLogitsFocalLoss
from emr.graph.metric import MacroAverageTracker, PRF1Tracker
from emr.graph.solver import Solver


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
        'lbp': {
            'model': lambda graph: IMLModel(
                graph,
                loss=MacroAverageTracker(BWithLogitsIMLoss(0)),
                metric=PRF1Tracker(),
                solver_fn=lambda graph: ilpOntSolverFactory.getOntSolverInstance(graph, Solver)
                ),
            # 'model': lambda graph: TorchModel(
            #     graph,
            #     loss=MacroAverageTracker(BCEWithLogitsLoss()),
            #     metric=PRF1Tracker(),
            #     solver_fn=lambda graph: ilpOntSolverFactory.getOntSolverInstance(graph, Solver)
            #     )
        }
    },
    'Train': {
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'seed': 1,
        'opt': torch.optim.Adam,
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
