import os

from emr.utils import Namespace, caller_source

from emr.graph.metric import BCEWithLogitsLoss, MacroAverageTracker, PRF1Tracker


config = {
    'Data': {
        'train_path': os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/EntityMentionRelation/conll04.corp"),
        'skip_none': False,
    },
    'Model': {
        'loss': MacroAverageTracker(BCEWithLogitsLoss()),
        'metric': PRF1Tracker()
    },
    'Train': {
        'batch_size': 1,
    },
    'Source': {
        'emr': caller_source(),
    }
}

Config = Namespace(config)
