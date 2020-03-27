from emr.utils import Namespace, caller_source

from emr.graph.metric import BCEWithLogitsLoss, MacroAverageTracker, PRF1Tracker


config = {
    'Data': {
        'train_path': "data/EntityMentionRelation/conll04.corp_1_train.corp",
        'valid_path': "data/EntityMentionRelation/conll04.corp_1_test.corp",
        'skip_none': False,
    },
    'Model': {
        'loss': MacroAverageTracker(BCEWithLogitsLoss()),
        'metric': PRF1Tracker()
    },
    'Train': {
        'batch_size': 8,
    },
    'Source': {
        'emr': caller_source(),
    }
}

Config = Namespace(config)
