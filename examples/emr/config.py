from emr.utils import Namespace, caller_source

config = {
    'Data': {
        'train_path': "data/EntityMentionRelation/conll04.corp_1_train.corp",
        'valid_path': "data/EntityMentionRelation/conll04.corp_1_test.corp"
    },
    'Model': {
    },
    'Train': {
    },
    'Source': {
        'emr': caller_source()
    }
}

Config = Namespace(config)
