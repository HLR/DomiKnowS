from .utils import Namespace


config = {
    'Data': { # data setting
        'relative_path': "data/EntityMentionRelation",
        'train_path': "conll04.corp_1_train.corp",
        'valid_path': "conll04.corp_1_test.corp"
    },
    'Model': { # model setting
        'embedding_dim': 8,
        'ngram': 5,
        'bidirectional': True,
        'dropout': 0.3,
        'pretrained_files': {
            'word': 'data/glove.6B/glove.6B.50d.txt'
        },
        'pretrained_dims': {
            'word': 50
        },
        'graph': {
            'balance_factor': 0.5,
            'label_smoothing': 0.1,
            'focal_gamma': 2.,
            'inference_interval': 10
        }
    },
    'Train': {
        'lr':0.001,
        'wd':0.0001,
        'batch': 8,
        'epoch': 100,
        'patience': None
    }
}

Config = Namespace(config)
