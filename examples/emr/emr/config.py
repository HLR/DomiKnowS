if __package__ is None or __package__ == '':
    from utils import Namespace, caller_source
else:
    from .utils import Namespace, caller_source

import time


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
        'dropout': 0.35,
        'pretrained_files': {
            'word': 'data/glove.6B/glove.6B.50d.txt'
        },
        'pretrained_dims': {
            'word': 50
        },
        'graph': {
            'balance_factor': 1.5,
            'label_smoothing': 0.01,
            'focal_gamma': 2,
            'inference_interval': 100,
            'inference_training_set': False
        }
    },
    'Train': {
        'pretrained_files': {
            'word': 'data/glove.6B/glove.6B.50d.txt'
        },
        'trainer': {
            'lr':0.0001,
            'wd':0.00001,
            'batch': 8,
            'epoch': 100,
            'patience': None,
            'serialization_dir': 'log.{}'.format(time.strftime("%Y%m%d-%H%M%S", time.gmtime())),
        }
    },
    'Source': {
        'emr': caller_source()
    }
}

Config = Namespace(config)
