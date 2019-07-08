from .utils import Namespace


config = {
    'Data': { # data setting
        'relative_path': "data/EntityMentionRelation",
        'train_path': "conll04_train.corp",
        'valid_path': "conll04_test.corp"
    },
    'Model': { # model setting
        'embedding_dim': 8,
        'pretrained_files': {
            'word': 'data/glove.6B/glove.6B.50d.txt'
        },
        'pretrained_dims': {
            'word': 50
        }
    },
    'Train': {
        'lr':0.001,
        'wd':0.0001,
        'batch': 8,
        'epoch': 50,
        'patience': None
    }
}

Config = Namespace(config)
