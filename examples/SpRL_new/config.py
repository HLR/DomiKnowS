from utils import Namespace


config = {
    'Data': { # data setting
        'relative_path': "data",
        'train_path': "sprl2017_train.xml",
        'valid_path': "sprl2017_gold.xml"
    },
    'Model': { # model setting
        'embedding_dim': 8
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