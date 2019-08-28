import time
from utils import Namespace


config = {
    'Data': { # data setting
        'relative_path': "data",
        'train_path': "sprl2017_train.xml",
        'valid_path': "sprl2017_gold.xml"
    },
    'Model': {  # model setting
        'embedding_dim': 8,
        'ngram': 5,
        'dropout': 0.5,
        'graph': {
            'balance_factor': 0,
            'label_smoothing': 0.01,
            'focal_gamma': 0,
            'inference_interval': 50,
            'inference_training_set': False
        },
        'pretrained_files':{
            'word': 'data/glove.6B.50d.txt'
        },
        'pretrained_dims': {
            'word': 50
        },
    },
    'Train': {
        'pretrained_files':{
            'word': 'data/glove.6B.50d.txt'
        },
        'trainer':{
            'num_epochs': 50,
            'patience': None,
            'serialization_dir':'log.{}'.format(time.strftime("%Y%m%d-%H%M%S", time.gmtime())),
        },
        'optimizer': {
            'type': 'adam',
            'lr': 0.001,
            'weight_decay': 0.0001,
        },

        'scheduler': {
            'type': 'reduce_on_plateau',
            'patience': 10
        },
        'iterator': {
            'batch_size': 8,
        }
    }
}

Config = Namespace(config)