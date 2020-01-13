import time
from utils import Namespace


config = {
    'Data': {  # data setting
        'relative_path': "data",
#         'train_path': "new_train.xml",
#         'valid_path': "new_gold.xml",
        'train_path': "sprl2017_train.xml",
        'valid_path': "sprl2017_gold.xml"
    },
    'Model': {  # model setting
        'embedding_dim': 8,
        'dropout': 0.5,
        'ngram': 5,
        'encode': {
            'layers': [64, 64],
        },
        'compact': 32,
        'graph': {
            'balance_factor': 1,
            'label_smoothing': 0.01,
            'focal_gamma': 2,
            'inference_interval': 50,
            'inference_training_set': False,
            'inference_loss': False,
            'log_solver': False,
            'soft_penalty': 0.
        }
    },
    'Train': {
        'pretrained_files': None,
        'trainer':{
            'num_epochs': 50,
            'patience': None,
            'serialization_dir': 'log.{}'.format(time.strftime("%Y%m%d-%H%M%S", time.gmtime())),
#             'serialization_dir':None,
        },
        'optimizer': {
            'type': 'adam',
            'lr': 0.005,
            'weight_decay': 0.001,
        },

        'scheduler': {
            'type': 'reduce_on_plateau',
            'patience': 10
        },
        'iterator': {
            'batch_size': 2,
        }
    }
}

Config = Namespace(config)