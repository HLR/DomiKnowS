import time
from utils import Namespace

log_dir = 'log.{}'.format(time.strftime("%Y%m%d-%H%M%S", time.gmtime()))
# log_dir = None

config = {
    'Data': {  # data setting
        'relative_path': "data",
        # 'train_path': "new_train.pkl",
        # 'valid_path': "new_gold.pkl",
        # 'train_path': "new_train.xml",
        # 'valid_path': "new_gold.xml",
        # 'train_path': "sprl2017_train.xml.pkl",
        # 'valid_path': "sprl2017_gold.xml.pkl",
        'train_path': "sprl2017_train.xml",
        'valid_path': "sprl2017_gold.xml"
    },
    'Model': {  # model setting
        'embedding_dim': 16,
        'dropout': 0.5,
        'ngram': 5,
        'encode': {
            'layers': [64, 64],
        },
        'compact': 64,
        'graph': {
            'balance_factor': 1,
            'label_smoothing': 0.01,
            'focal_gamma': 2,
            'inference_interval': 1,
            'inference_training_set': True,
            'inference_loss': True,
            'log_solver': True,
            'soft_penalty': 0.6
        },
        'log_dir': log_dir,
    },
    'Train': {
        'pretrained_files': None,
        'trainer':{
            'num_epochs': 50,
            'patience': None,
            'serialization_dir': log_dir
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