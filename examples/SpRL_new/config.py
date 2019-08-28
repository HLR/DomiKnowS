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
            'balance_factor': 0.25,
            'label_smoothing': 0.1,
            'focal_gamma': 2.,
            'inference_interval': 50,
            'inference_training_set': False
        }
    },
    'Train': {
        'pretrained_files': None,
        'trainer':{
            'num_epochs': 50,
            'patience': None,
            'serialization_dir':None
        },
        'optimizer': {
            'type': 'adam',
            'lr': 0.001,
            'weight_decay': 0.0001
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