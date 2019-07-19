from utils import Namespace



config = {
    'Data': { # data setting
        'relative_path': "data",
        'train_path': "sprl2017_train.xml",
        'valid_path': "sprl2017_gold.xml"
    },
    'Model': {  # model setting
        'embedding_dim': 8,
        'pretrained_files': None,
        'pretrained_dims': {
            'word': 50
        },
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
            'lr':0.001,
            'wd':0.0001,
            'batch': 8,
            'epoch': 50,
            'patience': None,
            'serialization_dir':None
        }
    }
}

Config = Namespace(config)