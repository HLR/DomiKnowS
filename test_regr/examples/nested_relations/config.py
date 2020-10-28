import os

from regr.utils import Namespace, caller_source
from regr.program.model.pytorch import PoiModel


from .graph import sentence, word, phrase, pair


config = {
    'Model': {
        'Model': PoiModel,
        'poi': (sentence, word, phrase, pair),
        'loss': None,
        'metric': None,
    },
    'Train': {
        'batch_size': 1,
    },
    'Source': {
        'emr': caller_source(),
    }
}

CONFIG = Namespace(config)
