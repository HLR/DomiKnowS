import os

from domiknows.utils import Namespace, caller_source
from domiknows.program.model.pytorch import PoiModel


from .graph import sentence, word, char, phrase, pair


config = {
    'Model': {
        'Model': PoiModel,
        'poi': (sentence, word, char, phrase, pair),
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
