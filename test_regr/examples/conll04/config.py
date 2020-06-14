import os

from regr.utils import Namespace, caller_source
from regr.program.model.pytorch import PoiModel


config = {
    'Model': {
        'Model': PoiModel,
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
