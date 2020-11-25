import os

from regr.utils import Namespace, caller_source
from regr.program.model.pytorch import PoiModel


from .graph import sentence, word, phrase, pair, work_for, located_in, live_in, orgbase_on, kill, Oword, Bword, Iword, Eword, people, organization, other, location


config = {
    'Model': {
        'Model': PoiModel,
        'poi': (sentence, word, Oword, Bword, Iword, Eword, phrase, location, other, people, organization, pair, work_for, located_in, live_in, orgbase_on, kill ),
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
