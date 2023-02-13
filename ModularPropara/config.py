import os

from regr.utils import Namespace, caller_source
from regr.program.model.pytorch import PoiModel


from .graph import graph, procedure, word, step, entity, entity_step, entity_step_word, location_start, location_end, non_existence, unknown_loc, known_loc


config = {
    'Model': {
        'Model': PoiModel,
        'poi': (entity_step, entity_step_word, location_start, location_end, non_existence, unknown_loc, known_loc),
        'loss': None,
        'metric': None,
    },
    'Train': {
        'batch_size': 1,
    },
}

CONFIG = Namespace(config)
