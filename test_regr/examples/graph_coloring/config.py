import torch

from regr.utils import Namespace, caller_source
from regr.program import LearningBasedProgram
from regr.program.model.pytorch import PoiModel


config = {
    'Data': {
    },
    'Model': {
        'program': {
            'Type': LearningBasedProgram,
            'Model': PoiModel,
        }
    },
    'Source': {
        'emr': caller_source(),
    }
}

CONFIG = Namespace(config)
