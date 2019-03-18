import numpy as np
from .base import Backend


class NumpyBackend(Backend):

    def array(self, a):
        return np.array(a)

    def matmul(self, a1, a2):
        return np.matmul(a1, a2)

    def sum(self, a):
        return np.sum(a)

    def log(self, a):
        return np.log(a)

    def prod(self, a):
        return np.prod(a)

    def shape(self, a):
        return np.shape(a)

    def reshape(self, a, newshape):
        return np.reshape(a, newshape)

    def flatten(self, a):
        return np.ndarray.flatten(a)

    def transpose(self, a, axes):
        return np.transpose(a, axes)
    
    def norm(self, a):
        return np.linalg.norm(a)
