def seed():
    import random
    import numpy as np
    import torch

    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)


class Namespace(dict):
    def __init__(self, __dict={}, **kwargs):
        self.__dict__ = self
        self.update(__dict)
        self.update(kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = Namespace(v)

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, ','.join('\'{}\':{}'.format(k,v) for k,v in self.items()))
