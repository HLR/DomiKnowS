def seed(s=1, deterministic=True):
    import os
    import random
    import numpy as np
    import torch

    os.environ['PYTHONHASHSEED'] = str(s)  # https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED
    np.random.seed(s)
    random.seed(s)
    torch.manual_seed(s)  # this function will call torch.cuda.manual_seed_all(s) also

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# http://code.activestate.com/recipes/117214-knuth-morris-pratt-string-matching/

class KnuthMorrisPratt():
    def __init__(self, pattern):
        self.pattern = pattern

    @property
    def pattern(self):
        return self._pattern

    @pattern.setter
    def pattern(self, pattern):
        # allow indexing into pattern and protect against change during yield
        pattern = list(pattern)
        self._pattern = pattern

        # build table of shift amounts
        shifts = [1] * (len(pattern) + 1)
        shift = 1
        for pos in range(len(pattern)):
            while shift <= pos and pattern[pos] != pattern[pos-shift]:
                shift += shifts[pos-shift]
            shifts[pos+1] = shift
        self._shifts = shifts

    @staticmethod
    def ele_ne(pattern_elm, other_elm):
        return pattern_elm != other_elm

    def __call__(self, inputs):
        # do the actual search
        startPos = 0
        matchLen = 0
        for elm in inputs:
            while matchLen == len(self._pattern) or \
                matchLen >= 0 and self.ele_ne(self._pattern[matchLen], elm):
                startPos += self._shifts[matchLen]
                matchLen -= self._shifts[matchLen]
            matchLen += 1
            if matchLen == len(self._pattern):
                yield startPos, startPos + matchLen
