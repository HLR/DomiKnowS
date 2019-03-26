import abc


class Backend(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, a): pass

    @abc.abstractmethod
    def matmul(self, a1, a2): pass

    @abc.abstractmethod
    def sum(self, a): pass

    @abc.abstractmethod
    def log(self, a): pass

    @abc.abstractmethod
    def prod(self, a): pass

    @abc.abstractmethod
    def shape(self, a): pass

    @abc.abstractmethod
    def reshape(self, a, newshape): pass

    @abc.abstractmethod
    def flatten(self, a): pass

    @abc.abstractmethod
    def transpose(elf, a, axes): pass
