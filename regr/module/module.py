import abc

class Module(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self) -> Dict: pass
