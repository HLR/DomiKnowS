import abc
from typing import NoReturn

class Compiler(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compile(self, src:str, dst:str) -> NoReturn: pass
