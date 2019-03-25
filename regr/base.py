from collections import Counter, defaultdict
from contextlib import contextmanager
from threading import Lock
import pprint

class Scoped(object):
    '''
    A lock that count all trials of acquiring and require same number of releasing to really unlock.
    Who else need this...? Well, let me leave it nested inside then...
    '''
    class LevelLock(object):
        # note: threading.Lock is a `builtin_function_or_method`, not a class!
        def __init__(self):
            self.__lock = Lock()
            self.__level = 0
            
        @property
        def level(self):
            return self.__level
        
        def acquire(self, blocking=False): # in this setting, you don't usually want blocking
            retval = self.__lock.acquire(blocking)
            self.__level += 1
            return retval
        
        def release(self):
            self.__level -= 1
            if self.__level == 0:
                self.__lock.release()
                
        def __repr__(self):
            return repr(self.__level)
        
    __locks = defaultdict( LevelLock )
    
    def __init__(self, blocking=False):
        self._blocking = blocking
    
    @contextmanager
    def scope(self, blocking=None):
        if blocking is None:
            blocking = self._blocking
        lock = self.__context.__locks[self.scope_key]
        try:
            yield lock.acquire(self._blocking)
        finally:
            lock.release()
            
    @property
    def scope_key(self):
        return type(self)
    

Scoped._Scoped__context = Scoped

class Named(Scoped):
    def __init__(self, name):
        Scoped.__init__(self, blocking=False)
        self.name = name
    
    def __repr__(self):
        cls = type(self)
        with self.scope() as need_detail:
            if need_detail and callable(getattr(self, 'what', None)):
                repr_str = '{class_name}(name=\'{name}\', what={what})'.format(class_name=cls.__name__,
                                                                               name=self.name,
                                                                               what=pprint.pformat(self.what(), width=8))
            else:
                repr_str = '{class_name}(name=\'{name}\')'.format(class_name=cls.__name__,
                                                                  name=self.name)
        return repr_str

def local_names_and_objs(cls):
    cls._names = Counter()
    cls._objs = dict()
    return cls
    
@local_names_and_objs
class AutoNamed(Named):
    @classmethod
    def clear(cls):
        cls._names.clear()
        cls._objs.clear()

    @classmethod
    def get(cls, name, value=None):
        return _objs.get(name, value)
        
    @classmethod
    def suggest_name(cls):
        return cls.__name__.lower()
    
    def assign_suggest_name(self, name=None):
        cls = type(self)
        if name is None:
            name = cls.suggest_name()
        if cls._names[name] > 0:
            while True:
                name_attempt = '{}-{}'.format(name, cls._names[name])
                cls._names[name] += 1
                if cls._names[name_attempt] == 0:
                    name = name_attempt
                    break # while True
        assert cls._names[name] == 0
        cls._names[name] += 1
        self.name = name
        cls._objs[name] = self

    def __init__(self, name=None):
        Named.__init__(self, name) # temporary name may apply
        self.assign_suggest_name(name)
