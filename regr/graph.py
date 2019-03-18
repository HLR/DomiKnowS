from .base import AutoNamed
from .entity import Entity

'''
A graph contain entities and their corresponding relationship
'''
class Graph(AutoNamed, list):
    '''
    '''
    def __init__(self, name=None):
        AutoNamed.__init__(self, name)
        list.__init__(self)
        self._sub = []
        self._super = None

    @property
    def _repr_scope(self):
        # claim repr_scope at instance level
        return self
    
    @property
    def sub(self):
        return self._sub
    
    @property
    def super(self):
        return self._super
        
    @super.setter
    def super(self, super):
        # TODO resolve and prevent recursive definition
        if super is not None:
            super._sub.append(self)
            self._super = super
            
    def __enter__(self):
        self.super = Entity.default_graph
        Entity.default_graph = self
        if self.super is not None:
            self.super
        return self
    
    def __exit__(self, type, value, traceback):
        Entity.default_graph = self._super
        
    def what(self):
        return {'super': self.super,
                'entities': list(self),
                'subgraphs': self.sub}
    
    def __call__(self, depth=float('inf')):
        cost = 0
        for entity in self:
            cost += entity()
        if depth > 0:
            for sub in self.sub:
                cost += sub(depth=depth-1)
        return cost