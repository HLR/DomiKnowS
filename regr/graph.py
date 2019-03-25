from collections import Counter
from .base import AutoNamed, local_names_and_objs

'''
A graph contains entities
'''


@local_names_and_objs
class Graph(AutoNamed):
    default_graph = None

    '''
    '''

    def __init__(self, name=None):
        AutoNamed.__init__(self, name)
        self._ent = []
        self._sub = []
        self._super = None

    @property
    def scope_key(self):
        # claim recursive check at instance level
        return self

    @property
    def ent(self):
        return self._ent

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
        self.super = Graph.default_graph
        Graph.default_graph = self
        if self.super is not None:
            self.super
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        Graph.default_graph = self._super

    def what(self):
        return {'supergraph': self.super,
                'entities': self.ent,
                'subgraphs': self.sub}

    def __call__(self, depth=float('inf')):
        cost = 0
        for entity in self.ent:
            cost += entity()
        if depth > 0:
            for sub in self.sub:
                cost += sub(depth=depth - 1)
        return cost
