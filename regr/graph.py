from collections import Counter
from .base import AutoNamed, local_names_and_objs

'''
A graph contains concepts
'''


@local_names_and_objs
class Graph(AutoNamed):
    default_graph = None

    '''
    '''

    def __init__(self, name=None):
        AutoNamed.__init__(self, name)
        self._concept = []
        self._sub = []
        self._super = None

    @property
    def scope_key(self):
        # claim recursive check at instance level
        return self

    @property
    def concept(self):
        return self._concept

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
                'concepts': self.concept,
                'subgraphs': self.sub}

    def release(self, prop=None):
        for concept in self.concept:
            concept.release(prop)
        for graph in self.sub:
            graph.release(prop)

    def __call__(self, prop=None, depth=float('inf')):
        cost = 0
        for concept in self.concept:
            cost += concept(prop)
        if depth > 0:  # TODO: why would I need depth?
            for sub in self.sub:
                cost += sub(prop, depth=depth - 1)
        return cost
