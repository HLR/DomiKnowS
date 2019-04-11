from collections import Counter
from .base import AutoNamed, local_names_and_objs


@local_names_and_objs
class Graph(SubScorable):
    # local tree structure
    default = None

    def __init__(self, name=None):
        SubScorable.__init__(self, name)
        self._concept = []

    @property
    def concept(self):
        return self._concept

    def what(self):
        wht = SubScorable.what(self)
        wht.extend({'concepts': self.concept})
        return wht

    def release(self, prop=None):
        for concept in self.concept:
            concept.release(prop)
        for graph in self.subs:
            graph.release(prop)

    def score(self, val):
        return sum([concept(prop) for concept in self.concept])
