from collections import Counter
from .base import SubScorable, local_names_and_objs


@local_names_and_objs
class Graph(SubScorable):
    # local tree structure
    default = None

    def __init__(self, name=None):
        SubScorable.__init__(self, name)
        self._concepts = []

    @property
    def concepts(self):
        return self._concepts

    def what(self):
        wht = SubScorable.what(self)
        wht.extend({'concepts': self.concept})
        return wht

    def release(self, prop=None):
        for concept in self.concept:
            concept.release(prop)
        for sub in self.subs:
            sub.release(prop)

    def score(self, val):
        return sum([concept(prop) for concept in self.concept])

    def __getattr__(self, name):
        for sub in self.subs:
            if sub.name == name:
                return sub
        for concept in self.concepts:
            if concept.name == name:
                return concept

    def __getitem__(self, name):
        tokens = name.split('/', 2)
        if len(tokens) > 1:
            return self.subs[tokens[0]].retrieve(tokens[1])
        for concept in self.concepts:
            if concept.name == tokens[0]:
                return concept
        return None
