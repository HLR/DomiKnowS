from collections import OrderedDict
if __package__ is None or __package__ == '':
    from base import SubScorable
else:
    from .base import SubScorable

@SubScorable.localize_namespace
class Graph(SubScorable):
    # local tree structure
    default = None
    
    def __init__(self, name=None, ontology=None):
        SubScorable.__init__(self, name)
        self.ontology = ontology
        self._concepts = OrderedDict()

    def add(self, obj):
        if __package__ is None or __package__ == '':
            from concept import Concept
        else:
            from .concept import Concept

        if isinstance(obj, Graph):
            self._subs[obj.name] = obj
        elif isinstance(obj, Concept):
            self._concepts[obj.name] = obj
        else:
            raise TypeError('Add Graph or Concept instance, {} instance given.'.format(type(obj)))

    @property
    def subs(self):
        return self._subs + self._concepts

    @property
    def concepts(self):
        return self._concepts

    def what(self):
        wht = SubScorable.what(self)
        wht['concepts'] = dict(self.concepts)
        return wht

    def release(self, prop=None):
        for sub in self.subs.values():
            sub.release(prop)

    def get_multiassign(self):
        for sub in self.subs.values():
            for graph, concept, prop, value in sub.get_multiassign():
                yield graph, concept, prop, value

    def score(self, *args, **kwargs):
        subscore = SubScorable.score(self, *args, **kwargs)
        return subscore + sum([concept(*args, **kwargs) for concept in self.concepts.values()])

