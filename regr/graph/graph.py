from collections import OrderedDict
if __package__ is None or __package__ == '':
    from base import AutoNamed, NamedTree
else:
    from .base import AutoNamed, NamedTree

@AutoNamed.localize_namespace
class Graph(AutoNamed, NamedTree):
    # local tree structure
    default = None
    
    def __init__(self, name=None, ontology=None):
        AutoNamed.__init__(self, name) # name may be update
        NamedTree.__init__(self, self.name)
        self.ontology = ontology
        self._concepts = OrderedDict()

    def get_apply(self, name):
        if name in self.concepts:
            return self.concepts[name]
        return NamedTree.get_apply(self, name)

    def set_apply(self, name, sub):
        if __package__ is None or __package__ == '':
            from concept import Concept
        else:
            from .concept import Concept
        # TODO: what if a concept has same name with a subgraph?
        if isinstance(sub, Graph):
            NamedTree.set_apply(self, name, sub)
        elif isinstance(sub, Concept):
            self.concepts[name] = sub

    @property
    def concepts(self):
        return self._concepts

    def what(self):
        wht = NamedTree.what(self)
        wht['concepts'] = dict(self.concepts)
        return wht
