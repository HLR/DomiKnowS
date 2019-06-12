from collections import OrderedDict
if __package__ is None or __package__ == '':
    from base import BaseGraphTree
else:
    from .base import BaseGraphTree


@BaseGraphTree.localize_namespace
class Graph(BaseGraphTree):
    def __init__(self, name=None, ontology=None):
        BaseGraphTree.__init__(self, name)
        self.ontology = ontology
        self._concepts = OrderedDict()

    def __iter__(self):
        yield from BaseGraphTree.__iter__(self)
        yield from self._concepts

    @staticmethod
    def get_apply(self, name):
        if name in self.concepts:
            return self.concepts[name]
        return BaseGraphTree.get_apply(self, name)

    @staticmethod
    def set_apply(self, name, sub):
        if __package__ is None or __package__ == '':
            from concept import Concept
        else:
            from .concept import Concept
        # TODO: what if a concept has same name with a subgraph?
        if isinstance(sub, Graph):
            BaseGraphTree.set_apply(self, name, sub)
        elif isinstance(sub, Concept):
            self.concepts[name] = sub
        else:
            # FIXME: known issue is that relation goes here
            pass

    @property
    def concepts(self):
        return self._concepts

    def what(self):
        wht = BaseGraphTree.what(self)
        wht['concepts'] = dict(self.concepts)
        return wht
