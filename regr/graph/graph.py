from collections import OrderedDict, namedtuple
from itertools import chain

if __package__ is None or __package__ == '':
    from base import BaseGraphTree
    from property import Property
else:
    from .base import BaseGraphTree
    from .property import Property


@BaseGraphTree.localize_namespace
class Graph(BaseGraphTree):
    def __init__(self, name=None, ontology=None, iri=None, local=None):
        BaseGraphTree.__init__(self, name)
        if ontology is None:
            self.ontology = (iri, local)
        elif isinstance(ontology, Graph.Ontology):
            self.ontology = ontology
        elif isinstance(ontology, str):
            self.ontology = (ontology, local)
        self._concepts = OrderedDict()
        self._logicalConstrains = OrderedDict()
        self._relations = OrderedDict()

    def __iter__(self):
        yield from BaseGraphTree.__iter__(self)
        yield from self._concepts
        yield from self._relations

    @property
    def ontology(self):
        return self._ontology
    
    @ontology.setter
    def ontology(self, ontology):
        if isinstance(ontology, Graph.Ontology):
            self._ontology = ontology
        elif isinstance(ontology, str):
            self._ontology = Graph.Ontology(ontology)
        elif isinstance(ontology, tuple) and len(ontology) == 2:
            self._ontology = Graph.Ontology(*ontology)

    def get_sensors(self, *tests):
        def func(node):
            # use a closure to collect sensors
            if isinstance(node, Property):
                return node.find(*tests)
        return list(chain(*self.traversal_apply(func)))

    def get_apply(self, name):
        if name in self.concepts:
            return self.concepts[name]
        if name in self.relations:
            return self.relations[name]
        return BaseGraphTree.get_apply(self, name)

    def set_apply(self, name, sub):
        if __package__ is None or __package__ == '':
            from concept import Concept
            from relation import Relation
        else:
            from .concept import Concept
            from .relation import Relation
        # TODO: what if a concept has same name with a subgraph?
        if isinstance(sub, Graph):
            BaseGraphTree.set_apply(self, name, sub)
        elif isinstance(sub, Concept):
            self.concepts[name] = sub
        elif isinstance(sub, Relation):
            self.relations[name] = sub
        else:
            # TODO: what are other cases
            pass

    @property
    def concepts(self):
        return self._concepts
    
    @property
    def logicalConstrains(self):
        return self._logicalConstrains

    @property
    def relations(self):
        return self._relations

    def what(self):
        wht = BaseGraphTree.what(self)
        wht['concepts'] = dict(self.concepts)
        return wht

    # NB: for namedtuple, defaults are right-most first
    # `local` default to None,
    # python 3.7+
    # Ontology = namedtuple('Ontology', ('iri', 'local'), defaults=(None,))
    # python 2.6+
    Ontology = namedtuple('Ontology', ('iri', 'local'))
    Ontology.__new__.__defaults__ = (None,)