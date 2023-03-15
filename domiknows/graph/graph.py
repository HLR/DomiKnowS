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
    def __init__(self, name=None, ontology=None, iri=None, local=None, auto_constraint=None, reuse_model=False):
        BaseGraphTree.__init__(self, name)
        if ontology is None:
            self.ontology = (iri, local)
        elif isinstance(ontology, Graph.Ontology):
            self.ontology = ontology
        elif isinstance(ontology, str):
            self.ontology = (ontology, local)
        self.auto_constraint = auto_constraint
        self.reuse_model = reuse_model
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

    @property
    def auto_constraint(self):
        if self._auto_constraint is None and self.sup:
            return self.sup.auto_constraint
        return self._auto_constraint or False  # if None, return False instead

    @auto_constraint.setter
    def auto_constraint(self, value):
        self._auto_constraint = value

    @ontology.setter
    def ontology(self, ontology):
        if isinstance(ontology, Graph.Ontology):
            self._ontology = ontology
        elif isinstance(ontology, str):
            self._ontology = Graph.Ontology(ontology)
        elif isinstance(ontology, tuple) and len(ontology) == 2:
            self._ontology = Graph.Ontology(*ontology)

    def get_properties(self, *tests):
        def func(node):
            if isinstance(node, Property):
                for test in tests:
                    if not test(node):
                        return None
                else:
                    return node
            return None
        return list(self.traversal_apply(func))

    def get_sensors(self, *tests):
        return list(chain(*(prop.find(*tests) for prop in self.get_properties())))

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


    def visualize(self, filename, open_image=False):
        import graphviz
        concept_graph = graphviz.Digraph(name=f"{self.name}")
        concept_graph.attr(label=f"Graph: {self.name}") 

        for concept_name, concept in self.concepts.items():
            concept_graph.node(concept_name)

        for subgraph_name, subgraph in self.subgraphs.items():
            sub_graph_viz = subgraph.visualize(filename=None)
            sub_graph_viz.name = 'cluster_' + sub_graph_viz.name
            concept_graph.subgraph(sub_graph_viz)

        for relation_name, relation in self.relations.items():
            if not relation.name.endswith('reversed') and not ('not_a' in relation.name):
                # add case for HasA
                concept_graph.edge(relation.src.name, relation.dst.name, label=relation.__class__.__name__)

        if filename is not None:
            concept_graph.render(filename, format='png', view=open_image)
        else:
            return concept_graph

    @property
    def subgraphs(self):
        return OrderedDict(self)

    @property
    def concepts(self):
        return self._concepts

    @property
    def logicalConstrains(self):
        return self._logicalConstrains

    @property
    def logicalConstrainsRecursive(self):
        def func(node):
            if isinstance(node, Graph):
                yield from node.logicalConstrains.items()
        yield from self.traversal_apply(func)

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
