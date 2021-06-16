from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, notL, andL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph(name='global') as graph:
    x = Concept(name='x')
    y = x(name='y0')
