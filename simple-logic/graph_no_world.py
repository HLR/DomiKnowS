from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, notL, andL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph(name='global') as graph:
    x = Concept(name='x')
    y0 = x(name='y0')
    y1 = x(name='y1')
    orL(andL(y0, notL(y1)), andL(notL(y0), y1))
