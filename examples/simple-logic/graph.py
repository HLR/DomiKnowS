from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, notL, andL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    with Graph('input'):
        x = Concept('x')
    with Graph('output'):
        y0 = x('y0')
        y1 = x('y1')
        orL(andL(y0, notL(y1)), andL(notL(y0), y1))
