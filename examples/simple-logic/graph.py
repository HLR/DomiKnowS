from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, notL, andL, V

Graph.clear()
Concept.clear()
Relation.clear()

with Graph(name='global') as graph:
    world = Concept(name='world')
    x = Concept(name='x')
    (world_contains_x,) = world.contains(x)
    y0 = x(name='y0')
    y1 = x(name='y1')
    
    orL(andL(y0,  V(name='x'), notL(y1,  V(name='y'))), andL(notL(y0,  V(name='z')), y1,  V(name='t')))
