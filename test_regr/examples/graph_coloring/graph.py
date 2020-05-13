from itertools import permutations

from regr.graph import Graph, Concept, Relation
from regr.graph.relation import disjoint
from regr.graph.logicalConstrain import ifL, andL, nandL


Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    with Graph('application') as app_graph:
        world = Concept(name='world')
        city = Concept(name='city')
        (world_contains_city,) = world.contains(city)
        neighbor = Concept(name='neighbor')
        (neighbor_city1, neighbor_city2) = neighbor.has_a(arg1=city, arg2=city)

        # Add the constraints here
        # Remember to add that if city a is neighbor of city b then the reverse also holds and should contain the same properties on the edge
