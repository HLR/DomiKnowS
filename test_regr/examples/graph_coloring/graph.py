from itertools import permutations

from regr.graph import Graph, Concept, Relation
from regr.graph.relation import disjoint
from regr.graph.logicalConstrain import orL, andL, existsL


Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    with Graph('application') as app_graph:
        world = Concept(name='world')
        city = Concept(name='city')
        (world_contains_city,) = world.contains(city)
        firestationCity = Concept(name='firestationCity')
        (city_contains_firestationCity,) = city.contains(firestationCity)

        neighbor = Concept(name='neighbor')
        (neighbor_city1, neighbor_city2) = neighbor.has_a(arg1=city, arg2=city)

        # Constraints
        existsL(orL(firestationCity, (x,), andL(neighbor, (x, y), firestationCity, (y,))))
        
        # Remember to add that if city a is neighbor of city b then the reverse also holds and should contain the same properties on the edge