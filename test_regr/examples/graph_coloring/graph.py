from itertools import permutations

from regr.graph import Graph, Concept, Relation
from regr.graph.relation import disjoint
from regr.graph.logicalConstrain import orL, andL, existsL, notL


Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
        world = Concept(name='world')
        city = Concept(name='city')
        (world_contains_city,) = world.contains(city)
         
        neighbor = Concept(name='neighbor')
        (city1, city2) = neighbor.has_a(arg1=city, arg2=city)
        
        firestationCity = city(name='firestationCity')
        
        # Constraints
        existsL(orL(firestationCity, ('x',), andL(neighbor, ('x', 'y'), firestationCity, ('y',))))