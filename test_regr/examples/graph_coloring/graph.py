from itertools import permutations

from regr.graph import Graph, Concept, Relation
from regr.graph.relation import disjoint
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL


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
        
        # Constraints - For each city x either it is a firestationCity or exists a city y which is in neighbor relation to city x and y is a firestationCity
        orL(firestationCity, ('x',), existsL(('y',), andL(neighbor, ('x', 'y'), firestationCity, ('y',))), ('x',))
         
        # FirestationCity has at least 2 neighbors
        #atLeastL(2, ('x',), andL(firestationCity, ('x',), neighbor, ('x', 'y')))
               
        # FirestationCity has at most 3 neighbors
        #atMostL(3, ('x',), andL(firestationCity, ('x',), neighbor, ('x', 'y')))
        
        