from itertools import permutations

from regr.graph import Graph, Concept, Relation
from regr.graph.relation import disjoint
from regr.graph.logicalConstrain import orL, andL, existsL, notL


Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    with Graph('structure') as structure_graph:
        world = Concept(name='world')
        city = Concept(name='city')
        (world_contains_city,) = world.contains(city)
         
        cityLink = Concept(name='cityLink')
        (city1, city2) = cityLink.has_a(arg1=city, arg2=city)
        
    with Graph('application') as app_graph:
        firestationCity = city(name='firestationCity')
        #city[firestationCity] = None # declare property for name only

        neighbor = cityLink(name='neighbor')
        
        # Constraints
        existsL(orL(firestationCity, ('x',), andL(neighbor, ('x', 'y'), firestationCity, ('y',))))