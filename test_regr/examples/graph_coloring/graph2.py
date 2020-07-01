from itertools import permutations

from regr.graph import Graph, Concept, Relation
from regr.graph.relation import disjoint
from regr.graph.logicalConstrain import orL, andL, existsL, notL


Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph2:
        world = Concept(name='world')
        city = Concept(name='city')
        (world_contains_city,) = world.contains(city)
        
        cityLink = Concept(name='cityLink')
        (city1, city2) = cityLink.has_a(arg1=city, arg2=city)

        firestationCity = city(name='firestationCity')
        
        # Constraints
        orL(firestationCity, ('x',), existsL((y,), andL(eql(cityLink, 'neighbor', 1), ('x', 'y'), firestationCity, ('y',))))

       
        # Each city has no more then 3 neighbors
        #atMostL(3, (x, ), andL(firestationCity, ('x',), eql(cityLink, 'neighbor', 1), ('x', 'y'))
