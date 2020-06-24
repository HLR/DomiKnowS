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

        neighbor = cityLink(name='neighbor')
        
        # Nee to change arg1 to arg3 and arg2 to arg4 - system changed them  if there are the same but do change is not consistent
        (neighbor1, neighbor2) = neighbor.has_a(arg3=city, arg4=city) # Required for CandidateReaderSensor to work
       
        firestationCity = city(name='firestationCity')
        
        # Constraints
        existsL(orL(firestationCity, ('x',), andL(neighbor, ('x', 'y'), firestationCity, ('y',))), ('x',))