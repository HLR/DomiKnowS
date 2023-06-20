from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import orL, existsL


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
        #orL(firestationCity, V(name='x'), existsL(firestationCity, V(name='y', v=('x', neighbor.name, city2.name)), 'y'), V(name='z'))
        orL(firestationCity('x'), existsL(firestationCity('y', path=('x', city1.reversed, city2))))