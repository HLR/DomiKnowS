from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import V, orL, andL, existsL, notL, eqL, atLeastL, atMostL, exactL

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
        
        # No less then 1 firestationCity
        atLeastL(1, ('x', ), firestationCity, ('x', ), p=90)
        
        # At most 1 firestationCity
        atMostL(1, ('x', ), firestationCity, ('x', ), p=80)
        
        # Exactly 2 firestationCity
        exactL(2, ('x', ), firestationCity, ('x', ), p=55)
        
        # Constraints - For each city x either it is a firestationCity or exists a city y which is in cityLink relation with neighbor attribute equal 1 to city x and y is a firestationCity
        orL(firestationCity, ('x',), existsL(('y',), andL(eqL(cityLink, 'neighbor', {1}), ('x', 'y'), firestationCity, ('y',))))

        # Each city has no more then 4 neighbors
        atMostL(4, ('y', ), andL(firestationCity, ('x',), eqL(cityLink, 'neighbor', 1), ('x', 'y')))
