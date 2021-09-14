from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, andL, ifL, notL, existsL, eqL, existsI, atMostL, atLeastI, exactI, atMostI


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
        
        # Constraints - For each city x either it is a firestationCity or exists a city y which is in cityLink relation with neighbor attribute equal True to city x and y is a firestationCity
        orL(firestationCity('x'), existsL(firestationCity(path=('x', eqL(cityLink, 'neighbor', {True}),  city2))))
        
        # No less then 1 firestationCity in the world
        atLeastI(firestationCity, p=90)
        
        # At most 3 firestationCity in the world
        atMostI(firestationCity, 3, p=80)
        
        # Each city has no more then 4 neighbors which are not firestationCity
        ifL(city('x'), atMostL(notL(firestationCity(path=('x', eqL(cityLink, 'neighbor', {True}), city2)), 1)), p=100)

        # Exactly 2 firestationCity in the world 
        exactI(firestationCity, 2, p=55)