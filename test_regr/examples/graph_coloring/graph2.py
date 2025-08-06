from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import lessEqL, orL, ifL, notL, existsL, eqL, atMostL, atMostAL, atLeastAL, exactAL


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
        
        # Constraints - For each city x either it is a firestationCity or 
        # exists a city y which is in cityLink relation with neighbor attribute equal True to city x and y is a firestationCity
        orL(firestationCity('x'), existsL(firestationCity(path=('x', city1.reversed, eqL(cityLink, "neighbor", {True}),  city2))))
        
        # No less then 1 firestationCity in the world
        atLeastAL(firestationCity, p=90)
        
        # At most 3 firestationCity in the world
        atMostAL(firestationCity, 3, p=80)
        
        # Each city has no more then 4 neighbors which are not firestationCity
        ifL(city('x'), atMostL(notL(firestationCity(path=('x', city1.reversed, eqL(cityLink, "neighbor", {True}),  city2))), 3), p=90)

        # Exactly 2 firestationCity in the world 
        exactAL(firestationCity, 2, p=55)
        
        # (A) **Global**: the number of fire-station cities may never exceed
        #     the number of cities in total.
        lessEqL(firestationCity, city, p=85)        # count(F) ≤ count(C)

        # (B) **Nested** inside an implication:  *if* the (single) world node
        #     exists, then the global relation in (A) must hold (it already
        #     does, but this demonstrates nesting compare-counts under another
        #     higher-level LC).
        ifL(world('w'), lessEqL(firestationCity, city), p=80)