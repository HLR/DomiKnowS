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
        
        # Use range instead of exact count
        atLeastAL(firestationCity, 2)  # At least 2
        atMostAL(firestationCity, 5)   # At most 5 (allows flexibility)

        # Coverage constraint: Each city must be a fire station OR have a neighbor that is
        orL(firestationCity('x'), existsL(firestationCity(path=('x', city1.reversed, eqL(cityLink, "neighbor", {True}), city2))))
        
        # Neighbor constraint: Each city has at most 3 neighbors that are NOT fire stations
        ifL(city('x'), atMostL(notL(firestationCity(path=('x', city1.reversed, eqL(cityLink, "neighbor", {True}), city2))), 3))

        # Global constraint: Fire stations cannot exceed total cities
        lessEqL(firestationCity, city)
        
        # Nested constraint: If world exists, then fire stations ≤ cities
        ifL(world('w'), lessEqL(firestationCity, city))