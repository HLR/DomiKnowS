from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import (
    lessEqL, orL, ifL, notL, existsL, eqL, atMostL, atMostAL, 
    atLeastAL, exactAL, greaterL, greaterEqL, lessL, equalCountsL, notEqualCountsL
)

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    world = Concept(name='world')
    city = Concept(name='city')
    (world_contains_city,) = world.contains(city)
    
    cityLink = Concept(name='cityLink')
    (city1, city2) = cityLink.has_a(arg1=city, arg2=city)

    # Fire station categories
    firestationCity = city(name='firestationCity')
    mainFirestation = firestationCity(name='mainFirestation')
    ancillaryFirestation = firestationCity(name='ancillaryFirestation')
    
    # Other city services
    emergencyService = city(name='emergencyService')
    groceryShop = city(name='groceryShop')
    
    # Fire station composition constraints
    exactAL(mainFirestation, 1)                    # Exactly 1 main firestation
    atLeastAL(ancillaryFirestation, 2)             # At least 2 ancillary firestations
    atMostAL(ancillaryFirestation, 5)              # At most 5 ancillary firestations
    lessL(mainFirestation, ancillaryFirestation)   # Main < ancillary (1 < 2+)
    equalCountsL(firestationCity, mainFirestation, ancillaryFirestation)  # Total = main + ancillary
    
    # This gives us: firestationCity = 1 + [2,5] = [3,6] total firestations
    
    # Service hierarchy constraints
    atLeastAL(emergencyService, 6)                 # At least 6 emergency services
    atMostAL(emergencyService, 7)                  # At most 7 emergency services  
    greaterEqL(emergencyService, firestationCity)  # Emergency ≥ firestation (6-7 ≥ 3-6 ✓)
    
    atLeastAL(groceryShop, 8)                      # At least 8 grocery shops
    atMostAL(groceryShop, 9)                       # At most 9 grocery shops
    greaterL(groceryShop, emergencyService)        # Grocery > emergency (8-9 > 6-7 ✓)
    
    # Other constraints
    notEqualCountsL(emergencyService, groceryShop) # Emergency ≠ grocery (different counts ✓)
    lessEqL(firestationCity, city)                 # Firestations ⊆ cities
    lessEqL(groceryShop, city)                     # Grocery shops ≤ total cities
    
    # Each city with grocery shop should have at least one neighbor with emergency service
    ifL(groceryShop('x'), 
        existsL(emergencyService(path=('x', city1.reversed, eqL(cityLink, "neighbor", {True}), city2))), 
        p=70)