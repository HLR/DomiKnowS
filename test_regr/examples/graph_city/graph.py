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
    
    # Original Constraints
    orL(firestationCity('x'), existsL(firestationCity(path=('x', city1.reversed, eqL(cityLink, "neighbor", {True}),  city2))))
    atLeastAL(firestationCity, p=90)
    atMostAL(firestationCity, 3, p=80)
    ifL(city('x'), atMostL(notL(firestationCity(path=('x', city1.reversed, eqL(cityLink, "neighbor", {True}),  city2))), 3), p=90)
    exactAL(firestationCity, 2, p=55)
    lessEqL(firestationCity, city, p=85)
    ifL(world('w'), lessEqL(firestationCity, city), p=80)
    
    # New Comparison Constraints
    
    # 1. Main firestations should be less than ancillary firestations
    lessL(mainFirestation, ancillaryFirestation, p=75)
    
    # 2. Emergency services should be greater than or equal to fire stations
    greaterEqL(emergencyService, firestationCity, p=85)
    
    # 3. Grocery shops should be greater than emergency services
    greaterL(groceryShop, emergencyService, p=80)
    
    # 4. Total of main and ancillary should equal firestationCity count
    equalCountsL(firestationCity, mainFirestation, ancillaryFirestation, p=95)
    
    # 5. Emergency services should not equal grocery shops (they should be different counts)
    notEqualCountsL(emergencyService, groceryShop, p=70)
    
    # 6. At least 1 main firestation in the world
    atLeastAL(mainFirestation, 1, p=90)
    
    # 7. At most 1 main firestation
    atMostAL(mainFirestation, 1, p=85)
    
    # 8. At least 2 emergency services
    atLeastAL(emergencyService, 2, p=80)
    
    # 9. At least 3 grocery shops
    atLeastAL(groceryShop, 3, p=75)
    
    # 10. Grocery shops should be less than or equal to total cities
    lessEqL(groceryShop, city, p=90)
    
    # 11. Each city with grocery shop should have at least one neighbor with emergency service
    ifL(groceryShop('x'), 
        existsL(emergencyService(path=('x', city1.reversed, eqL(cityLink, "neighbor", {True}), city2))), 
        p=70)