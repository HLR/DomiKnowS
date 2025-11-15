from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import (
    atLeastL, exactL, lessEqL, orL, ifL, notL, existsL, eqL, atMostL, atMostAL, 
    atLeastAL, exactAL, greaterL, greaterEqL, lessL, equalCountsL, notEqualCountsL, sumL
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


    # Example 1: Simple sumL - Total emergency + grocery services MUST equal 14
    # Using exactL to enforce equality between sum and constant
    exactL(
        sumL(emergencyService, groceryShop),
        14  # Must equal exactly 14
    )
    
    # Example 2: sumL in comparison - Total emergency services + firestations should be at least 8
    atLeastL(
        sumL(emergencyService, mainFirestation, ancillaryFirestation),  # Sum emergency + all firestations
        8  # Must be >= 8
    )
    
    # Example 3: sumL in another comparison - emergency services > sum of firestation types
    greaterL(
        emergencyService,  # Emergency count
        sumL(mainFirestation, ancillaryFirestation)  # Sum of firestation types
    )
    
    # Example 4: Complex nested sumL in ifL
    # If a city has more than 10 total services (emergency + grocery), 
    # then it must have at least one firestation
    ifL(
        atLeastL(
            sumL(emergencyService, groceryShop),  # Total emergency + grocery
            6  # More than 6
        ),
        existsL(firestationCity)  # Must have at least one firestation    
    )
    
    # Example 5: sumL with equalCountsL and usage of explicit constant
    # The total number of emergency services and grocery shops combined 
    # must equal 0 - the number of cities
    equalCountsL(
        sumL(emergencyService, groceryShop),  # Total emergency + grocery
        9  # Must equal number 9 which represent number of cities
    )