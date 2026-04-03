def get_graph(constraint, atmost, atleast, test_number):
    from domiknows.graph import Graph, Concept, Relation
    from domiknows.graph.logicalConstrain import orL, existsL, ifL, notL, andL, atMostAL, atLeastAL
    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph('global'+str(test_number)) as graph:
        world = Concept(name='world_'+str(test_number))
        city = Concept(name='city_'+str(test_number))
        (world_contains_city,) = world.contains(city)
        
        neighbor = Concept(name='neighbor_'+str(test_number))
        (city1, city2) = neighbor.has_a(arg1=city, arg2=city)
        
        firestationCity = city(name='firestationCity_'+str(test_number))

        # At most {atmost} cities can be fire stations
        atMostAL(firestationCity, atmost)
        # At least {atleast} cities must be fire stations
        atLeastAL(firestationCity, atleast)

        if constraint == "existL":
            # City is fire station OR exists a neighboring city that is a fire station
            orL(firestationCity('x'), existsL(
                ifL(neighbor("z", path=('x', city1.reversed)),
                    firestationCity("p", path=('z', city2))
                )
            ))
        elif constraint == "orLnotLexistL":
            # City is fire station OR none of its neighbors are fire stations
            orL(firestationCity('x'), notL(existsL(
                ifL(neighbor("z", path=('x', city1.reversed)),
                    firestationCity("p", path=('z', city2))
                )
            )), name="orLnotLexistL")
        elif constraint == "ifLnotLexistL":
            # If city is fire station THEN none of its neighbors are fire stations
            ifL(firestationCity('x'), notL(existsL(
                ifL(neighbor("z", path=('x', city1.reversed)),
                    firestationCity("p", path=('z', city2))
                )
            )))
        elif constraint == "orL":
            # For each neighbor pair, at least one city must be a fire station
            ifL(neighbor("z1", "z2"), orL(
                firestationCity("z1"),
                firestationCity("z2")
            ))
        else:
            pass

    return graph, world, city, world_contains_city, neighbor, city1, city2, firestationCity