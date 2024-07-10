from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import orL, existsL, ifL, notL, andL, atMostAL, atLeastAL

def get_graph(constraint,atmost,atleast):

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


        atMostAL(firestationCity,atmost)
        atLeastAL(firestationCity,atleast)

        if constraint=="existL":
            orL(firestationCity('x'), existsL(
                ifL(neighbor("z",path=('x', city1.reversed))
                    ,firestationCity("p",path=('z', city2))
                )
            ))
        elif constraint=="orLnotLexistL":
            orL(firestationCity('x'), notL(existsL(
                ifL(neighbor("z",path=('x', city1.reversed))
                    , firestationCity("p",path=('z', city2))
                )
            )))
        elif constraint=="ifLnotLexistL":
            ifL(firestationCity('x'), notL(existsL(
                ifL(neighbor("z",path=('x', city1.reversed))
                    , firestationCity("p",path=('z', city2))
                )
            )))
        elif constraint=="orL":
            ifL(neighbor('z'), orL(
                firestationCity("x",path=('z', city1)),
                firestationCity("p",path=('z', city2))
            ))
        else:
            pass
            #print("no constraint.")
    return graph, world, city, world_contains_city, neighbor, city1, city2, firestationCity
