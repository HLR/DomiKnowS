from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL
from regr.graph.relation import disjoint

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('CIFAR') as graph:
    image = Concept(name='image')

    airplane = image(name='airplane')
    dog = image(name='dog')
    truck = image(name='truck')
    disjoint(truck, dog, airplane)

    # # The constraint
    # andL(andL(notL(airplane, ('x', )),notL(dog, ('x', )), truck, ('x', )),
    #     andL(notL(airplane, ('x', )),dog, ('x', ), notL(truck, ('x', ))),
    #     andL(airplane, ('x',), notL(dog, ('x',)), notL(truck, ('x',)))
    #     )

