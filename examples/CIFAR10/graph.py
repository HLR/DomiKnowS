from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import nandL, orL, andL, existsL, notL, atLeastL, atMostL, ifL
from regr.graph.relation import disjoint

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('CIFAR10') as graph:
    image = Concept(name='image')

    airplane = image(name='airplane')
    dog = image(name='dog')
    truck = image(name='truck')
    automobile = image(name='automobile')
    bird = image(name='bird')
    cat = image(name='cat')
    deer = image(name='deer')
    frog = image(name='frog')
    horse = image(name='horse')
    ship = image(name='ship')

#     l = [truck, dog, airplane, automobile, bird, cat, deer, frog, horse, ship]
#     for i in range(len(l)):
#         for j in range(i,len(l)):
#             if l[i] != l[j]:
#                 nandL(l[i],l[j])
    
    
#     orL(truck, dog, airplane, automobile, bird, cat, deer, frog, horse, ship)
#     disjoint()
    
#     ##### Old version constraints
#     nandL(truck, dog, airplane, automobile, bird, cat, deer, frog, horse, ship)

    disjoint(truck, dog, airplane, automobile, bird, cat, deer, frog, horse, ship)

