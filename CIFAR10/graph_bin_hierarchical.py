from regr.graph import Graph, Concept, Relation
from regr.graph.concept import EnumConcept
from regr.graph.logicalConstrain import nandL
from itertools import combinations
from regr.graph.relation import disjoint

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('CIFAR10') as graph:
    Image = Concept(name='image')
    vehicle = Image(name="vehicle")
    animal = Image(name="animal")

    airplane = animal(name='airplane')
    dog = animal(name='dog')
    truck = vehicle(name='truck')
    automobile = vehicle(name='automobile')
    bird = animal(name='bird')
    cat = animal(name='cat')
    deer = animal(name='deer')
    frog = animal(name='frog')
    horse = animal(name='horse')
    ship = vehicle(name='ship')



    disjoint(truck, dog, airplane, automobile, bird, cat, deer, frog, horse, ship)
    nandL(animal, vehicle)

    nandL(animal, airplane)
    nandL(animal, truck)
    nandL(animal, automobile)
    nandL(animal, ship)

    nandL(vehicle, dog)
    nandL(vehicle, bird)
    nandL(vehicle, cat)
    nandL(vehicle, deer)
    nandL(vehicle, frog)
    nandL(vehicle, horse)
    l = [truck, dog, airplane, automobile, bird, cat, deer, frog, horse, ship]
    for i in range(len(l)):
        for j in range(i,len(l)):
            if l[i] != l[j]:
                nandL(l[i],l[j])
