from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import nandL, orL, andL, existsL, notL, atLeastL, atMostL, ifL
from domiknows.graph.relation import disjoint

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('CIFAR10') as graph:
    image = Concept(name='image')
    
    # First level
    animal = image(name='animal')
    vehicle = image(name='vehicle')

    # Second level
    airplane = image(name='airplane')
    airplane.is_a(vehicle)

    dog = image(name='dog')
    dog.is_a(animal)

    truck = image(name='truck')
    truck.is_a(vehicle)

    automobile = image(name='automobile')
    automobile.is_a(vehicle)

    bird = image(name='bird')
    bird.is_a(animal)

    cat = image(name='cat')
    cat.is_a(animal)

    deer = image(name='deer')
    deer.is_a(animal)

    frog = image(name='frog')
    frog.is_a(animal)

    horse = image(name='horse')
    horse.is_a(animal)

    ship = image(name='ship')
    ship.is_a(vehicle)


    disjoint(dog, bird, cat, deer, frog, horse)
    disjoint(airplane, truck, automobile, ship)


