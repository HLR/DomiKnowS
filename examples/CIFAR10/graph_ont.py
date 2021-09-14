from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, ifL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('CIFAR10') as graph_ont:
    graph_ont.ontology = ('http://ontology.ihmc.us/ML/CIFAR10.owl', './')
    
    image = Concept(name='image')

    animal = image(name='animal')
    vehicle = image(name='vehicle')
    
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
    
    lc_sets = {'LC_CW_U': ('LC_CW_U', False),
               'LC_CW_A': ('LC_CW_A', False),
               'LC_CW_V': ('LC_CW_V', False)
               }
    
    ifL(
        image('i'),
        orL(animal(path='i'), vehicle(path='i')),
        active = lc_sets['LC_CW_U'][1], name = lc_sets['LC_CW_U'][0]
        )
    
    ifL(
        animal('a'),
        orL(dog(path='a'), bird(path='a'), cat(path='a'), deer(path='a'), frog(path='a'), horse(path='a')),
        active = lc_sets['LC_CW_A'][1], name = lc_sets['LC_CW_A'][0]
        )
    
    ifL(
        vehicle('v'),
        orL(airplane(path='v'), truck(path='v'), automobile(path='v'), ship(path='v')),
        active = lc_sets['LC_CW_V'][1], name = lc_sets['LC_CW_V'][0]
        )