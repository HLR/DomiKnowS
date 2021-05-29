from regr.graph import Graph, Concept, Relation

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
