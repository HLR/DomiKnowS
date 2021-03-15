from regr.graph import Graph, Concept, Relation
from regr.graph.concept import EnumConcept
from regr.graph.logicalConstrain import nandL
from itertools import combinations

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('CIFAR10') as graph:
    Image = Concept(name='image')
    vehicle = Image(name="vehicle")
    animal = Image(name="animal")

    tag = Image(name="tag", ConceptClass=EnumConcept, values=["airplane", "dog", "truck", "automobile", "bird", "cat", "deer", "frog", "horse", "ship"])

    nandL(animal, tag.airplane)
    nandL(animal, tag.truck)
    nandL(animal, tag.automobile)
    nandL(animal, tag.ship)

    nandL(vehicle, tag.dog)
    nandL(vehicle, tag.bird)
    nandL(vehicle, tag.cat)
    nandL(vehicle, tag.deer)
    nandL(vehicle, tag.frog)
    nandL(vehicle, tag.horse)
     
    for l1, l2 in combinations(tag.attributes, 2):
        nandL(l1, l2)
        
    nandL(animal, vehicle)