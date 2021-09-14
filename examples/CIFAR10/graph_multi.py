from regr.graph import Graph, Concept, Relation
from regr.graph.concept import EnumConcept
from regr.graph.logicalConstrain import nandL
from itertools import combinations

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('CIFAR10') as graph:
    Image = Concept(name='image')
    Category = Image(name="category", ConceptClass=EnumConcept, values=["animal", "vehicle"])
    Label = Image(name="tag", ConceptClass=EnumConcept, values=["airplane", "dog", "truck", "automobile", "bird", "cat", "deer", "frog", "horse", "ship"])

    nandL(Category.animal, Label.airplane)
    nandL(Category.animal, Label.truck)
    nandL(Category.animal, Label.automobile)
    nandL(Category.animal, Label.ship)

    nandL(Category.vehicle, Label.dog)
    nandL(Category.vehicle, Label.bird)
    nandL(Category.vehicle, Label.cat)
    nandL(Category.vehicle, Label.deer)
    nandL(Category.vehicle, Label.frog)
    nandL(Category.vehicle, Label.horse)

    for l1, l2 in combinations(Category.attributes, 2):
        nandL(l1, l2)

    for l1, l2 in combinations(Label.attributes, 2):
        nandL(l1, l2)