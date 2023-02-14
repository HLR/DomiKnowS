from domiknows.graph import Graph, Concept, Relation, EnumConcept
from domiknows.graph.logicalConstrain import nandL, orL, andL, existsL, notL, atLeastL, atMostL, ifL
from domiknows.graph.relation import disjoint, IsA
from itertools import combinations

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('AnimalAndFlower') as graph:
    image_group = Concept(name='image_group')
    image = Concept(name='image')
    image_group_contains, = image_group.contains(image)

    category = image(name="category", ConceptClass=EnumConcept, values=["animal", "flower"])
    tag = image(
        name="tag",
        ConceptClass=EnumConcept,
        values=["cat", "dog", "monkey", "squirrel", "daisy", "dandelion", "rose", "sunflower", "tulip"]
    )

    nandL(category.animal, tag.daisy, active=True)
    nandL(category.animal, tag.dandelion, active=True)
    nandL(category.animal, tag.rose, active=True)
    nandL(category.animal, tag.sunflower, active=True)
    nandL(category.animal, tag.tulip, active=True)

    nandL(category.flower, tag.cat, active=True)
    nandL(category.flower, tag.dog, active=True)
    nandL(category.flower, tag.monkeyv, active=True)
    nandL(category.flower, tag.squirrel, active=True)

    for l1, l2 in combinations(category.attributes, 2):
        nandL(l1, l2, active=True)

    for l1, l2 in combinations(tag.attributes, 2):
        nandL(l1, l2, active=True)

    ifL(category.flower, orL(tag.daisy, tag.dandelion, tag.rose, tag.sunflower, tag.tulip), active=True)
    ifL(category.animal, orL(tag.cat, tag.dog, tag.monkey, tag.squirrel), active=True)
