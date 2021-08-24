from regr.graph import Graph, Concept, Relation, EnumConcept
from regr.graph.logicalConstrain import nandL, orL, andL, existsL, notL, atLeastL, atMostL, ifL
from regr.graph.relation import disjoint, IsA
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

    nandL(category.animal, tag.daisy)
    nandL(category.animal, tag.dandelion)
    nandL(category.animal, tag.rose)
    nandL(category.animal, tag.sunflower)
    nandL(category.animal, tag.tulip)

    nandL(category.flower, tag.cat)
    nandL(category.flower, tag.dog)
    nandL(category.flower, tag.monkey)
    nandL(category.flower, tag.squirrel)

    for l1, l2 in combinations(category.attributes, 2):
        nandL(l1, l2)

    for l1, l2 in combinations(tag.attributes, 2):
        nandL(l1, l2)

    ifL(category.flower, orL(tag.daisy, tag.dandelion, tag.rose, tag.sunflower, tag.tulip))
    ifL(category.animal, orL(tag.cat, tag.dog, tag.monkey, tag.squirrel))
