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

    animal = image(name='animal')
    cat = animal(name='cat')
    dog = animal(name='dog')
    monkey = animal(name='monkey')
    squirrel = animal(name='squirrel')

    flower = image(name='flower')
    daisy = flower(name='daisy')
    dandelion = flower(name='dandelion')
    rose = flower(name='rose')
    sunflower = flower(name='sunflower')
    tulip = flower(name='tulip')

    ifL(cat('x'), animal('x'))
    ifL(dog('x'), animal('x'))
    ifL(monkey('x'), animal('x'))
    ifL(squirrel('x'), animal('x'))

    ifL(daisy('x'), flower('x'))
    ifL(dandelion('x'), flower('x'))
    ifL(rose('x'), flower('x'))
    ifL(sunflower('x'), flower('x'))
    ifL(tulip('x'), flower('x'))
    ifL(flower, orL(daisy, dandelion, rose, sunflower, tulip))
    ifL(animal, orL(cat, dog, monkey, squirrel))
    disjoint(cat, dog, monkey, squirrel, daisy, dandelion, rose, sunflower, tulip)

    for l1, l2 in combinations([daisy, dandelion, rose, sunflower, tulip, cat, dog, monkey, squirrel], 2):
        # TODO isn't this the same as disjoint?
        nandL(l1, l2)

    for l1, l2 in combinations([animal, flower], 2):
        nandL(l1, l2)
