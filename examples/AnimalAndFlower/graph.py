from regr.graph import Graph, Concept, Relation, EnumConcept
from regr.graph.logicalConstrain import nandL, orL, andL, existsL, notL, atLeastL, atMostL, ifL
from regr.graph.relation import disjoint, IsA
from itertools import combinations

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('AnimalAndFlower') as graph:
    image = Concept(name='image')

    animal = image(name='animal')
    cat = image(name='cat')
    dog = image(name='dog')
    monkey = image(name='monkey')
    squirrel = image(name='squirrel')

    flower = image(name='flower')
    daisy = image(name='daisy')
    dandelion = image(name='dandelion')
    rose = image(name='rose')
    sunflower = image(name='sunflower')
    tulip = image(name='tulip')

    ifL(cat('x'), animal('x'))
    ifL(dog('x'), animal('x'))
    ifL(monkey('x'), animal('x'))
    ifL(squirrel('x'), animal('x'))

    ifL(daisy('x'), flower('x'))
    ifL(dandelion('x'), flower('x'))
    ifL(rose('x'), flower('x'))
    ifL(sunflower('x'), flower('x'))
    ifL(tulip('x'), flower('x'))

    disjoint(cat, dog, monkey, squirrel, daisy, dandelion, rose, sunflower, tulip)

    for l1, l2 in combinations([daisy, dandelion, rose, sunflower, tulip, cat, dog, monkey, squirrel], 2):
        nandL(l1, l2)

    for l1, l2 in combinations([animal, flower], 2):
        nandL(l1, l2)
