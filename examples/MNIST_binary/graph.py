import sys

sys.path.append("../..")
sys.path.append(".")

from itertools import combinations
from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import nandL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('MNISTGraph') as graph:
    image_group = Concept(name='image_group')
    image = Concept(name='image')
    image_group_contains, = image_group.contains(image)

    Zero = image(name='image')
    One = image(name='image')
    Two = image(name='image')
    Three = image(name='image')
    Four = image(name='image')
    Five = image(name='image')
    Six = image(name='image')
    Seven = image(name='image')
    Eight = image(name='image')
    Nine = image(name='image')


    Numbers=[Zero,One,Two,Three,Four,Five,Six,Seven,Eight,Nine]
    for l1, l2 in combinations(Numbers, 2):
        nandL(l1, l2)

