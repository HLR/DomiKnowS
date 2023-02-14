import sys

from domiknows.graph.relation import disjoint

sys.path.append("../..")
sys.path.append(".")

from itertools import combinations
from domiknows.graph import Graph, Concept, Relation, EnumConcept
from domiknows.graph.logicalConstrain import nandL, orL, notL, ifL, exactL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('MNISTGraph') as graph:
    image_group = Concept(name='image_group')
    image = Concept(name='image')
    image_group_contains, = image_group.contains(image)
    label=image(name="category", ConceptClass=EnumConcept,
                     values=[f"n{i}" for i in range(0, 10)])

    Zero = image(name='zerohandwriting')
    One = image(name='onehandwriting')
    Two = image(name='twohandwriting')
    Three = image(name='threehandwriting')
    Four = image(name='fourhandwriting')
    Five = image(name='fivehandwriting')
    Six = image(name='sixhandwriting')
    Seven = image(name='sevenhandwriting')
    Eight = image(name='eighthandwriting')
    Nine = image(name='ninehandwriting')


    Numbers=[Zero,One,Two,Three,Four,Five,Six,Seven,Eight,Nine]
    #for l1, l2 in combinations(Numbers, 2):
    #    nandL(l1, l2)
    #disjoint(*Numbers)
    ifL(image("x"),exactL(Zero("x"),One("x"),Two("x"),Three("x"),Four("x"),Five("x"),Six("x"),Seven("x"),Eight("x"),Nine("x")), active = False)
    
    exactL(Zero, One, Two, Three, Four, Five, Six, Seven, Eight, Nine, active = True)