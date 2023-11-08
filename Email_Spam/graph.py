import sys
sys.path.append("../..")

from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, V

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    email = Concept(name='email')

    Spam = email(name='spam')

    Regular = email(name='regular')

    # The constraint of
    orL(andL(notL(Spam), Regular, andL(notL(Regular), Spam)))
