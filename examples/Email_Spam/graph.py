import sys
sys.path.append("../..")

from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    email = Concept(name='email')

    Spam = email(name='spam')

    Regular = email(name='regular')

    # The constraint of
    orL(andL(notL(Spam, ('x', )), Regular, ('x', )), andL(notL(Regular, ('x', )), Spam, ('x', )))

