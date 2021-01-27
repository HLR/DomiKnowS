from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, V

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    email = Concept(name='email')

    Spam = email(name='spam')

    Regular = email(name='regular')

    # The constraint of
    orL(andL(notL(Spam, V(name='x')), Regular, V(name='y')), andL(notL(Regular, V(name='z')), Spam, V(name='t')))

