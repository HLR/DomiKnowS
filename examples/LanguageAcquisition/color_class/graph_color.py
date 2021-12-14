'''
    Current setting: Match predicate to color word
    
    EX: re1 -> red
    
    Concepts: predicate and word
'''


# import sys
# sys.path.append('../..')
# print("sys.path - %s"%(sys.path))

from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import ifL, andL, notL, nandL, V, orL, exactL
from regr.graph.relation import disjoint

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    
    predicate = Concept(name='predicate')
    
    Red = predicate(name='red')
    Blue = predicate(name='blue')
    Yellow = predicate(name='yellow')
    Purple = predicate(name='purple')
    Orange = predicate(name='orange')
    Green = predicate(name='green')
    
    disjoint(Red,Blue,Yellow,Purple,Orange,Green)
    
    # orL(andL(notL(Red), Blue, andL(notL(Blue), Red)))