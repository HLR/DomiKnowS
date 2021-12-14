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
from regr.graph.concept import EnumConcept
from itertools import combinations

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('IL') as graph:
    
    situation = Concept(name='situation')
    utterance = Concept(name='utterance')
    
    # (situation_contains_predicate,) = situation.contains(predicate)
    
    
    # utterance = Concept(name='utterance')
    # word = Concept(name='word')
    # (utterance_contains_word,) = utterance.contains(word)
    