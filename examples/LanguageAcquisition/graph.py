'''
    This script sets the graph for the interaction model.
    
    Two concepts: Situation and Utterance

'''

import sys

if "../.." not in sys.path:
    sys.path.append('../..')
# print("sys.path - %s"%(sys.path))

from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import ifL, andL, nandL, V, orL, exactL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('IL') as graph:
    
    situation = Concept(name='situation')
    predicate = Concept(name='predicate')
    (situation_contains_predicate,) = situation.contains(predicate)
    
    
    utterance = Concept(name='utterance')
    word = Concept(name='word')
    (utterance_contains_word,) = utterance.contains(word)
    
    
graph.visualize('image')

