from itertools import permutations

from regr.graph import Graph, Concept, Relation
from regr.graph.relation import disjoint
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    sentence = Concept(name='sentence')
    word = Concept(name='word')
    word1 = Concept(name='word1')
    (word_equal_word1, ) = word.equal(word1)
    (sentence_con_word, ) = sentence.contains(word)

