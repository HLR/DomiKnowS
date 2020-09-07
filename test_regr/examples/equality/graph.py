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
    (word_equal_word1, ) = word.equal(arg1=word1)
    (sentence_con_word, ) = sentence.contains(word)
    
    #(sentence_con_word1, ) = sentence.contains(word1)
    
    #   sentence_con_word_list = graph['sentence'].contains() How you can retrieve this relation
    #   sentence_con_word = sentence_con_word_list[0]
    #   sentence_con_word1 = sentence_con_word_list[1]
    #   sentence_con_word = sentence.relate_to(word)[0]
    

