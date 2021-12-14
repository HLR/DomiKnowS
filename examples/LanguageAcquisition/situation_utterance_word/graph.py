'''
    Current setting: Generate an utterance followed by the words from the situation
    
    EX: re1, st1 -> "the red star"
    
    Concepts: situation, utterance, word 
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

# Build the vocabulary
vocabulary = [x.strip() for x in open("../data/vocabulary.txt")]

# with Graph('IL') as graph:
#
#     # Establish relation between situation and utterance
#     situation = Concept(name='situation')
#
#     utterance = Concept(name='utterance')
#     word = utterance(name='word')
#
#     (utterance_contains_word,) = utterance.contains(word)
#
#     word_label = word(name='word_label', ConceptClass=EnumConcept, values=vocabulary)

with Graph('IL') as graph:

    task = Concept(name='task')
    situation = Concept(name='situation')
    word = Concept(name='word')
    utterance = Concept(name='utterance')
    (task_sit, task_utt) = task.has_a(arg1=situation, arg2=utterance)
    word_label = word(name='word_label', ConceptClass=EnumConcept, values=vocabulary)

    # Establish relationship one-to-many
    (utterance_contains_words,) = utterance.contains(word)
