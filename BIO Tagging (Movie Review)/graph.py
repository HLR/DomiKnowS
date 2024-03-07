from domiknows.graph.dataNodeDummy import createDummyDataNode
from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import nandL, orL, andL, existsL, notL, atLeastL, atMostL, ifL
from domiknows.graph.relation import disjoint
from domiknows.graph.concept import EnumConcept

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    NERSentence = Concept(name="NERSentence")
    tag = Concept(name='tag')
    NER_contains, = NERSentence.contains(tag)
    Generated_label=tag(name="label", ConceptClass=EnumConcept, values=['B', 'I', 'O'])
        
    #next_word = Concept(name="next_word")
    #(first_word, second_word) = next_word.has_a(arg1=Generated_label, arg2=Generated_label)
    
    #ifL(getattr(Generated_label, '<O>')('x'), 
    #        notL(
    #                andL(
     #                   next_word('y', path=("x", first_word.reversed)), 
    #                    getattr(Generated_label, '<I>')('t', path=("y", second_word))
    #            )
    #    ))
