from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import ifL, andL, nandL
from domiknows.graph import EnumConcept
from itertools import combinations

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph_multi:
    with Graph('linguistic') as ling_graph:
        ling_graph.ontology = ('http://ontology.ihmc.us/ML/PhraseGraph.owl', './')
       
        word = Concept(name='word')
        phrase = Concept(name='phrase')
        sentence = Concept(name='sentence')
        (rel_sentence_contains_word,) = sentence.contains(word)
        (rel_sentence_contains_phrase,) = sentence.contains(phrase)
        (rel_phrase_contains_word,) = phrase.contains(word)

        pair = Concept(name='pair')
        (rel_pair_phrase1, rel_pair_phrase2, ) = pair.has_a(arg1=phrase, arg2=phrase)

    with Graph('application') as app_graph:
        entity = phrase(name='entity')
        entity_label = entity(name="entity_label", ConceptClass=EnumConcept, values=["people", "organization", 'location', 'other', 'O'])
        pair_label = pair(name="pair_label", ConceptClass=EnumConcept, values=["work_for", "located_in", 'live_in', 'orgbase_on', 'kill'])
#        
        for l1, l2 in combinations(entity_label.attributes, 2):
            nandL(l1, l2)

        for l1, l2 in combinations(pair_label.attributes, 2):
            nandL(l1, l2)
        
        ifL(pair_label.work_for('x', 'y'), andL(entity_label.people('x'), entity_label.organization('y')))
        ifL(pair_label.located_in('x', 'y'), andL(entity_label.location('x'), entity_label.location('y')))
        ifL(pair_label.live_in('x', 'y'), andL(entity_label.people('x'), entity_label.location('y')))
        ifL(pair_label.orgbase_on('x', 'y'), andL(entity_label.organization('x'), entity_label.location('y')))
        ifL(pair_label.kill('x', 'y'), andL(entity_label.people('x'), entity_label.people('y')))
