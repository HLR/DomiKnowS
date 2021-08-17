from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import ifL, andL, nandL
from regr.graph import EnumConcept


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
#         app_graph.ontology = ('http://ontology.ihmc.us/ML/EMR.owl', './')

        entity = phrase(name='entity')
        entity_label = entity(name="entity_label", ConceptClass=EnumConcept, values=["people", "organization", 'location', 'other', 'O'])
#         people = entity(name='people')
#         organization = entity(name='organization')
#         location = entity(name='location')
#         other = entity(name='other')
#         o = entity(name='O')

        # nandL(people, organization, location, other, o)
        pair_label = pair(name="pair_label", ConceptClass=EnumConcept, values=["work_for", "located_in", 'live_in', 'orgbase_on', 'kill'])
#         work_for = pair(name='work_for')
#         work_for.has_a(people, organization)
        
#         located_in = pair(name='located_in')
#         located_in.has_a(location, location)

#         live_in = pair(name='live_in')
#         live_in.has_a(people, location)

#         orgbase_on = pair(name='orgbase_on')
#         kill = pair(name='kill')
        

        # ifL(work_for, ('x', 'y'), andL(people, ('x',), organization, ('y',)))
        # ifL(located_in, ('x', 'y'), andL(location, ('x',), location, ('y',)))
        # ifL(live_in, ('x', 'y'), andL(people, ('x',), location, ('y',)))
        # ifL(orgbase_on, ('x', 'y'), andL(organization, ('x',), location, ('y',)))
        # ifL(kill, ('x', 'y'), andL(people, ('x',), people, ('y',)))
