from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import ifL, andL, nandL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph_ont:
    with Graph('linguistic', auto_constraint=False) as ling_graph:
        ling_graph.ontology = ('http://ontology.ihmc.us/ML/PhraseGraph.owl', './')
       
        word = Concept(name='word')
        phrase = Concept(name='phrase')
        sentence = Concept(name='sentence')
        (rel_sentence_contains_word,) = sentence.contains(word)
        (rel_sentence_contains_phrase,) = sentence.contains(phrase)
        (rel_phrase_contains_word,) = phrase.contains(word)

        pair = Concept(name='pair')
        (rel_pair_phrase1, rel_pair_phrase2, ) = pair.has_a(arg1=phrase, arg2=phrase)

    with Graph('application', auto_constraint=False) as app_graph:
        app_graph.ontology = ('http://ontology.ihmc.us/ML/EMR.owl', './')

        entity = phrase(name='entity')
        people = entity(name='people')
        assert people.relate_to(entity)[0].auto_constraint == False
        organization = entity(name='organization')
        assert organization.relate_to(entity)[0].auto_constraint == False
        location = entity(name='location', auto_constraint=None)
        # auto_constraint->True due to its graph
        assert location.relate_to(entity)[0].auto_constraint == False
        other = entity(name='other')
        o = entity(name='O')

        # nandL(people, organization, location, other, o)

        work_for = pair(name='work_for')
        work_for.has_a(people, organization)
        assert work_for.relate_to(people)[0].auto_constraint == False
        assert work_for.relate_to(organization)[0].auto_constraint == False
        
        located_in = pair(name='located_in')
        located_in.has_a(location, location)
        assert located_in.relate_to(location)[0].auto_constraint == False
        assert located_in.relate_to(location)[1].auto_constraint == False

        live_in = pair(name='live_in')
        live_in.has_a(people, location, auto_constraint=None)
        # auto_constraint->True due to its graph
        assert live_in.relate_to(people)[0].auto_constraint == False
        assert live_in.relate_to(location)[0].auto_constraint == False

        orgbase_on = pair(name='orgbase_on')
        kill = pair(name='kill')