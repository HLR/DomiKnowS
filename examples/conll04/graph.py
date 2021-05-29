from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import ifL, andL, exactL, orL, atMostL, atMostI

Graph.clear()
Concept.clear()
Relation.clear()

# Control which logical constrains sets are used
LC_SET_BASED = False
LC_SET_REL = False
LC_SET_ADDITIONAL = True
        
GRAPH_CONSTRAIN = LC_SET_BASED
 
with Graph('global') as graph:
    with Graph('linguistic') as ling_graph:
        word = Concept(name='word')
        phrase = Concept(name='phrase')
        sentence = Concept(name='sentence')
        (rel_sentence_contains_word,) = sentence.contains(word)
        (rel_sentence_contains_phrase,) = sentence.contains(phrase)
        (rel_phrase_contains_word,) = phrase.contains(word)

        pair = Concept(name='pair')
        (rel_pair_phrase1, rel_pair_phrase2) = pair.has_a(arg1=phrase, arg2=phrase)

    with Graph('application', auto_constraint=False) as app_graph:
       
        entity = phrase(name='entity')
        people = entity(name='people', auto_constraint=GRAPH_CONSTRAIN)
        assert people.relate_to(entity)[0].auto_constraint == GRAPH_CONSTRAIN
        organization = entity(name='organization', auto_constraint=False)
        assert organization.relate_to(entity)[0].auto_constraint == False
        location = entity(name='location', auto_constraint=None)
        # auto_constraint->TRUE due to its graph
        assert location.relate_to(entity)[0].auto_constraint == GRAPH_CONSTRAIN
        other = entity(name='other')
        o = entity(name='O')
        
        work_for = pair(name='work_for')
        work_for.has_a(people, organization, auto_constraint=GRAPH_CONSTRAIN)
        assert work_for.relate_to(people)[0].auto_constraint == GRAPH_CONSTRAIN
        assert work_for.relate_to(organization)[0].auto_constraint == GRAPH_CONSTRAIN
        
        located_in = pair(name='located_in')
        located_in.has_a(location, location, auto_constraint=False)
        assert located_in.relate_to(location)[0].auto_constraint == False
        assert located_in.relate_to(location)[1].auto_constraint == False

        live_in = pair(name='live_in')
        live_in.has_a(people, location, auto_constraint=None)
        # auto_constraint->True due to its graph
        assert live_in.relate_to(people)[0].auto_constraint == GRAPH_CONSTRAIN
        assert live_in.relate_to(location)[0].auto_constraint == GRAPH_CONSTRAIN

        orgbase_on = pair(name='orgbase_on')
        kill = pair(name='kill')
        
        # Have exactly one label
        exactL(people, organization, location, other, o,
               active = LC_SET_BASED
               )

        # work_for  -> people, organization
        ifL(
            work_for('x'), 
            andL(people(path=('x', rel_pair_phrase1)), organization(path=('x', rel_pair_phrase2))), 
            active = LC_SET_BASED
            )
        #ifL(andL(pair('x'), people(path=('x', rel_pair_phrase1)), organization(path=('x', rel_pair_phrase2))), work_for(path=('x')))
        
        #  rel_pair_phrase2 is organization - > pair is  work_for
        ifL(
            andL(pair('x'), organization(path=('x', rel_pair_phrase2))),
            work_for(path=('x')), 
            active = LC_SET_REL
            )

        # located_in -> location, location
        ifL(
            located_in('x'), 
            andL(location(path=('x', rel_pair_phrase1)), location(path=('x', rel_pair_phrase2))), 
            active = LC_SET_BASED
            )
        #ifL(andL(pair('x'), location(path=('x', rel_pair_phrase1)), location(path=('x', rel_pair_phrase2))), located_in(path=('x')))
        
        # rel_pair_phrase1 is location -> pair is located_in
        ifL(
            andL(pair('x'), location(path=('x', rel_pair_phrase1))), 
            located_in(path=('x')), 
            active = LC_SET_REL
            )

        # live_in <-> people, location
        ifL(
            live_in('x'), 
            andL(people(path=('x', rel_pair_phrase1)), location(path=('x', rel_pair_phrase2))), 
            active = LC_SET_BASED
            )
        
        ifL(
            andL(pair('x'), people(path=('x', rel_pair_phrase1)), location(path=('x', rel_pair_phrase2))), 
            live_in(path=('x')), 
            active = LC_SET_REL
            )

        # orgbase_on <-> organization, location
        ifL(
            orgbase_on('x'), 
            andL(organization(path=('x', rel_pair_phrase1)), location(path=('x', rel_pair_phrase2))), 
            active = LC_SET_BASED
            )
        
        #ifL(andL(pair('x'), organization(path=('x', rel_pair_phrase1)), location(path=('x', rel_pair_phrase2))), orgbase_on(path=('x')))
        
        # rel_pair_phrase1 is organization -> pair is orgbase_on
        ifL(
            andL(pair('x'), organization(path=('x', rel_pair_phrase1))), 
            orgbase_on(path=('x')), 
            active = LC_SET_REL
            )
       
        # kill -> people, people
        ifL(
            kill('x'), 
            andL(people(path=('x', rel_pair_phrase1)), people(path=('x', rel_pair_phrase2))), 
            active = LC_SET_BASED
            )
        #ifL(andL(pair('x'), people(path=('x', rel_pair_phrase1)), people(path=('x', rel_pair_phrase2))), kill(path=('x')))
        
        # rel_pair_phrase2 is  people -> pair is kill
        ifL(
            andL(pair('x'), people(path=('x', rel_pair_phrase2))), 
            kill(path=('x')), 
            active = LC_SET_REL
            )

        
        # rel_pair_phrase1 is people -> pair is work_for or kill or live_in
        ifL(
            andL(pair('x'), people(path=('x', rel_pair_phrase1)) ), 
            orL(work_for(path=('x')), kill(path=('x')), live_in(path=('x'))), 
            active = LC_SET_REL
            )
        
        # rel_pair_phrase2 is location -> pair is live_in or orgbase_on or located_in
        ifL(
            andL(pair('x'), location(path=('x', rel_pair_phrase2))), 
            orL(live_in(path=('x')), orgbase_on(path=('x')), located_in(path=('x'))), 
            active = LC_SET_REL
            )
        
        # people - at most 1 live in relation with people
        ifL(
            people('p'), 
            atMostL(live_in(path=('p', rel_pair_phrase1.reversed))),
            active = LC_SET_ADDITIONAL
            )
        
        