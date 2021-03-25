from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import ifL, andL, atMostL, V


Graph.clear()
Concept.clear()
Relation.clear()

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

    with Graph('application', auto_constraint=True) as app_graph:
        entity = phrase(name='entity')
        people = entity(name='people', auto_constraint=True)
        assert people.relate_to(entity)[0].auto_constraint == True
        organization = entity(name='organization', auto_constraint=False)
        assert organization.relate_to(entity)[0].auto_constraint == False
        location = entity(name='location', auto_constraint=None)
        # auto_constraint->True due to its graph
        assert location.relate_to(entity)[0].auto_constraint == True
        other = entity(name='other')
        o = entity(name='O')

        #atMostL(people, organization, location, other, o)

        work_for = pair(name='work_for')
        work_for.has_a(people, organization, auto_constraint=True)
        assert work_for.relate_to(people)[0].auto_constraint == True
        assert work_for.relate_to(organization)[0].auto_constraint == True
        
        located_in = pair(name='located_in')
        located_in.has_a(location, location, auto_constraint=False)
        assert located_in.relate_to(location)[0].auto_constraint == False
        assert located_in.relate_to(location)[1].auto_constraint == False

        live_in = pair(name='live_in')
        live_in.has_a(people, location, auto_constraint=None)
        # auto_constraint->True due to its graph
        assert live_in.relate_to(people)[0].auto_constraint == True
        assert live_in.relate_to(location)[0].auto_constraint == True

        orgbase_on = pair(name='orgbase_on')
        kill = pair(name='kill')

        #ifL(work_for, ('x', 'y'), andL(people, ('x',), organization, ('y',)))
        ifL(work_for, V(name='x'), andL(people, V(v=('x', rel_pair_phrase1.name)), organization, V(v=('x', rel_pair_phrase2.name))))

        #ifL(located_in, ('x', 'y'), andL(location, ('x',), location, ('y',)))
        ifL(located_in, V(name='x'), andL(location, V(v=('x', rel_pair_phrase1.name)), location, V(v=('x', rel_pair_phrase2.name))))
        
        #ifL(live_in, ('x', 'y'), andL(people, ('x',), location, ('y',)))
        ifL(live_in, V(name='x'), andL(people, V(v=('x', rel_pair_phrase1.name)), location, V(v=('x', rel_pair_phrase2.name))))

        #ifL(orgbase_on, ('x', 'y'), andL(organization, ('x',), location, ('y',)))
        ifL(orgbase_on, V(name='x'), andL(organization, V(v=('x', rel_pair_phrase1.name)), location, V(v=('x', rel_pair_phrase2.name))))
        
        #ifL(kill, ('x', 'y'), andL(people, ('x',), people, ('y',)))
        ifL(kill, V(name='x'), andL(people, V(v=('x', rel_pair_phrase1.name)), people, V(v=('x', rel_pair_phrase2.name))))
