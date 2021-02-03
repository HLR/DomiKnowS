from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import ifL, andL, nandL


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
        (rel_pair_phrase1, rel_pair_phrase2, ) = pair.has_a(arg1=phrase, arg2=phrase)

    with Graph('application') as app_graph:
        people = phrase(name='people')
        organization = phrase(name='organization')
        location = phrase(name='location')
        other = phrase(name='other')
        o = phrase(name='O')

        # nandL(people, organization, location, other, o)

        work_for = pair(name='work_for')
        located_in = pair(name='located_in')
        live_in = pair(name='live_in')
        orgbase_on = pair(name='orgbase_on')
        kill = pair(name='kill')

        # ifL(work_for, ('x', 'y'), andL(people, ('x',), organization, ('y',)))
        # ifL(located_in, ('x', 'y'), andL(location, ('x',), location, ('y',)))
        # ifL(live_in, ('x', 'y'), andL(people, ('x',), location, ('y',)))
        # ifL(orgbase_on, ('x', 'y'), andL(organization, ('x',), location, ('y',)))
        # ifL(kill, ('x', 'y'), andL(people, ('x',), people, ('y',)))
