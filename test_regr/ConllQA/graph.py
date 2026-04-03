from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import ifL, andL, nandL
from domiknows.graph import EnumConcept
from itertools import combinations


Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    from domiknows.graph.logicalConstrain import orL, existsL, ifL, notL, andL, atMostAL, atLeastAL, exactAL, atMostL, \
        atLeastL, exactL

    word = Concept(name='word')
    phrase = Concept(name='phrase')
    sentence = Concept(name='sentence')
    (rel_sentence_contains_word,) = sentence.contains(word)
    (rel_sentence_contains_phrase,) = sentence.contains(phrase)
    (rel_phrase_contains_word,) = phrase.contains(word)


    pair = Concept(name='pair')
    (rel_pair_phrase1, rel_pair_phrase2,) = pair.has_a(arg1=phrase, arg2=phrase)

    # define entity
    entity = phrase(name='entity')
    people = entity(name='people')
    organization = entity(name='organization')
    location = entity(name='location',)
    other = entity(name='other')
    o = entity(name='O')

    # define relation
    work_for = pair(name='work_for')
    work_for.has_a(people, organization)

    located_in = pair(name='located_in')
    located_in.has_a(location, location)

    live_in = pair(name='live_in')
    live_in.has_a(people, location)

    orgbase_on = pair(name='orgbase_in')
    orgbase_on.has_a(organization, location)

    kill = pair(name='kill')
    kill.has_a(people, people)

