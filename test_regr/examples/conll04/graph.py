from itertools import permutations

from regr.graph import Graph, Concept, Relation
from regr.graph.relation import disjoint
from regr.graph.logicalConstrain import ifL, andL, nandL


Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    with Graph('linguistic') as ling_graph:
        char = Concept(name='char')
        word = Concept(name='word')
        phrase = Concept(name='phrase')
        sentence = Concept(name='sentence')
        (rel_sentence_contains_word,) = sentence.contains(word)
        (rel_sentence_contains_phrase,) = sentence.contains(phrase)
        (rel_phrase_contains_word,) = phrase.contains(word)
        (rel_word_contains_char,) = word.contains(char)

        pair = Concept(name='pair')
        (rel_pair_word1, rel_pair_word2, ) = pair.has_a(arg1=word, arg2=word)

    with Graph('application') as app_graph:
        people = word(name='people')
        organization = word(name='organization')
        location = word(name='location')
        other = word(name='other')
        o = word(name='O')

        disjoint(people, organization, location, other, o)

        #nandL(people,organization)
        #for c1, c2 in permutations((people, organization, location, other, o), r=2):
        #   nandL(c1, c2)

        work_for = pair(name='work_for')
        located_in = pair(name='located_in')
        live_in = pair(name='live_in')
        orgbase_on = pair(name='orgbase_on')
        kill = pair(name='kill')

        work_for.has_a(people, organization)
        located_in.has_a(location, location)
        live_in.has_a(people, location)
        orgbase_on.has_a(organization, location)
        kill.has_a(people, people)

        # LC1
        ifL(work_for, ('x', 'y'), andL(people, ('x',), organization, ('y',)))
        
        #ifL(located_in, ('x', 'y'), andL(location, ('x',), location, ('y',)))
        #ifL(live_in, ('x', 'y'), andL(people, ('x',), location, ('y',)))
        #ifL(orgbase_on, ('x', 'y'), andL(organization, ('x',), location, ('y',)))
        #ifL(kill, ('x', 'y'), andL(people, ('x',), people, ('y',)))
