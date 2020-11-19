from itertools import permutations

from regr.graph import Graph, Concept, Relation
from regr.graph.relation import disjoint
from regr.graph.logicalConstrain import ifL, andL, nandL, orL, atLeastL


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
        (rel_phrase_word1, rel_phrase_word2) = phrase.has_a(word, word)

        pair = Concept(name='pair')
        (rel_pair_phrase1, rel_pair_phrase2, ) = pair.has_a(arg1=phrase, arg2=phrase)

    with Graph('application') as app_graph:
        Oword = word(name="Oword")
        Bword = word(name="Bword")
        Iword = word(name="Iword")
        Eword = word(name="Eword")
        RealPhrase = phrase(name="real_phrase")
        people = phrase(name='people')
        organization = phrase(name='organization')
        location = phrase(name='location')
        other = phrase(name='other')
        o = phrase(name='O')

        disjoint(people, organization, location, other, o)

        nandL(people,organization)
        #for c1, c2 in permutations((people, organization, location, other, o), r=2):
        #nandL(c1, c2)

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

        # LC1 work_for has arg1 people and arg2 organization
        ifL(work_for, ('x',), andL(people, ('x','arg1'), organization, ('x','arg2')))
        
        #LC2 Each sentence should contain at least one person phrase        
        atLeastL(1, ('x', 'arg1'), andL(rel_sentence_contains_phrase, ('x',), phrase, ('x', 'arg1')))
        
        #LC3 each real phrase is either the same word starting and end with type arg1=arg2=Iword or two different words with arg1 is Bword and arg2 is Eword
        ifL(phrase, ('x',), orL(andL(Iword ('x', 'arg1'), Iword ('x', 'arg2')), andL(Bword, ('x', 'arg1'), Eword, ('x', 'arg2'))))
        