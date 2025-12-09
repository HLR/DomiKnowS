from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import ifL, andL, nandL, exactL, sumL, atLeastL, atMostL


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
        (rel_pair_word1, rel_pair_word2,) = pair.has_a(arg1=word, arg2=word)

    with Graph('application', auto_constraint=True) as app_graph:
        people = word(name='people')
        organization = word(name='organization')
        location = word(name='location')
        other = word(name='other')
        o = word(name='O')

        #nandL(people, organization, active=True)

        work_for = pair(name='work_for')
        located_in = pair(name='located_in')
        live_in = pair(name='live_in')
        orgbase_on = pair(name='orgbase_on')
        kill = pair(name='kill')

        # LC: exactL for word entity types
        ifL(word('x'), exactL(people(path=('x',)), organization(path=('x',)), location(path=('x',)), other(path=('x',)), o(path=('x',))), active=True)

        # LC: work_for requires (people, organization)
        ifL(work_for('x', 'y'), andL(people('x'), organization('y')), active=True)

        # Counting constraint: count valid work_for relations (at least 2)
        atLeastL(
            sumL(
                andL(
                    people('a'),
                    work_for('b', path=('a', rel_pair_word1.reversed)),
                    organization('c', path=('b', rel_pair_word2))
                )
            ),
            2,
            active=True
        )

        # Counting constraint: count valid work_for relations (at most 3)
        atMostL(
            sumL(
                andL(
                    people('a'),
                    work_for('b', path=('a', rel_pair_word1.reversed)),
                    organization('c', path=('b', rel_pair_word2))
                )
            ),
            3,
            active=True
        )