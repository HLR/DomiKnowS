from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import ifL, andL, nandL, exactL
from domiknows.graph.relation import disjoint


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

    with Graph('application', auto_constraint=True) as app_graph:
        people = word(name='people')
        organization = word(name='organization')
        location = word(name='location')
        other = word(name='other')
        o = word(name='O')

        #disjoint(people, organization, location, other, o)

        # LC0
        nandL(people, organization, active = True)
        
        work_for = pair(name='work_for')
        located_in = pair(name='located_in')
        live_in = pair(name='live_in')
        orgbase_on = pair(name='orgbase_on')
        kill = pair(name='kill')
        
        # LC1
        ifL(word('x'), exactL(people(path=('x')), organization(path=('x')), location(path=('x')), other(path=('x')), o(path=('x'))), active = True)
        
        # LC2
        ifL(pair('x'), exactL(work_for(path=('x')), located_in(path=('x')), live_in(path=('x')), orgbase_on(path=('x')), kill(path=('x'))), active = True)
        
        # LC3
        #ifL(existsL(work_for('x')), andL(people(path=('x', rel_pair_word1.name)), organization(path=('x', rel_pair_word2.name))), active = True)
        ifL(work_for('x', 'y'), andL(people('x'), organization('y')), active = True)
        
        # LC4
        ifL(located_in('x', 'y'), andL(location('x'), organization('y')), active = True)
        
        # LC5
        ifL(live_in('x', 'y'), andL(people('x'), location('y')), active = True)
        
        # LC6
        ifL(orgbase_on('x', 'y'), andL(organization('x'), location('y')), active = True)
        
        # LC7
        ifL(kill('x', 'y'), andL(people('x'), people('y')), active = True)
        
