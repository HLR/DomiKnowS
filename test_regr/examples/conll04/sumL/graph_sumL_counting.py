from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import ifL, andL, exactL, sumL


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

    with Graph('application', auto_constraint=True) as app_graph:
        people = word(name='people')
        organization = word(name='organization')
        location = word(name='location')
        other = word(name='other')
        o = word(name='O')

        # LC: exactL for word entity types (mutual exclusivity)
        ifL(word('x'), exactL(people('x'), organization('x'), location('x'), other('x'), o('x')), active=True)

        # Constraint 1: Separate counts - count(people) + count(organizations)
        # This adds the counts: if 3 people and 2 orgs, result is 5
        sumL(people('x'), organization('y'), active=True)

        # Constraint 2: Overlap count - entities classified as BOTH person AND organization
        # With mutual exclusivity constraint above, this should be 0
        sumL(andL(people('x'), organization('x')), active=True)