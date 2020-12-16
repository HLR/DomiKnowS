from itertools import permutations

from regr.graph import Graph, Concept, Relation
from regr.graph.relation import disjoint
from regr.graph.logicalConstrain import V, ifL, andL, nandL, orL, atLeastL


Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    with Graph('linguistic') as ling_graph:
        word = Concept(name='word')
        
        sentence = Concept(name='sentence')
        (rel_sentence_contains_word,) = sentence.contains(word)
        
        phrase = Concept(name='phrase')
        (rel_sentence_contains_phrase,) = sentence.contains(arg=phrase)
        (rel_phrase_word1, rel_phrase_word2) = phrase.has_a(word, word)

        pair = Concept(name='pair')
        (rel_pair_phrase1, rel_pair_phrase2) = pair.has_a(arg1=phrase, arg2=phrase)

    with Graph('application') as app_graph:
        # word concepts
        Oword = word(name="Oword")
        Bword = word(name="Bword")
        Iword = word(name="Iword")
        Eword = word(name="Eword")
        
        # phrase concepts
        RealPhrase = phrase(name="real_phrase")
        people = phrase(name='people')
        organization = phrase(name='organization')
        location = phrase(name='location')
        other = phrase(name='other')
        o = phrase(name='O')

        # phrase constrains
        disjoint(people, organization, location, other, o)
        
        #nandL(people,  ('x',), organization, V(name='y', v=('x',)))
        #nandL(people, organization, V(name='y', v=('x',)))
        
        #nandL(people,  V(name='x'), organization, V())
        
        #LC2 Each sentence should contain at least one person phrase        
        atLeastL(andL(sentence, V(name='x'), people, V(name='y', v=('x', rel_sentence_contains_word.name))), V(name='a'), 1, 'y')
        
        nandL(people, V(name='x'), organization, V(match='x'))
        
        #nandL(people,  V(name='x'), location, V(v=('x',)))

        #for c1, c2 in permutations((people, organization, location, other, o), r=2):
        #nandL(c1, c2)

        # Pair relations
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
        ifL(work_for, V(name='x'), andL(people, V(name='y', v=('x', rel_pair_phrase1.name)), organization, V(name='z', v=('x', rel_pair_phrase2.name))), V(match='x'))
            
        # LC1bis if x is people and y is organization then they are in work_for relation
        ifL(andL(people, V(name='x'), organization, V(name='y')), V(name='z'), work_for, V(match='z', v=(('x', rel_pair_phrase1.backward.name), ('y', rel_pair_phrase2.backward.name))))

        
        #LC3 each real phrase is either the same word starting and end with type arg1=arg2=Iword or two different words with arg1 is Bword and arg2 is Eword
        ifL(phrase, V(name='x'), orL(andL(Iword, V(name='y', v=('x', rel_phrase_word1.name)), Iword, V(match='y', v=('x', rel_phrase_word2.name))), V(name='o'), andL(Bword, V(name='z', v=('x', rel_phrase_word1.name)), Eword, V(match='z', v=('x', rel_phrase_word2.name))), V(match='o')), V(match='x'))
