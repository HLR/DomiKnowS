from regr.graph import Graph, Concept, Relation
from regr.graph.relation import disjoint
from regr.graph.logicalConstrain import ifL, andL


Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    graph.ontology = ('http://ontology.ihmc.us/ML/EMR.owl', './')

    with Graph('linguistic') as ling_graph:
        word = Concept(name='word')
        sentence = Concept(name='sentence')
        (rel_sentence_contains_word,) = sentence.contains(word)

        pair = Concept(name='pair')
        (rel_pair_word1, rel_pair_word2, ) = pair.has_a(word, word)

    with Graph('application') as app_graph:
        people = word(name='people')
        organization = word(name='organization')
        location = word(name='location')
        other = word(name='other')
        o = word(name='O')

        disjoint(people, organization, location, other, o)

        for c1, c2 in permutations((people, organization, location, other, o), r=2):
            nandL(c1, c2)

        work_for = pair(name='work_for')
        located_in = Concept(name='located_in')
        live_in = Concept(name='live_in')
        orgbase_on = Concept(name='orgbase_on')
        kill = Concept(name='kill')

        work_for.has_a(people, organization)
        located_in.has_a(location, location)
        live_in.has_a(people, location)
        orgbase_on.has_a(organization, location)
        kill.has_a(people, people)

        ifL(work_for, ('x', 'y'), andL(people, ('x',), organization, ('y',)))
        ifL(located_in, ('x', 'y'), andL(location, ('x',), location, ('y',)))
        ifL(live_in, ('x', 'y'), andL(people, ('x',), location, ('y',)))
        ifL(orgbase_on, ('x', 'y'), andL(organization, ('x',), location, ('y',)))
        ifL(kill, ('x', 'y'), andL(people, ('x',), people, ('y',)))
