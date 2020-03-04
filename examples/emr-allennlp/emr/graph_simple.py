from regr.graph import Graph, Concept, Relation


Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    graph.ontology = 'http://ontology.ihmc.us/ML/EMR.owl'

    with Graph('linguistic') as ling_graph:
        word = Concept(name='word')
        phrase = Concept(name='phrase')
        sentence = Concept(name='sentence')
        phrase.has_many(word)
        sentence.has_many(phrase)

        pair = Concept(name='pair')
        pair.has_a(phrase, phrase)

    with Graph('application') as app_graph:
        entity = Concept(name='entity')
        entity.is_a(phrase)

        people = Concept(name='people')
        organization = Concept(name='organization')

        people.is_a(entity)
        organization.is_a(entity)

        people.not_a(organization)
        organization.not_a(people)

        work_for = Concept(name='work_for')
        work_for.is_a(pair)
        work_for.has_a(people, organization)
