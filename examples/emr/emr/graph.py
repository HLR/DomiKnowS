from regr import Graph, Concept


Graph.clear()
Concept.clear()

with Graph('global') as graph:
    graph.ontology='http://ontology.ihmc.us/ML/EMR.owl'

    with Graph('linguistic') as ling_graph:
        word = Concept(name='word')
        phrase = Concept(name='phrase')
        sentence = Concept(name='sentence')
        phrase.have(word)
        sentence.have(phrase)

        # if we need POS tags, for example, we can have
        #nn = Concept(name='NN')
        #nn.is(phrase)
        # ...

        pair = Concept(name='pair')
        pair.be((phrase, phrase))

    with Graph('application') as app_graph:
        entity = Concept(name='entity')
        entity.be(phrase)

        people = Concept(name='people')
        organization = Concept(name='organization')
        location = Concept(name='location')
        other = Concept(name='other')
        o = Concept(name='O')
        people.be(entity)
        organization.be(entity)
        location.be(entity)
        other.be(entity)
        o.be(entity)

        work_for = Concept(name='work_for')
        work_for.be(pair)
        work_for.be((people, organization))

        located_in = Concept(name='located_in')
        located_in.be(pair)
        located_in.be((location, location))

        live_in = Concept(name='live_in')
        live_in.be(pair)
        live_in.be((people, location))

        orgbase_on = Concept(name='orgbase_on')
        orgbase_on.be(pair)
        #orgbase_on.be((organization, location))
