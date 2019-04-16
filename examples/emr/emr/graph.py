from regr import Graph, Concept


with Graph('global') as graph:
    with Graph('linguistic') as ling_graph:
        word = Concept(name='word')
        phrase = Concept(name='phrase')
        sentence = Concept(name='sentence')
        phrase.have(word)
        sentence.have(phrase)

        #nn = Concept(name='NN')
        #nn.is(phrase)
        # ...


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
        #entity.be(people or ororganization or location or other or o)

        work_for = Concept(name='work_for')
        work_for.be((people, organization))

        located_in = Concept(name='located_in')
        located_in.be((organization, location))

        live_in = Concept(name='live_in')
        live_in.be((people, location))

        orgbase_on = Concept(name='orgbase_on')
        orgbase_on.be((organization, location))
