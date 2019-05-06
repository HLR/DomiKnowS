from regr import Graph, Concept


Graph.clear()
Concept.clear()

# simpler example just fit the case
with Graph('global') as graph:
    word = Concept(name='word')

    people = Concept(name='people')
    organization = Concept(name='organization')
    people.be(word)
    organization.be(word)

    pair = Concept(name='pair')
    pair.be((word, word))
    workfor = Concept(name='workfor')
    workfor.be({'employee': people, 'employer': organization})
