from regr import Graph, Concept
Graph.clear()
Concept.clear()

with Graph('spLanguage') as splang_Graph:

    word = Concept(name ='word')
    phrase = Concept(name = 'phrase')
    sentence = Concept(name = 'sentence')
    phrase.have(word)
    sentence.have(phrase)

    pair = Concept (name= 'pair')
    pair.be((phrase,phrase))

    with Graph('application') as app_graph:
        entity = Concept(name='entity')
        entity.be(phrase)

        tr = Concept(name='trajector')
        lm = Concept(name='landmark')
        o = Concept(name='O')
        tr.be(entity)
        lm.be(entity)
        o.be(entity)

        sp_tr = Concept(name='spr')
        sp_tr.be(pair)
        sp_tr.be((tr, lm))
