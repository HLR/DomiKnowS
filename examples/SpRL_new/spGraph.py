from regr.graph import Graph, Concept, Relation
Graph.clear()
Concept.clear()

with Graph('spLanguage') as splang_Graph:
    splang_Graph.ontology = 'http://ontology.ihmc.us/ML/EMR.owl'


    with Graph('linguistic') as ling_graph:

       word = Concept(name ='word')
       phrase = Concept(name = 'phrase')
       sentence = Concept(name = 'sentence')


    # pair = Concept (name= 'pair')
    # pair.be((phrase,phrase))

    with Graph('application') as app_graph:
        entity = Concept(name='entity')
        entity.is_a(phrase)

        trajector = Concept(name='Trajector')
        landmark = Concept(name='Landmark')
        noneentity=Concept(name='None')
       # o = Concept(name='O')
        trajector.is_a(entity)
        landmark.is_a(entity)
        noneentity.is_a(entity)
       # o.be(entity)
       #  trajector.not_a(landmark)
       #  landmark.not_a(trajector)

        # sp_tr = Concept(name='spr')
        # sp_tr.be(pair)
        # sp_tr.be((tr, lm))


