from regr.graph import Graph, Concept, Relation

Graph.clear()
Concept.clear()

with Graph('spLanguage') as splang_Graph:
    splang_Graph.ontology = 'http://ontology.ihmc.us/ML/EMR.owl'



    with Graph('linguistic') as ling_graph:

       word = Concept(name ='word')
       phrase = Concept(name = 'phrase')
       sentence = Concept(name = 'sentence')
       pair = Concept(name='pair')
       pair.has_a(phrase, phrase)



    with Graph('application') as app_graph:
        entity = Concept(name='entity')
        entity.is_a(phrase)

        trajector = Concept(name='TRAJECTOR')
        landmark = Concept(name='LANDMARK')
        noneentity=Concept(name='NONE')
        spatialindicator=Concept(name='SPATIALINDICATOR')
       # o = Concept(name='O')
        trajector.is_a(entity)
        landmark.is_a(entity)
        noneentity.is_a(entity)
        spatialindicator.is_a(entity)
       # o.be(entity)
        trajector.not_a(landmark)
        landmark.not_a(trajector)

        # sp_tr = Concept(name='spr')
        # sp_tr.be(pair)
        # sp_tr.be((tr, lm))
        region = Concept(name='region')
        region.is_a(pair)
        relation_none= Concept(name='relation_none')
        relation_none.is_a(pair)
        distance = Concept(name='distance')
        distance.is_a(pair)
        direction = Concept(name='direction')
        direction.is_a(pair)
        is_triplet=Concept (name='is_triplet')
        # is_not_triplet=Concept (name="is_not_triplet")




