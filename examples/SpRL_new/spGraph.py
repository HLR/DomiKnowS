from regr.graph import Graph, Concept, Relation

Graph.clear()
Concept.clear()

with Graph('spLanguage') as splang_Graph:
    splang_Graph.ontology = ('http://ontology.ihmc.us/ML/SPRL.owl', './')


    with Graph('linguistic') as ling_graph:
        ling_graph.ontology = ('http://ontology.ihmc.us/ML/PhraseGraph.owl', './')
        word = Concept(name ='word')
        phrase = Concept(name = 'phrase')
        sentence = Concept(name = 'sentence')
        phrase.has_many(word)
        sentence.has_many(phrase)

    with Graph('application') as app_graph:
        splang_Graph.ontology = ('http://ontology.ihmc.us/ML/SPRL.owl', './')

        trajector = Concept(name='TRAJECTOR')
        landmark = Concept(name='LANDMARK')
        noneentity = Concept(name='NONE')
        spatialindicator = Concept(name='SPATIALINDICATOR')
        trajector.is_a(phrase)
        landmark.is_a(phrase)
        noneentity.is_a(phrase)
        spatialindicator.is_a(phrase)

        trajector.not_a(noneentity)
        trajector.not_a(spatialindicator)
        landmark.not_a(noneentity)
        landmark.not_a(spatialindicator)
        noneentity.not_a(trajector)
        noneentity.not_a(landmark)
        noneentity.not_a(spatialindicator)
        spatialindicator.not_a(trajector)
        spatialindicator.not_a(landmark)
        spatialindicator.not_a(noneentity)

        triplet = Concept(name='triplet')
        region = Concept(name='region')

        region.is_a(triplet)
        relation_none= Concept(name='relation_none')
        relation_none.is_a(triplet)
        distance = Concept(name='distance')
        distance.is_a(triplet)
        direction = Concept(name='direction')
        direction.is_a(triplet)
