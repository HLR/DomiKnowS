from regr.graph import Graph, Concept, Relation

Graph.clear()
Concept.clear()

with Graph('spLanguage') as splang_Graph:
    splang_Graph.ontology = ('http://ontology.ihmc.us/ML/SPRL.owl', './')

    with Graph('linguistic') as ling_graph:
        ling_graph.ontology = ('http://ontology.ihmc.us/ML/PhraseGraph.owl', './')
        word = Concept(name='word')
        phrase = Concept(name = 'phrase')
        sentence = Concept(name = 'sentence')
        
        phrase.has_many(word)
        sentence.has_many(phrase)

    with Graph('application') as app_graph:
        splang_Graph.ontology = ('http://ontology.ihmc.us/ML/SPRL.owl', './')

        trajector = Concept(name='TRAJECTOR')
        landmark = Concept(name='LANDMARK')
        none_entity = Concept(name='NONE_ENTITY')
        spatial_indicator = Concept(name='SPATIAL_INDICATOR')
        trajector.is_a(phrase)
        landmark.is_a(phrase)
        none_entity.is_a(phrase)
        spatial_indicator.is_a(phrase)

        spatial_indicator.not_a(trajector)
        spatial_indicator.not_a(landmark)
        none_entity.not_a(trajector)
        none_entity.not_a(landmark)
        none_entity.not_a(spatial_indicator)

        triplet = Concept(name='triplet')
        triplet.has_a(first=phrase, second=phrase, third=phrase)

        spatial_triplet = Concept(name='spatial_triplet')
        spatial_triplet.is_a(triplet)
        spatial_triplet.has_a(first=landmark)
        spatial_triplet.has_a(second=trajector)
        spatial_triplet.has_a(third=spatial_indicator)

        region = Concept(name='region')
        region.is_a(spatial_triplet)
        distance = Concept(name='distance')
        distance.is_a(spatial_triplet)
        direction = Concept(name='direction')
        direction.is_a(spatial_triplet)

        none_relation= Concept(name='none_relation')
        none_relation.is_a(triplet)
        none_relation.not_a(spatial_triplet)
