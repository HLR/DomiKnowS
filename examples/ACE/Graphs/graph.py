from regr.graph import Graph, Concept, Relation


Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    graph.ontology = ('http://ontology.ihmc.us/ML/ACE.owl', './')

    with Graph('linguistic') as ling_graph:
        # phrase = Concept(name='phrase')
        word = Concept(name='word')
        sentence = Concept(name='sentence')
        # sentence.has_many(phrase)

        # pair = Concept(name='pair')
        # pair.has_a(phrase, phrase)

    with Graph('application') as app_graph:
        entity = Concept(name='entity')
        # entity.is_a(phrase)

        FAC = Concept(name='FAC')
        GPE = Concept(name='GPE')
        LOC = Concept(name='LOC')
        ORG = Concept(name='ORG')
        PER = Concept(name='PER')
        VEH = Concept(name='VEH')
        WEA = Concept(name='WEA')

        FAC.is_a(entity)
        GPE.is_a(entity)
        LOC.is_a(entity)
        VEH.is_a(entity)
        WEA.is_a(entity)
        PER.is_a(entity)
        ORG.is_a(entity)

        FAC.not_a(GPE)
        FAC.not_a(LOC)
        FAC.not_a(VEH)
        FAC.not_a(WEA)
        FAC.not_a(PER)
        FAC.not_a(ORG)

        GPE.not_a(FAC)
        GPE.not_a(LOC)
        GPE.not_a(VEH)
        GPE.not_a(WEA)
        GPE.not_a(PER)
        GPE.not_a(ORG)

        LOC.not_a(FAC)
        LOC.not_a(GPE)
        LOC.not_a(VEH)
        LOC.not_a(WEA)
        LOC.not_a(PER)
        LOC.not_a(ORG)

        VEH.not_a(FAC)
        VEH.not_a(GPE)
        VEH.not_a(LOC)
        VEH.not_a(WEA)
        VEH.not_a(PER)
        VEH.not_a(ORG)

        WEA.not_a(FAC)
        WEA.not_a(GPE)
        WEA.not_a(LOC)
        WEA.not_a(VEH)
        WEA.not_a(PER)
        WEA.not_a(ORG)

        PER.not_a(FAC)
        PER.not_a(GPE)
        PER.not_a(LOC)
        PER.not_a(VEH)
        PER.not_a(WEA)
        PER.not_a(ORG)

        ORG.not_a(FAC)
        ORG.not_a(GPE)
        ORG.not_a(LOC)
        ORG.not_a(VEH)
        ORG.not_a(WEA)
        ORG.not_a(PER)
