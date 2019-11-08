from regr.graph import Concept, Relation, Graph
# from .base import NewGraph as Graph

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    graph.ontology = ('http://ontology.ihmc.us/ML/ACE.owl', './')

    with Graph('linguistic') as ling_graph:
        word = Concept('word')
        phrase = Concept('phrase')
        sentence = Concept('sentence')
        sentence.contains(phrase)
        sentence.contains(word)
        phrase.contains(word)
        pair = Concept('pair')
        pair.has_a(phrase, phrase)

    with Graph('application') as app_graph:
        entity = phrase('entity')

        FAC = entity(name='FAC')
        GPE = entity(name='GPE')
        LOC = entity(name='LOC')
        ORG = entity(name='ORG')
        PER = entity(name='PER')
        VEH = entity(name='VEH')
        WEA = entity(name='WEA')

        FAC.not_a(GPE, LOC, VEH, WEA, PER, ORG)

        GPE.not_a(FAC, LOC, VEH, WEA, PER, ORG)

        LOC.not_a(FAC, GPE, VEH, WEA, PER, ORG)

        VEH.not_a(FAC, GPE, LOC, WEA, PER, ORG)

        WEA.not_a(FAC, GPE, LOC, VEH, PER, ORG)

        PER.not_a(FAC, GPE, LOC, VEH, WEA, ORG)

        ORG.not_a(FAC, GPE, LOC, VEH, WEA, PER)