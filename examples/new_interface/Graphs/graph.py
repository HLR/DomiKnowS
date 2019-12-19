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
        (rel_sentence_contains_phrase,) = sentence.contains(phrase)
        (rel_sentence_contains_word,) = sentence.contains(word)
        (rel_phrase_contains_word,) = phrase.contains(word)
        pair = Concept('pair')
        (rel_pair_phrase1, rel_pair_phrase2, ) = pair.has_a(phrase, phrase)

    with Graph('application') as app_graph:
        FAC = phrase(name='FAC')
        GPE = phrase(name='GPE')
        LOC = phrase(name='LOC')
        ORG = phrase(name='ORG')
        PER = phrase(name='PER')
        VEH = phrase(name='VEH')
        WEA = phrase(name='WEA')

        ART = pair(name="ART")
        GEN_AFF = pair(name="GEN-AFF")
        METONYMY = pair(name="METONYMY")
        ORG_AFF = pair(name="ORG-AFF")
        PART_WHOLE = pair(name="PART-WHOLE")
        PER_SOC = pair(name="PER-SOC")
        PHYS = pair(name="PHYS")

        FAC.not_a(GPE, LOC, VEH, WEA, PER, ORG)

        GPE.not_a(FAC, LOC, VEH, WEA, PER, ORG)

        LOC.not_a(FAC, GPE, VEH, WEA, PER, ORG)

        VEH.not_a(FAC, GPE, LOC, WEA, PER, ORG)

        WEA.not_a(FAC, GPE, LOC, VEH, PER, ORG)

        PER.not_a(FAC, GPE, LOC, VEH, WEA, ORG)

        ORG.not_a(FAC, GPE, LOC, VEH, WEA, PER)
