from regr.graph import Concept, Relation, Graph
# from .base import NewGraph as Graph
from regr.graph.logicalConstrain import *

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
        FAC = word(name='FAC')
        GPE = word(name='GPE')
        LOC = word(name='LOC')
        ORG = word(name='ORG')
        PER = word(name='PER')
        VEH = word(name='VEH')
        WEA = word(name='WEA')

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

        ART.not_a(GEN_AFF, ORG_AFF, PART_WHOLE, PER_SOC, PHYS)
        GEN_AFF.not_a(ART, ORG_AFF, PART_WHOLE, PER_SOC, PHYS)
        ORG_AFF.not_a(ART, GEN_AFF, PART_WHOLE, PER_SOC, PHYS)
        PART_WHOLE.not_a(ART, GEN_AFF, ORG_AFF, PER_SOC, PHYS)
        PER_SOC.not_a(ART, GEN_AFF, ORG_AFF, PART_WHOLE, PHYS)
        PHYS.not_a(ART, GEN_AFF, ORG_AFF, PART_WHOLE, PER_SOC)

        # list_of_concepts = [GPE, FAC, LOC, VEH, WEA, PER, ORG]
        #
        # for item in list_of_concepts:
        #     for item1 in list_of_concepts:
        #         if item != item1:
        #             nandL(item, item1)

        ifL(PHYS, ('x', 'y'), ifL(PER, ('x',), orL(LOC, FAC, GPE), ('y',)))

        # 'PHYS': ['FAC', 'LOC'], ['FAC', 'FAC'], ['FAC', 'GPE']
        ifL(PHYS, ('x', 'y'), ifL(FAC, ('x',), orL(LOC, FAC, GPE, PER, ORG), ('y',)))

        # 'PHYS': ['LOC', 'LOC'], ['LOC', 'GPE'], ['LOC', 'FAC'],  ['LOC', 'PER']
        ifL(PHYS, ('x', 'y'), ifL(LOC, ('x',), orL(LOC, GPE, FAC, PER), ('y',)))

        # 'PHYS': ['GPE', 'LOC'], ['GPE', 'GPE'], ['GPE', 'FAC'], ['GPE', 'PER']
        ifL(PHYS, ('x', 'y'), ifL(GPE, ('x',), orL(LOC, GPE, FAC, PER, ORG), ('y',)))

        # 'PHYS': ['ORG', 'GPE'], ['ORG', 'FAC']
        ifL(PHYS, ('x', 'y'), ifL(ORG, ('x',), orL(GPE, FAC), ('y',)))


        # 'GEN-AFF': ['PER', 'GPE'], ['PER', 'PER'], ['PER', 'LOC'], ['PER', 'ORG']
        ifL(GEN_AFF, ('x', 'y'), ifL(PER, ('x',), orL(GPE, PER, LOC, ORG), ('y',)))

        # 'GEN-AFF': ['ORG', 'LOC'], ['ORG', 'GPE'], ['ORG', 'ORG']
        ifL(GEN_AFF, ('x', 'y'), ifL(ORG, ('x',), orL(LOC, GPE, ORG, PER), ('y',)))

        # 'GEN-AFF':['GPE', 'PER']
        ifL(GEN_AFF, ('x', 'y'), ifL(GPE, ('x',), orL(PER, ORG), ('y',)))

        ifL(GEN_AFF, ('x', 'y'), ifL(LOC, ('x', ), orL(PER, ORG), ('y',)))


        # 'ORG-AFF':['ORG', 'PER'], ['ORG', 'GPE'], ['ORG', 'ORG'],
        ifL(ORG_AFF, ('x', 'y'), ifL(ORG, ('x',), orL(PER, GPE, ORG), ('y',)))

        # 'ORG-AFF':['PER', 'PER'], ['PER', 'GPE'], ['PER', 'ORG'],
        ifL(ORG_AFF, ('x', 'y'), ifL(PER, ('x',), orL(PER, GPE, ORG), ('y',)))

        # 'ORG-AFF':['GPE', 'PER'],  ['GPE', 'ORG'],
        ifL(ORG_AFF, ('x', 'y'), ifL(GPE, ('x',), orL(PER, ORG), ('y',)))


        # 'PER-SOC':['PER', 'PER'],  ['PER', 'ORG'],
        ifL(PER_SOC, ('x', 'y'), ifL(PER, ('x',), orL(PER, ORG), ('y',)))

        # 'PER-SOC':['ORG', 'PER']
        ifL(PER_SOC, ('x', 'y'), ifL(ORG, ('x',), PER, ('y',)))


        # 'PART-WHOLE':['LOC', 'GPE'], ['LOC', 'LOC'], ['LOC', 'FAC']
        ifL(PART_WHOLE, ('x', 'y'), ifL(LOC, ('x',), orL(GPE, LOC, FAC), ('y',)))

        # 'PART-WHOLE':['FAC', 'GPE'], ['FAC', 'LOC'], ['FAC', 'FAC']
        ifL(PART_WHOLE, ('x', 'y'), ifL(FAC, ('x',), orL(GPE, LOC, FAC), ('y',)))

        # 'PART-WHOLE':['GPE', 'GPE'], ['GPE', 'LOC'], ['GPE', 'FAC'], ['GPE' , 'ORG']
        ifL(PART_WHOLE, ('x', 'y'), ifL(GPE, ('x',), orL(GPE, LOC, FAC, ORG), ('y',)))

        # 'PART-WHOLE':['ORG', 'GPE'], ['ORG' , 'ORG']
        ifL(PART_WHOLE, ('x', 'y'), ifL(ORG, ('x',), orL(GPE, ORG), ('y',)))

        # 'PART-WHOLE':['WEA', 'VEH'], ['WEA' , 'WEA']
        ifL(PART_WHOLE, ('x', 'y'), ifL(WEA, ('x',), orL(WEA, VEH), ('y',)))

        # 'PART-WHOLE':['VEH', 'VEH']
        ifL(PART_WHOLE, ('x', 'y'), ifL(VEH, ('x',), orL(VEH, WEA), ('y',)))


        # # 'ART':['PER', 'VEH'], ['PER', 'WEA'], ['PER', 'FAC']
        # ifL(ART, ('x', 'y'), ifL(PER, ('x',), orL(VEH, WEA, FAC), ('y',)))
        #
        # # 'ART':['ORG', 'VEH'], ['ORG', 'WEA'], ['ORG', 'FAC']
        # ifL(ART, ('x', 'y'), ifL(ORG, ('x',), orL(VEH, WEA, FAC), ('y',)))
        #
        # # 'ART':['GPE', 'VEH'], ['GPE', 'WEA'], ['GPE', 'FAC']
        # ifL(ART, ('x', 'y'), ifL(GPE, ('x',), orL(VEH, WEA, FAC), ('y',)))
        #
        # # 'ART':['VEH', 'ORG']
        # ifL(ART, ('x', 'y'), ifL(VEH, ('x',), orL(ORG, PER, GPE), ('y',)))
        #
        # # 'ART':['WEA', 'GPE']
        # ifL(ART, ('x', 'y'), ifL(WEA, ('x',), orL(GPE, ORG, PER), ('y',)))
        #
        # ifL(ART, ('x', 'y'), ifL(FAC, ('x',), orL(GPE, ORG, PER), ('y',)))


        # # PER_SOC
        # orL(
        #     andL(
        #         ifL(PER_SOC, ("x", "y",), PER("x", )),
        #         ifL(PER_SOC, ("x", "y",), PER("y", ))
        #     ),
        #     andL(
        #         ifL(PER_SOC, ("x", "y",), PER("x", )),
        #         ifL(PER_SOC, ("x", "y",), ORG("y",))
        #     ),
        #     andL(
        #         ifL(PER_SOC, ("x", "y",), ORG("x", )),
        #         ifL(PER_SOC, ("x", "y",), PER("y",))
        #     )
        # )
        #
        # #GEN_AFF
        # orL(
        #     andL(
        #         ifL(GEN_AFF, ("x", "y",), PER("x", )),
        #         ifL(GEN_AFF, ("x", "y",), GPE("y", ))
        #     ),
        #     andL(
        #         ifL(GEN_AFF, ("x", "y",), PER("x", )),
        #         ifL(GEN_AFF, ("x", "y",), PER("y",))
        #     ),
        #     andL(
        #         ifL(GEN_AFF, ("x", "y",), ORG("x", )),
        #         ifL(GEN_AFF, ("x", "y",), LOC("y",))
        #     ),
        #     andL(
        #         ifL(GEN_AFF, ("x", "y",), ORG("x", )),
        #         ifL(GEN_AFF, ("x", "y",), GPE, ("y",))
        #     ),
        #     andL(
        #         ifL(GEN_AFF, ("x", "y",), PER("x", )),
        #         ifL(GEN_AFF, ("x", "y",), LOC("y",))
        #     ),
        #     andL(
        #         ifL(GEN_AFF, ("x", "y",), PER("x", )),
        #         ifL(GEN_AFF, ("x", "y",), ORG("y",))
        #     ),
        #     andL(
        #         ifL(GEN_AFF, ("x", "y",), ORG("x", )),
        #         ifL(GEN_AFF, ("x", "y",), ORG("y",))
        #     ),
        #     andL(
        #         ifL(GEN_AFF, ("x", "y",), GPE("x", )),
        #         ifL(GEN_AFF, ("x", "y",), PER("y",))
        #     )
        # )
        #
        # # ORG_AFF
        # orL(
        #     andL(
        #         ifL(ORG_AFF, ("x", "y",), PER("x", )),
        #         ifL(ORG_AFF, ("x", "y",), GPE("y", ))
        #     ),
        #     andL(
        #         ifL(ORG_AFF, ("x", "y",), PER("x", )),
        #         ifL(ORG_AFF, ("x", "y",), ORG("y",))
        #     ),
        #     andL(
        #         ifL(ORG_AFF, ("x", "y",), GPE("x", )),
        #         ifL(ORG_AFF, ("x", "y",), ORG("y",))
        #     ),
        #     andL(
        #         ifL(ORG_AFF, ("x", "y",), ORG("x", )),
        #         ifL(ORG_AFF, ("x", "y",), GPE("y",))
        #     ),
        #     andL(
        #         ifL(ORG_AFF, ("x", "y",), ORG("x", )),
        #         ifL(ORG_AFF, ("x", "y",), PER("y",))
        #     ),
        #     andL(
        #         ifL(ORG_AFF, ("x", "y",), ORG("x", )),
        #         ifL(ORG_AFF, ("x", "y",), ORG("y",))
        #     ),
        #     andL(
        #         ifL(ORG_AFF, ("x", "y",), PER("x", )),
        #         ifL(ORG_AFF, ("x", "y",), PER("y",))
        #     ),
        #     andL(
        #         ifL(ORG_AFF, ("x", "y",), GPE("x", )),
        #         ifL(ORG_AFF, ("x", "y",), PER("y", ))
        #     )
        # )
        #
        # # PHYS
        # orL(
        #     andL(
        #         ifL(PHYS, ("x", "y",), PER("x", )),
        #         ifL(PHYS, ("x", "y",), LOC("y", ))
        #     ),
        #     andL(
        #         ifL(PHYS, ("x", "y",), FAC("x", )),
        #         ifL(PHYS, ("x", "y",), LOC("y",))
        #     ),
        #     andL(
        #         ifL(PHYS, ("x", "y",), PER("x", )),
        #         ifL(PHYS, ("x", "y",), FAC("y",))
        #     ),
        #     andL(
        #         ifL(PHYS, ("x", "y",), PER("x", )),
        #         ifL(PHYS, ("x", "y",), GPE("y",))
        #     ),
        #     andL(
        #         ifL(PHYS, ("x", "y",), LOC("x", )),
        #         ifL(PHYS, ("x", "y",), LOC("y",))
        #     ),
        #     andL(
        #         ifL(PHYS, ("x", "y",), GPE("x", )),
        #         ifL(PHYS, ("x", "y",), GPE("y",))
        #     ),
        #     andL(
        #         ifL(PHYS, ("x", "y",), FAC("x", )),
        #         ifL(PHYS, ("x", "y",), GPE("y",))
        #     ),
        #     andL(
        #         ifL(PHYS, ("x", "y",), LOC("x", )),
        #         ifL(PHYS, ("x", "y",), GPE("y", ))
        #     ),
        #     andL(
        #         ifL(PHYS, ("x", "y",), FAC("x", )),
        #         ifL(PHYS, ("x", "y",), FAC("y", ))
        #     ),
        #     andL(
        #         ifL(PHYS, ("x", "y",), ORG("x", )),
        #         ifL(PHYS, ("x", "y",), GPE("y", ))
        #     ),
        #     andL(
        #         ifL(PHYS, ("x", "y",), GPE("x", )),
        #         ifL(PHYS, ("x", "y",), LOC("y", ))
        #     ),
        #     andL(
        #         ifL(PHYS, ("x", "y",), GPE("x", )),
        #         ifL(PHYS, ("x", "y",), PER("y", ))
        #     ),
        #     andL(
        #         ifL(PHYS, ("x", "y",), LOC("x", )),
        #         ifL(PHYS, ("x", "y",), FAC("y", ))
        #     ),
        #     andL(
        #         ifL(PHYS, ("x", "y",), GPE("x", )),
        #         ifL(PHYS, ("x", "y",), FAC("y", ))
        #     ),
        #     andL(
        #         ifL(PHYS, ("x", "y",), LOC("x", )),
        #         ifL(PHYS, ("x", "y",), PER("y", ))
        #     ),
        #     andL(
        #         ifL(PHYS, ("x", "y",), ORG("x", )),
        #         ifL(PHYS, ("x", "y",), FAC("y", ))
        #     )
        # )
        #
        # # ART
        # orL(
        #     andL(
        #         ifL(ART, ("x", "y",), PER("x", )),
        #         ifL(ART, ("x", "y",), VEH("y", ))
        #     ),
        #     andL(
        #         ifL(ART, ("x", "y",), ORG("x", )),
        #         ifL(ART, ("x", "y",), VEH("y",))
        #     ),
        #     andL(
        #         ifL(ART, ("x", "y",), GPE("x", )),
        #         ifL(ART, ("x", "y",), VEH("y",))
        #     ),
        #     andL(
        #         ifL(ART, ("x", "y",), GPE("x", )),
        #         ifL(ART, ("x", "y",), WEA("y",))
        #     ),
        #     andL(
        #         ifL(ART, ("x", "y",), PER("x", )),
        #         ifL(ART, ("x", "y",), WEA("y",))
        #     ),
        #     andL(
        #         ifL(ART, ("x", "y",), ORG("x", )),
        #         ifL(ART, ("x", "y",), FAC("y",))
        #     ),
        #     andL(
        #         ifL(ART, ("x", "y",), PER("x", )),
        #         ifL(ART, ("x", "y",), FAC("y",))
        #     ),
        #     andL(
        #         ifL(ART, ("x", "y",), ORG("x", )),
        #         ifL(ART, ("x", "y",), WEA("y", ))
        #     ),
        #     andL(
        #         ifL(ART, ("x", "y",), GPE("x", )),
        #         ifL(ART, ("x", "y",), FAC("y", ))
        #     ),
        #     andL(
        #         ifL(ART, ("x", "y",), WEA("x", )),
        #         ifL(ART, ("x", "y",), GPE("y", ))
        #     ),
        #     andL(
        #         ifL(ART, ("x", "y",), VEH("x", )),
        #         ifL(ART, ("x", "y",), ORG("y", ))
        #     )
        # )
        #
        # # PART_WHOLE
        # orL(
        #     andL(
        #         ifL(PART_WHOLE, ("x", "y",), LOC("x", )),
        #         ifL(PART_WHOLE, ("x", "y",), GPE("y", ))
        #     ),
        #     andL(
        #         ifL(PART_WHOLE, ("x", "y",), FAC("x", )),
        #         ifL(PART_WHOLE, ("x", "y",), GPE("y", ))
        #     ),
        #     andL(
        #         ifL(PART_WHOLE, ("x", "y",), GPE("x", )),
        #         ifL(PART_WHOLE, ("x", "y",), GPE("y", ))
        #     ),
        #     andL(
        #         ifL(PART_WHOLE, ("x", "y",), ORG("x", )),
        #         ifL(PART_WHOLE, ("x", "y",), ORG("y", ))
        #     ),
        #     andL(
        #         ifL(PART_WHOLE, ("x", "y",), FAC("x", )),
        #         ifL(PART_WHOLE, ("x", "y",), FAC("y", ))
        #     ),
        #     andL(
        #         ifL(PART_WHOLE, ("x", "y",), GPE("x", )),
        #         ifL(PART_WHOLE, ("x", "y",), LOC("y", ))
        #     ),
        #     andL(
        #         ifL(PART_WHOLE, ("x", "y",), LOC("x", )),
        #         ifL(PART_WHOLE, ("x", "y",), LOC("y", ))
        #     ),
        #     andL(
        #         ifL(PART_WHOLE, ("x", "y",), FAC("x", )),
        #         ifL(PART_WHOLE, ("x", "y",), LOC("y", ))
        #     ),
        #     andL(
        #         ifL(PART_WHOLE, ("x", "y",), GPE("x", )),
        #         ifL(PART_WHOLE, ("x", "y",), ORG("y", ))
        #     ),
        #     andL(
        #         ifL(PART_WHOLE, ("x", "y",), GPE("x", )),
        #         ifL(PART_WHOLE, ("x", "y",), FAC("y", ))
        #     ),
        #     andL(
        #         ifL(PART_WHOLE, ("x", "y",), WEA("x", )),
        #         ifL(PART_WHOLE, ("x", "y",), VEH("y", ))
        #     ),
        #     andL(
        #         ifL(PART_WHOLE, ("x", "y",), VEH("x", )),
        #         ifL(PART_WHOLE, ("x", "y",), VEH("y", ))
        #     ),
        #     andL(
        #         ifL(PART_WHOLE, ("x", "y",), PER("x", )),
        #         ifL(PART_WHOLE, ("x", "y",), ORG("y", ))
        #     ),
        #     andL(
        #         ifL(PART_WHOLE, ("x", "y",), WEA("x", )),
        #         ifL(PART_WHOLE, ("x", "y",), WEA("y", ))
        #     ),
        #     andL(
        #         ifL(PART_WHOLE, ("x", "y",), LOC("x", )),
        #         ifL(PART_WHOLE, ("x", "y",), FAC("y", ))
        #     )
        # )