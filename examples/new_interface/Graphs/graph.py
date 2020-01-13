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

        list_of_concepts = [GPE, FAC, LOC, VEH, WEA, PER, ORG]

        for item in list_of_concepts:
            for item1 in list_of_concepts:
                if item != item1:
                    nandL(item, item1)

        # PER_SOC
        orL(
            andL(
                ifL(PER_SOC, ("x", "y",), PER("x", )),
                ifL(PER_SOC, ("x", "y",), PER("y", ))
            ),
            andL(
                ifL(PER_SOC, ("x", "y",), PER("x", )),
                ifL(PER_SOC, ("x", "y",), ORG("y",))
            ),
            andL(
                ifL(PER_SOC, ("x", "y",), ORG("x", )),
                ifL(PER_SOC, ("x", "y",), PER("y",))
            )
        )

        #GEN_AFF
        orL(
            andL(
                ifL(GEN_AFF, ("x", "y",), PER("x", )),
                ifL(GEN_AFF, ("x", "y",), GPE("y", ))
            ),
            andL(
                ifL(GEN_AFF, ("x", "y",), PER("x", )),
                ifL(GEN_AFF, ("x", "y",), PER("y",))
            ),
            andL(
                ifL(GEN_AFF, ("x", "y",), ORG("x", )),
                ifL(GEN_AFF, ("x", "y",), LOC("y",))
            ),
            andL(
                ifL(GEN_AFF, ("x", "y",), ORG("x", )),
                ifL(GEN_AFF, ("x", "y",), GPE, ("y",))
            ),
            andL(
                ifL(GEN_AFF, ("x", "y",), PER("x", )),
                ifL(GEN_AFF, ("x", "y",), LOC("y",))
            ),
            andL(
                ifL(GEN_AFF, ("x", "y",), PER("x", )),
                ifL(GEN_AFF, ("x", "y",), ORG("y",))
            ),
            andL(
                ifL(GEN_AFF, ("x", "y",), ORG("x", )),
                ifL(GEN_AFF, ("x", "y",), ORG("y",))
            ),
            andL(
                ifL(GEN_AFF, ("x", "y",), GPE("x", )),
                ifL(GEN_AFF, ("x", "y",), PER("y",))
            )
        )

        # ORG_AFF
        orL(
            andL(
                ifL(ORG_AFF, ("x", "y",), PER("x", )),
                ifL(ORG_AFF, ("x", "y",), GPE("y", ))
            ),
            andL(
                ifL(ORG_AFF, ("x", "y",), PER("x", )),
                ifL(ORG_AFF, ("x", "y",), ORG("y",))
            ),
            andL(
                ifL(ORG_AFF, ("x", "y",), GPE("x", )),
                ifL(ORG_AFF, ("x", "y",), ORG("y",))
            ),
            andL(
                ifL(ORG_AFF, ("x", "y",), ORG("x", )),
                ifL(ORG_AFF, ("x", "y",), GPE("y",))
            ),
            andL(
                ifL(ORG_AFF, ("x", "y",), ORG("x", )),
                ifL(ORG_AFF, ("x", "y",), PER("y",))
            ),
            andL(
                ifL(ORG_AFF, ("x", "y",), ORG("x", )),
                ifL(ORG_AFF, ("x", "y",), ORG("y",))
            ),
            andL(
                ifL(ORG_AFF, ("x", "y",), PER("x", )),
                ifL(ORG_AFF, ("x", "y",), PER("y",))
            ),
            andL(
                ifL(ORG_AFF, ("x", "y",), GPE("x", )),
                ifL(ORG_AFF, ("x", "y",), PER("y", ))
            )
        )

        # PHYS
        orL(
            andL(
                ifL(PHYS, ("x", "y",), PER("x", )),
                ifL(PHYS, ("x", "y",), LOC("y", ))
            ),
            andL(
                ifL(PHYS, ("x", "y",), FAC("x", )),
                ifL(PHYS, ("x", "y",), LOC("y",))
            ),
            andL(
                ifL(PHYS, ("x", "y",), PER("x", )),
                ifL(PHYS, ("x", "y",), FAC("y",))
            ),
            andL(
                ifL(PHYS, ("x", "y",), PER("x", )),
                ifL(PHYS, ("x", "y",), GPE("y",))
            ),
            andL(
                ifL(PHYS, ("x", "y",), LOC("x", )),
                ifL(PHYS, ("x", "y",), LOC("y",))
            ),
            andL(
                ifL(PHYS, ("x", "y",), GPE("x", )),
                ifL(PHYS, ("x", "y",), GPE("y",))
            ),
            andL(
                ifL(PHYS, ("x", "y",), FAC("x", )),
                ifL(PHYS, ("x", "y",), GPE("y",))
            ),
            andL(
                ifL(PHYS, ("x", "y",), LOC("x", )),
                ifL(PHYS, ("x", "y",), GPE("y", ))
            ),
            andL(
                ifL(PHYS, ("x", "y",), FAC("x", )),
                ifL(PHYS, ("x", "y",), FAC("y", ))
            ),
            andL(
                ifL(PHYS, ("x", "y",), ORG("x", )),
                ifL(PHYS, ("x", "y",), GPE("y", ))
            ),
            andL(
                ifL(PHYS, ("x", "y",), GPE("x", )),
                ifL(PHYS, ("x", "y",), LOC("y", ))
            ),
            andL(
                ifL(PHYS, ("x", "y",), GPE("x", )),
                ifL(PHYS, ("x", "y",), PER("y", ))
            ),
            andL(
                ifL(PHYS, ("x", "y",), LOC("x", )),
                ifL(PHYS, ("x", "y",), FAC("y", ))
            ),
            andL(
                ifL(PHYS, ("x", "y",), GPE("x", )),
                ifL(PHYS, ("x", "y",), FAC("y", ))
            ),
            andL(
                ifL(PHYS, ("x", "y",), LOC("x", )),
                ifL(PHYS, ("x", "y",), PER("y", ))
            ),
            andL(
                ifL(PHYS, ("x", "y",), ORG("x", )),
                ifL(PHYS, ("x", "y",), FAC("y", ))
            )
        )

        # ART
        orL(
            andL(
                ifL(ART, ("x", "y",), PER("x", )),
                ifL(ART, ("x", "y",), VEH("y", ))
            ),
            andL(
                ifL(ART, ("x", "y",), ORG("x", )),
                ifL(ART, ("x", "y",), VEH("y",))
            ),
            andL(
                ifL(ART, ("x", "y",), GPE("x", )),
                ifL(ART, ("x", "y",), VEH("y",))
            ),
            andL(
                ifL(ART, ("x", "y",), GPE("x", )),
                ifL(ART, ("x", "y",), WEA("y",))
            ),
            andL(
                ifL(ART, ("x", "y",), PER("x", )),
                ifL(ART, ("x", "y",), WEA("y",))
            ),
            andL(
                ifL(ART, ("x", "y",), ORG("x", )),
                ifL(ART, ("x", "y",), FAC("y",))
            ),
            andL(
                ifL(ART, ("x", "y",), PER("x", )),
                ifL(ART, ("x", "y",), FAC("y",))
            ),
            andL(
                ifL(ART, ("x", "y",), ORG("x", )),
                ifL(ART, ("x", "y",), WEA("y", ))
            ),
            andL(
                ifL(ART, ("x", "y",), GPE("x", )),
                ifL(ART, ("x", "y",), FAC("y", ))
            ),
            andL(
                ifL(ART, ("x", "y",), WEA("x", )),
                ifL(ART, ("x", "y",), GPE("y", ))
            ),
            andL(
                ifL(ART, ("x", "y",), VEH("x", )),
                ifL(ART, ("x", "y",), ORG("y", ))
            )
        )

        # PART_WHOLE
        orL(
            andL(
                ifL(PART_WHOLE, ("x", "y",), LOC("x", )),
                ifL(PART_WHOLE, ("x", "y",), GPE("y", ))
            ),
            andL(
                ifL(PART_WHOLE, ("x", "y",), FAC("x", )),
                ifL(PART_WHOLE, ("x", "y",), GPE("y", ))
            ),
            andL(
                ifL(PART_WHOLE, ("x", "y",), GPE("x", )),
                ifL(PART_WHOLE, ("x", "y",), GPE("y", ))
            ),
            andL(
                ifL(PART_WHOLE, ("x", "y",), ORG("x", )),
                ifL(PART_WHOLE, ("x", "y",), ORG("y", ))
            ),
            andL(
                ifL(PART_WHOLE, ("x", "y",), FAC("x", )),
                ifL(PART_WHOLE, ("x", "y",), FAC("y", ))
            ),
            andL(
                ifL(PART_WHOLE, ("x", "y",), GPE("x", )),
                ifL(PART_WHOLE, ("x", "y",), LOC("y", ))
            ),
            andL(
                ifL(PART_WHOLE, ("x", "y",), LOC("x", )),
                ifL(PART_WHOLE, ("x", "y",), LOC("y", ))
            ),
            andL(
                ifL(PART_WHOLE, ("x", "y",), FAC("x", )),
                ifL(PART_WHOLE, ("x", "y",), LOC("y", ))
            ),
            andL(
                ifL(PART_WHOLE, ("x", "y",), GPE("x", )),
                ifL(PART_WHOLE, ("x", "y",), ORG("y", ))
            ),
            andL(
                ifL(PART_WHOLE, ("x", "y",), GPE("x", )),
                ifL(PART_WHOLE, ("x", "y",), FAC("y", ))
            ),
            andL(
                ifL(PART_WHOLE, ("x", "y",), WEA("x", )),
                ifL(PART_WHOLE, ("x", "y",), VEH("y", ))
            ),
            andL(
                ifL(PART_WHOLE, ("x", "y",), VEH("x", )),
                ifL(PART_WHOLE, ("x", "y",), VEH("y", ))
            ),
            andL(
                ifL(PART_WHOLE, ("x", "y",), PER("x", )),
                ifL(PART_WHOLE, ("x", "y",), ORG("y", ))
            ),
            andL(
                ifL(PART_WHOLE, ("x", "y",), WEA("x", )),
                ifL(PART_WHOLE, ("x", "y",), WEA("y", ))
            ),
            andL(
                ifL(PART_WHOLE, ("x", "y",), LOC("x", )),
                ifL(PART_WHOLE, ("x", "y",), FAC("y", ))
            )
        )