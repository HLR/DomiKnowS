from regr.graph import Graph, Concept, Relation, EnumConcept
from regr.graph.logicalConstrain import nandL, orL, andL, existsL, notL, atLeastL, atMostL, ifL
from regr.graph.relation import disjoint, IsA
from itertools import combinations

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('Butterfly') as graph:
    # data = ['coliadinae', 'dismorphiinae', 'pierinae', 'polyommatinae', 'theclinae', 'lycaeninae', 'aphnaeinae',
    #         'charaxinae', 'limenitidinae', 'libytheinae', 'danainae', 'nymphalinae', 'apaturinae', 'satyrinae',
    #         'heliconiinae', 'pyrginae', 'hesperiinae', 'heteropterinae', 'parnassiinae', 'papilioninae',
    #         'nemeobiinae', 'pieridae', 'lycaenidae', 'nymphalidae', 'hesperiidae', 'papilionidae', 'riodinidae']
    ################################################################################################################################################################

    image_group = Concept(name='image_group')
    image = Concept(name='image')
    image_group_contains, = image_group.contains(image)

    family = image(name="family", ConceptClass=EnumConcept,
                   values=['pieridae', 'lycaenidae', 'nymphalidae', 'hesperiidae', 'papilionidae', "riodinidae"])
    subFamily = image(
        name="subFamily",
        ConceptClass=EnumConcept,
        values=['dismorphiinae', 'pierinae', 'coliadinae', 'polyommatinae', 'theclinae', 'aphnaeinae', 'lycaeninae',
                'limenitidinae', 'apaturinae', 'danainae', 'satyrinae', 'nymphalinae', 'libytheinae', 'heliconiinae',
                'charaxinae', 'heteropterinae', 'pyrginae', 'hesperiinae', 'parnassiinae', 'papilioninae',
                'nemeobiinae']
    )

    for l1, l2 in combinations(subFamily.attributes, 2):
        nandL(l1, l2)
    for l1, l2 in combinations(family.attributes, 2):
        nandL(l1, l2)

    ifL(subFamily.coliadinae, family.pieridae)
    ifL(subFamily.pierinae, family.pieridae)
    ifL(subFamily.dismorphiinae, family.pieridae)
    ifL(family.pieridae, orL(subFamily.coliadinae, subFamily.pierinae, subFamily.dismorphiinae))

    ifL(subFamily.polyommatinae, family.lycaenidae)
    ifL(subFamily.aphnaeinae, family.lycaenidae)
    ifL(subFamily.lycaeninae, family.lycaenidae)
    ifL(subFamily.theclinae, family.lycaenidae)
    ifL(family.lycaenidae,
        orL(subFamily.polyommatinae, subFamily.aphnaeinae, subFamily.lycaeninae, subFamily.theclinae))

    ifL(subFamily.apaturinae, family.nymphalidae)
    ifL(subFamily.libytheinae, family.nymphalidae)
    ifL(subFamily.limenitidinae, family.nymphalidae)
    ifL(subFamily.danainae, family.nymphalidae)
    ifL(subFamily.heliconiinae, family.nymphalidae)
    ifL(subFamily.charaxinae, family.nymphalidae)
    ifL(subFamily.satyrinae, family.nymphalidae)
    ifL(family.nymphalidae,
        orL(subFamily.apaturinae, subFamily.libytheinae, subFamily.limenitidinae, subFamily.danainae,
            subFamily.heliconiinae, subFamily.charaxinae, subFamily.satyrinae))

    ifL(subFamily.heteropterinae, family.hesperiidae)
    ifL(subFamily.pyrginae, family.hesperiidae)
    ifL(subFamily.hesperiinae, family.hesperiidae)
    ifL(family.hesperiidae, orL(subFamily.heteropterinae, subFamily.pyrginae, subFamily.hesperiinae))

    ifL(subFamily.papilioninae, family.papilionidae)
    ifL(subFamily.parnassiinae, family.papilionidae)
    ifL(family.papilionidae, orL(subFamily.papilioninae, subFamily.parnassiinae))

    ifL(subFamily.nemeobiinae, family.riodinidae)
    ifL(family.riodinidae, orL(subFamily.nemeobiinae))
