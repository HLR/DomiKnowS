from regr.graph import Graph, Concept, Relation, EnumConcept
from regr.graph.logicalConstrain import nandL, orL, andL, existsL, notL, atLeastL, atMostL, ifL
from regr.graph.relation import disjoint, IsA
from itertools import combinations

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('AnimalAndFlower') as graph:
    image_group = Concept(name='image_group')
    image = Concept(name='image')
    image_group_contains, = image_group.contains(image)

    pieridae = image(name='pieridae')
    coliadinae = pieridae(name='coliadinae')
    pierinae = pieridae(name='pierinae')
    dismorphiinae = pieridae(name='dismorphiinae')
    ifL(coliadinae('x'), pieridae('x'))
    ifL(pierinae('x'), pieridae('x'))
    ifL(dismorphiinae('x'), pieridae('x'))
    ifL(pieridae, orL(coliadinae, pierinae, dismorphiinae))

    lycaenidae = image(name='lycaenidae')
    polyommatinae = lycaenidae(name='polyommatinae')
    aphnaeinae = lycaenidae(name='aphnaeinae')
    lycaeninae = lycaenidae(name='lycaeninae')
    theclinae = lycaenidae(name='theclinae')

    ifL(polyommatinae('x'), lycaenidae('x'))
    ifL(aphnaeinae('x'), lycaenidae('x'))
    ifL(lycaeninae('x'), lycaenidae('x'))
    ifL(theclinae('x'), lycaenidae('x'))
    ifL(lycaenidae, orL(theclinae, aphnaeinae, dismorphiinae,lycaeninae))

    nymphalidae = image(name='nymphalidae')
    apaturinae = nymphalidae(name='apaturinae')
    libytheinae = nymphalidae(name='libytheinae')
    limenitidinae = nymphalidae(name='limenitidinae')
    danainae = nymphalidae(name='danainae')
    heliconiinae = nymphalidae(name='heliconiinae')
    charaxinae = nymphalidae(name='charaxinae')
    satyrinae = nymphalidae(name='satyrinae')
    nymphalinae = nymphalidae(name='nymphalinae')

    ifL(apaturinae('x'), nymphalidae('x'))
    ifL(libytheinae('x'), nymphalidae('x'))
    ifL(limenitidinae('x'), nymphalidae('x'))
    ifL(danainae('x'), nymphalidae('x'))
    ifL(heliconiinae('x'), nymphalidae('x'))
    ifL(charaxinae('x'), nymphalidae('x'))
    ifL(satyrinae('x'), nymphalidae('x'))
    ifL(nymphalidae, orL(apaturinae, libytheinae, limenitidinae, danainae, heliconiinae, charaxinae, satyrinae))

    hesperiidae = image(name='hesperiidae')
    heteropterinae = hesperiidae(name='heteropterinae')
    pyrginae = hesperiidae(name='pyrginae')
    hesperiinae = hesperiidae(name='hesperiinae')
    ifL(heteropterinae('x'), hesperiidae('x'))
    ifL(pyrginae('x'), hesperiidae('x'))
    ifL(hesperiinae('x'), hesperiidae('x'))
    ifL(hesperiidae, orL(heteropterinae, pyrginae, hesperiinae))

    papilionidae = image(name='papilionidae')
    papilioninae = papilionidae(name='papilioninae')
    parnassiinae = papilionidae(name='parnassiinae')
    ifL(papilioninae('x'), papilionidae('x'))
    ifL(parnassiinae('x'), papilionidae('x'))
    ifL(hesperiidae, orL(papilioninae, parnassiinae))

    riodinidae = image(name='riodinidae')
    nemeobiinae = riodinidae(name='nemeobiinae')

    ifL(nemeobiinae('x'), riodinidae('x'))
    ifL(riodinidae, orL(nemeobiinae))


    disjoint(dismorphiinae, pierinae, coliadinae, polyommatinae, theclinae, aphnaeinae, lycaeninae, limenitidinae, apaturinae, danainae, satyrinae, nymphalinae, libytheinae, heliconiinae, charaxinae, heteropterinae, pyrginae, hesperiinae, parnassiinae, papilioninae, nemeobiinae)
    # [dismorphiinae, pierinae, coliadinae, polyommatinae, theclinae, aphnaeinae, lycaeninae, limenitidinae, apaturinae, danainae, satyrinae, nymphalinae, libytheinae, heliconiinae, charaxinae, heteropterinae, pyrginae, hesperiinae, parnassiinae, papilioninae, nemeobiinae]

    for l1, l2 in combinations([pieridae, lycaenidae,nymphalidae,hesperiidae,papilionidae,riodinidae], 2):
        nandL(l1, l2)
