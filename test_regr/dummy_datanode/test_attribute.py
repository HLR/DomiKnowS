from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import nandL, notL, ifL
from domiknows.graph.dataNodeDummy import createDummyDataNode, satisfactionReportOfConstraints

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('test_graph') as graph:
    image = Concept(name='image')

    a = image(name='a')
    b = image(name='b')
    c = image(name='c')
    d = image(name='d')
    e = image(name='e')

    ifL(a,notL(b))
    ifL(a,notL(c))
    nandL(d,e)

ac_, t_ = 0, 0

datanode=createDummyDataNode(graph)
print(datanode.getAttribute(a))
datanode.inferILPResults()
verifyResult = datanode.verifyResultsLC()
verifyResultILP = datanode.verifyResultsLC()

report = satisfactionReportOfConstraints(datanode)