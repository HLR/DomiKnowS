def get_graph():
    from domiknows.graph import Graph, Concept, Relation
    from domiknows.graph.logicalConstrain import orL, existsL, ifL, notL, andL, atMostAL, atLeastAL
    from domiknows.graph import EnumConcept
    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph('global_PMD') as graph:
        a = Concept(name='a')
        b = Concept(name='b')
        a_contain_b,  = a.contains(b)

        b_answer = b(name="answer_b", ConceptClass=EnumConcept, values=["zero", "one"])

        expected_one = b_answer.__getattr__("one")

        ifL(a("x"),
            existsL(expected_one(path=('x', a_contain_b))))

        # print("no constraint.")
    return graph, a, b, a_contain_b, b_answer
