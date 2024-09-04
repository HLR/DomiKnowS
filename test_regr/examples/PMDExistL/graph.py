def get_graph(args, at_least_L=False, at_most_L=False):
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

        expected_zero = b_answer.__getattr__("zero")
        expected_one = b_answer.__getattr__("one")

        expected_value = expected_one
        #existsL(expected_zero)

        #if args.test_train:
        #    expected_value = expected_zero


        atLeastAL(expected_value,2)
        #elif at_most_L:
        #    ifL(a("x"),
        #        atMostAL(expected_value(path=('x', a_contain_b)),2))
        #else:
        #    ifL(a("x"),
        #        existsL(expected_value(path=('x', a_contain_b))))

        # print("no constraint.")
    return graph, a, b, a_contain_b, b_answer
