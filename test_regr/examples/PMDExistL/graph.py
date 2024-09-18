def get_graph(args):
    from domiknows.graph import Graph, Concept, Relation
    from domiknows.graph.logicalConstrain import orL, existsL, ifL, notL, andL, atMostL, atLeastL
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
        # existsL(expected_zero)

        # if args.test_train:
        #    expected_value = expected_zero

        # atLeastL(expected_zero, 1)

        # Test multiclass
        # existsL(expected_zero)
        # ifL(existsL(expected_zero), atMostL(expected_zero, 3))
        # atLeastL(expected_zero, 5)
        atMostL(expected_one, 4)
        # atLeastL(expected_one, 3)

        # atMostL(expected_one, 4)






        # if args.atLeastL:
        #     atLeastL(expected_value, args.expected_count)
        # elif args.atMostL:
        #     atMostL(expected_value, args.expected_count)
        # else:
        #     existsL(expected_value, args.expected_count)
        # elif at_most_L:
        #    ifL(a("x"),
        #        atMostAL(expected_value(path=('x', a_contain_b)),2)) => No constraint in the root
        # else:
        #    ifL(a("x"),
        #        existsL(expected_value(path=('x', a_contain_b))))

        # print("no constraint.")
    return graph, a, b, a_contain_b, b_answer



# Error Case
#         atLeastL(expected_zero, 1)
#         atLeastL(expected_one, 1)


# But 2 atMostL is fine, why?

# Sampling Loss is not working -> all the closs is empty from atlestL