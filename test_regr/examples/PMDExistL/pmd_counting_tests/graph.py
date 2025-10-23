def get_graph(args):
    from domiknows.graph import Graph, Concept, Relation
    from domiknows.graph.logicalConstrain import orL, existsL, ifL, notL, andL, atMostAL, atLeastAL, exactAL
    from domiknows.graph.logicalConstrain import atLeastL, atMostL, exactL
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

        expected_value = expected_zero if args.expected_value == 0 else expected_one

        if args.atLeastL and args.atMostL:
            atLeastL(expected_value, args.expected_atLeastL)
            atMostL(expected_value, args.expected_atMostL)
        elif args.atMostL:
            atMostL(expected_value, args.expected_atMostL)
        elif args.atLeastL:
            atLeastL(expected_value, args.expected_atLeastL)
        else:
            exactL(expected_value, args.expected_atLeastL)
            
        for lc in graph.logicalConstrains():
            print(f"Defined logical constraint: {lc.strEs()}")

    return graph, a, b, a_contain_b, b_answer