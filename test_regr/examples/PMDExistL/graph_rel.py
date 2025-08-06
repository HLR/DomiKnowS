def get_graph(args):
    from domiknows.graph import Graph, Concept, Relation
    from domiknows.graph.logicalConstrain import orL, existsL, ifL, notL, andL, atMostL, atLeastL, exactL
    from domiknows.graph import EnumConcept
    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph('global_PMD') as graph:
        scene = Concept(name="scene")
        obj = Concept(name="obj")
        scene_contain_obj,  = scene.contains(obj)

        # obj = list of 6 numbers
        is_cond1 = obj(name="is_cond1") # Sum of number [-0.5, 0.1, 0.5] > 0
        is_cond2 = obj(name="is_cond2") # Absolute of (sum of number) > 0.2

        relation = Concept('relation_obj1_obj2')
        (obj1, obj2) = relation.has_a(arg1=obj, arg2=obj)

        is_relation = relation(name="is_relation") # sum of x more than sum of y

        if args.constraint_2_existL:
            learning_condition = existsL(is_cond1('x'),
                                         is_relation('rel1', path=('x', obj1.reversed)),
                                         existsL(is_cond2('y', path=('rel1', obj2)))
                                         )
        else:
            learning_condition = existsL(is_cond1('x'),
                                         is_relation('rel1', path=('x', obj1.reversed)),
                                         is_cond2('y', path=('rel1', obj2)))

    return graph, scene, obj, scene_contain_obj, is_cond1, is_cond2, relation, obj1, obj2, is_relation, learning_condition

