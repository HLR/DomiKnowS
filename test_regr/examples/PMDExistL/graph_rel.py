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

        # obj = list of M numbers

        # Making this to 4 conditions
        is_cond1 = obj(name="is_cond1") # Sum of number [-0.5, 0.1, 0.5] > 0
        is_cond2 = obj(name="is_cond2") # Absolute of (sum of number) > 0.2
        is_cond3 = obj(name="is_cond3") # Sum of number [-0.5, 0.1, 0.5] > 0
        is_cond4 = obj(name="is_cond4") # Absolute of (sum of number) > 0.2

        relation = Concept('relation_obj1_obj2')
        (obj1, obj2) = relation.has_a(arg1=obj, arg2=obj)

        # Making this to 4 relations
        is_relation1 = relation(name="is_relation1") # sum of x more than sum of y
        is_relation2 = relation(name="is_relation2")  # sum of x more than sum of y
        is_relation3 = relation(name="is_relation3")  # sum of x more than sum of y
        is_relation4 = relation(name="is_relation4")  # sum of x more than sum of y

    return (graph, scene, obj, scene_contain_obj, relation, obj1, obj2,
            is_cond1, is_cond2, is_cond3, is_cond4,
            is_relation1, is_relation2, is_relation3, is_relation4)

