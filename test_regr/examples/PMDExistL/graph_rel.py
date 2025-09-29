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
        is_cond1 = obj(name="is_cond1")
        is_cond2 = obj(name="is_cond2")
        is_cond3 = obj(name="is_cond3")
        is_cond4 = obj(name="is_cond4")

        relation_obj1_obj2 = Concept('relation_obj1_obj2')
        (obj1, obj2) = relation_obj1_obj2.has_a(arg1=obj, arg2=obj)

        # Making this to 4 relations
        is_relation1 = relation_obj1_obj2(name="is_relation1")
        is_relation2 = relation_obj1_obj2(name="is_relation2")
        is_relation3 = relation_obj1_obj2(name="is_relation3")
        is_relation4 = relation_obj1_obj2(name="is_relation4")

    return (graph, scene, obj, scene_contain_obj, relation_obj1_obj2, obj1, obj2,
            is_cond1, is_cond2, is_cond3, is_cond4,
            is_relation1, is_relation2, is_relation3, is_relation4)

