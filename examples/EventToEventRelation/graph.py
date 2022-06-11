import copy

from regr.graph import Graph, Concept, Relation, EnumConcept
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, V, nandL, exactL

Graph.clear()
Concept.clear()
Relation.clear()
with Graph('event_to_event') as graph:
    # Group of sentence
    paragraph = Concept(name="paragraph")
    event_relation = Concept(name="event_relation")
    paragraph_contain, = paragraph.contains(event_relation)

    # each relation could be one the following four concepts:
    # sub-event relation consists of four relations, parent child, child parent, coref, and noref
    relation_classes = event_relation(name="relation_classes", ConceptClass=EnumConcept,
                                      values=["before", "after",
                                              "EQUAL", "VAGUE"])

    # Only one of the labels to be true

    # ifL(event_relation, exactL( #     relation_classes.parent_child,
    # relation_classes.child_parent, relation_classes.coref, relation_classes.norel, #     relation_classes.before,
    # relation_classes.after, relation_classes.EQUAL, relation_classes.VAGUE))
    ifL(event_relation,
        exactL(relation_classes.before, relation_classes.after, relation_classes.EQUAL, relation_classes.VAGUE))

    # ifL(event_relation, atMostL(
    #     relation_classes.parent_child, relation_classes.child_parent, relation_classes.coref, relation_classes.norel,
    #     relation_classes.before, relation_classes.after, relation_classes.EQUAL, relation_classes.VAGUE))

    # Symmetric Constrain
    symmetric = Concept("symmetric")
    s_event1, s_event2 = symmetric.has_a(arg1=event_relation, arg2=event_relation)
    # # TODO: Symmetric of subevent relation
    #
    # # Before(e1, e2) <=> After(e2, e1)
    # ifL(relation_classes.before('x'), relation_classes.after(path=('x', symmetric, s_event2)))
    # ifL(relation_classes.after('x'), relation_classes.before(path=('x', symmetric, s_event2)))
    #
    # # Equal(e1, e2) <=> Vague(e2, e1)
    # ifL(relation_classes.EQUAL('x'), relation_classes.VAGUE(path=('x', symmetric, s_event2)))
    # ifL(relation_classes.VAGUE('x'), relation_classes.EQUAL(path=('x', symmetric, s_event2)))
    #
    # ifL(relation_classes.child_parent('x'), relation_classes.parent_child(path=('x', symmetric, s_event2)))
    # ifL(relation_classes.parent_child('x'), relation_classes.child_parent(path=('x', symmetric, s_event2)))
    #
    # ifL(relation_classes.COREF('x'), relation_classes.NOREL(path=('x', symmetric, s_event2)))
    # ifL(relation_classes.NOREL('x'), relation_classes.COREF(path=('x', symmetric, s_event2)))
    #
    # # # TODO: Transitive Constrains
    transitive = Concept("transitive")
    t_event1, t_event2, t_event3 = transitive.has_a(arg11=event_relation, arg22=event_relation, arg33=event_relation)
    # ifL(andL(relation_classes.before('x'), relation_classes.parent_child(path=('x', transitive, t_event2))),
    #     orL(
    #         notL(relation_classes.COREF(path=('x', transitive, t_event3))),
    #         relation_classes.before(path=('x', transitive, t_event3)),
    #         notL(relation_classes.child_parent(path=('x', transitive, t_event3)))
    #     ))

    # all_classes = {"after": relation_classes.after, "before": relation_classes.before, "vague": relation_classes.VAGUE,
    #                "equal": relation_classes.EQUAL, "parent_child": relation_classes.parent_child,
    #                "child_parent": relation_classes.child_parent, "coref": relation_classes.COREF,
    #                "norel": relation_classes.NOREL}
    #
    #
    # def create_transitive_relation(relation1, relation2, relation3s):
    #     if not len(relation3s):
    #         return
    #     if len(relation3s) == 1:
    #         relation = all_classes[relation3s[0]](path=('x', transitive, t_event3)) if relation3s[0] in all_classes\
    #             else notL(all_classes[relation3s[0].split(" ")[1]](path=('x', transitive, t_event3)))
    #
    #         ifL(andL(all_classes[relation1]('x'), all_classes[relation2](path=('x', transitive, t_event2))),
    #             relation)
    #     else:
    #         relation3 = []
    #         for relation in relation3s:
    #             if relation in all_classes:
    #                 relation3.append(all_classes[relation](path=('x', transitive, t_event3)))
    #             else:
    #                 relation3.append(notL(all_classes[relation.split(" ")[1]](path=('x', transitive, t_event3))))
    #         # ifL(andL(all_classes[relation1]('x'), all_classes[relation2](path=('x', transitive, t_event2))),
    #         #     orL(*relation3))
    #
    #
    # transitive_table = {relation_class:
    #                         {relation_class_inner: [] for relation_class_inner in all_classes}
    #                     for relation_class in all_classes}
    #
    # # create_transitive_relation(relation_classes.before, relation_classes.before, relation_classes.before
    # case1 = ("before", "not child_parent")
    # transitive_table["before"]["parent_child"] = transitive_table["parent_child"]["before"] = case1
    # ifL(andL(relation_classes.before('x'), relation_classes.parent_child(path=('x', transitive, t_event2))),
    #     notL(relation_classes.COREF(path=('x', transitive, t_event3))))
    # ifL(andL(relation_classes.parent_child('x'), relation_classes.before(path=('x', transitive, t_event2))),
    #     notL(relation_classes.COREF(path=('x', transitive, t_event3))))
    #
    # transitive_table["before"]["coref"] = transitive_table["coref"]["before"] = case1
    # ifL(andL(relation_classes.before('x'), relation_classes.COREF(path=('x', transitive, t_event2))),
    #     notL(relation_classes.COREF(path=('x', transitive, t_event3))))
    # ifL(andL(relation_classes.COREF('x'), relation_classes.before(path=('x', transitive, t_event2))),
    #     notL(relation_classes.COREF(path=('x', transitive, t_event3))))
    #
    # transitive_table["before"]["before"] = case1
    # ifL(andL(relation_classes.before('x'), relation_classes.before(path=('x', transitive, t_event2))),
    #     notL(relation_classes.COREF(path=('x', transitive, t_event3))))
    #
    # transitive_table["before"]["equal"] = transitive_table["equal"]["before"] = case1
    # ifL(andL(relation_classes.before('x'), relation_classes.EQUAL(path=('x', transitive, t_event2))),
    #     notL(relation_classes.COREF(path=('x', transitive, t_event3))))
    # ifL(andL(relation_classes.EQUAL('x'), relation_classes.before(path=('x', transitive, t_event2))),
    #     notL(relation_classes.COREF(path=('x', transitive, t_event3))))
    #
    # transitive_table["parent_child"]["equal"] = case1
    # ifL(andL(relation_classes.parent_child('x'), relation_classes.EQUAL(path=('x', transitive, t_event2))),
    #     notL(relation_classes.COREF(path=('x', transitive, t_event3))))
    #
    #
    # case2 = ("after", "not parent_child", "not coref")
    # transitive_table["after"]["child_parent"] = transitive_table["child_parent"]["after"] = case2
    # ifL(andL(relation_classes.after('x'), relation_classes.parent_child(path=('x', transitive, t_event2))),
    #     notL(relation_classes.COREF(path=('x', transitive, t_event3))))
    # ifL(andL(relation_classes.parent_child('x'), relation_classes.after(path=('x', transitive, t_event2))),
    #     notL(relation_classes.COREF(path=('x', transitive, t_event3))))
    # transitive_table["after"]["coref"] = transitive_table["coref"]["after"] = case2
    # ifL(andL(relation_classes.after('x'), relation_classes.COREF(path=('x', transitive, t_event2))),
    #     notL(relation_classes.COREF(path=('x', transitive, t_event3))))
    # ifL(andL(relation_classes.COREF('x'), relation_classes.after(path=('x', transitive, t_event2))),
    #     notL(relation_classes.COREF(path=('x', transitive, t_event3))))
    # transitive_table["after"]["equal"] = transitive_table["equal"]["after"] = case2
    # ifL(andL(relation_classes.after('x'), relation_classes.EQUAL(path=('x', transitive, t_event2))),
    #     notL(relation_classes.COREF(path=('x', transitive, t_event3))))
    # ifL(andL(relation_classes.EQUAL('x'), relation_classes.after(path=('x', transitive, t_event2))),
    #     notL(relation_classes.COREF(path=('x', transitive, t_event3))))
    #
    #
    # transitive_table["after"]["after"] = case2
    # ifL(andL(relation_classes.after('x'), relation_classes.after(path=('x', transitive, t_event2))),
    #     notL(relation_classes.COREF(path=('x', transitive, t_event3))))
    # transitive_table["child_parent"]["equal"] = case2
    # ifL(andL(relation_classes.child_parent('x'), relation_classes.EQUAL(path=('x', transitive, t_event2))),
    #     notL(relation_classes.COREF(path=('x', transitive, t_event3))))
    #
    # transitive_table["parent_child"]["parent_child"] = ("parent_child", "not after")
    # transitive_table["parent_child"]["coref"] = \
    #     transitive_table["coref"]["parent_child"] = ("parent_child", "not after")
    # transitive_table["parent_child"]["norel"] = \
    #     transitive_table["norel"]["parent_child"] = ("not child_parent", "not coref")
    #
    # transitive_table["child_parent"]["child_parent"] = ("child_parent", "not before")
    # transitive_table["child_parent"]["coref"] = \
    #     transitive_table["coref"]["child_parent"] = ("child_parent", "not before")
    # transitive_table["child_parent"]["norel"] = \
    #     transitive_table["norel"]["child_parent"] = ("not parent_child", "not coref")
    #
    # transitive_table["coref"]["coref"] = ("coref", "equal")
    # transitive_table["coref"]["norel"] = transitive_table["norel"]["coref"] = ("norel",)
    # transitive_table["coref"]["equal"] = transitive_table["equal"]["coref"] = ("equal",)
    # transitive_table["coref"]["vague"] = transitive_table["vague"]["coref"] = ("vague",)
    #
    # transitive_table["before"]["vague"] = transitive_table["vague"]["before"] = ("not after", "not equal")
    # transitive_table["after"]["vague"] = transitive_table["vague"]["after"] = ("not before", "not equal")
    #
    # transitive_table["equal"]["parent_child"] = ("not after",)
    # transitive_table["equal"]["child_parent"] = ("not before",)
    # transitive_table["equal"]["vague"] = transitive_table["vague"]["equal"] = ("not coref", "vague")
    #
    # for relation1, dict_relation1 in transitive_table.items():
    #     for relation2, relation3_list in dict_relation1.items():
    #         create_transitive_relation(relation1, relation2, relation3_list)
