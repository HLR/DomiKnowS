import copy

from domiknows.graph import Graph, Concept, Relation, EnumConcept
from domiknows.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, V, nandL, exactL

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
                                      values=["before", "after", "EQUAL", "VAGUE"])

    # Symmetric constraint
    symmetric = Concept("symmetric")
    s_event1, s_event2 = symmetric.has_a(arg1=event_relation, arg2=event_relation)


    # Before(e1, e2) <=> After(e2, e1)
    ifL(andL(relation_classes.after('x'), existsL(symmetric('s', path=('x', symmetric)))),
        relation_classes.before(path=('x', symmetric, s_event2)))
    ifL(andL(relation_classes.before('x'), existsL(symmetric('s', path=('x', symmetric)))),
        relation_classes.after(path=('x', symmetric, s_event2)))

    # Equal(e1, e2) <=> Vague(e2, e1)
    ifL(andL(relation_classes.EQUAL('x'), existsL(symmetric('s', path=('x', symmetric)))),
        relation_classes.VAGUE(path=('x', symmetric, s_event2)))
    ifL(andL(relation_classes.VAGUE('x'), existsL(symmetric('s', path=('x', symmetric)))),
        relation_classes.EQUAL(path=('x', symmetric, s_event2)))

    # Child Parent(e1, e2) <=> Child Parent(e2, e1)
    # ifL(andL(relation_classes.child_parent('x'), existsL(symmetric('s', path=('x', symmetric)))),
    #     relation_classes.parent_child(path=('x', symmetric, s_event2)))
    # ifL(andL(relation_classes.parent_child('x'), existsL(symmetric('s', path=('x', symmetric)))),
    #     relation_classes.child_parent(path=('x', symmetric, s_event2)))
    #
    # # Coref(e1, e2) <=> Norel(e2, e1)
    # ifL(andL(relation_classes.COREF('x'), existsL(symmetric('s', path=('x', symmetric)))),
    #     relation_classes.NOREL(path=('x', symmetric, s_event2)))
    # ifL(andL(relation_classes.NOREL('x'), existsL(symmetric('s', path=('x', symmetric)))),
    #     relation_classes.COREF(path=('x', symmetric, s_event2)))


    transitive = Concept("transitive")
    t_event1, t_event2, t_event3 = transitive.has_a(arg11=event_relation, arg22=event_relation, arg33=event_relation)

    # case1 = ("before", "not child_parent", "not coref")
    # transitive_table["before"]["parent_child"] = transitive_table["parent_child"]["before"] = case1
    # transitive_table["before"]["coref"] = transitive_table["coref"]["before"] = case1
    # transitive_table["before"]["before"] = case1
    # transitive_table["before"]["equal"] = transitive_table["equal"]["before"] = case1
    # transitive_table["parent_child"]["equal"] = case1
    # ifL(andL(relation_classes.before('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.parent_child(path=('t', t_event2))),
    #     orL(
    #         notL(relation_classes.COREF(path=('t', t_event3))),
    #         relation_classes.before(path=('t', t_event3)),
    #         notL(relation_classes.child_parent(path=('t', t_event3)))
    #     ))
    # ifL(andL(relation_classes.parent_child('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.before(path=('t', t_event2))),
    #     orL(
    #         notL(relation_classes.COREF(path=('t', t_event3))),
    #         relation_classes.before(path=('t', t_event3)),
    #         notL(relation_classes.child_parent(path=('t', t_event3)))
    #     ))
    #
    # ifL(andL(relation_classes.before('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.COREF(path=('t', t_event2))),
    #     orL(
    #         notL(relation_classes.COREF(path=('t', t_event3))),
    #         relation_classes.before(path=('t', t_event3)),
    #         notL(relation_classes.child_parent(path=('t', t_event3)))
    #     ))
    #
    # ifL(andL(relation_classes.COREF('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.before(path=('t', t_event2))),
    #     orL(
    #         notL(relation_classes.COREF(path=('t', t_event3))),
    #         relation_classes.before(path=('t', t_event3)),
    #         notL(relation_classes.child_parent(path=('t', t_event3)))
    #     ))

    ifL(andL(relation_classes.before('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.before(path=('t', t_event2))),
        relation_classes.before(path=('t', t_event3))
        )

    ifL(andL(relation_classes.before('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.EQUAL(path=('t', t_event2))),
        relation_classes.before(path=('t', t_event3)))

    ifL(andL(relation_classes.EQUAL('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.before(path=('t', t_event2))),
        relation_classes.before(path=('t', t_event3)))

    # ifL(andL(relation_classes.parent_child('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.EQUAL(path=('t', t_event2))),
    #     orL(
    #         notL(relation_classes.COREF(path=('t', t_event3))),
    #         relation_classes.before(path=('t', t_event3)),
    #         notL(relation_classes.child_parent(path=('t', t_event3)))
    #     ))

    #
    # case2 = ("after", "not parent_child", "not coref")
    # transitive_table["after"]["child_parent"] = transitive_table["child_parent"]["after"] = case2

    # ifL(andL(relation_classes.after('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.child_parent(path=('t', t_event2))),
    #     orL(
    #         notL(relation_classes.COREF(path=('t', t_event3))),
    #         relation_classes.after(path=('t', t_event3)),
    #         notL(relation_classes.parent_child(path=('t', t_event3)))
    #     ))
    #
    # ifL(andL(relation_classes.child_parent('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.after(path=('t', t_event2))),
    #     orL(
    #         notL(relation_classes.COREF(path=('t', t_event3))),
    #         relation_classes.after(path=('t', t_event3)),
    #         notL(relation_classes.parent_child(path=('t', t_event3)))
    #     ))
    #
    # ifL(andL(relation_classes.COREF('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.after(path=('t', t_event2))),
    #     orL(
    #         notL(relation_classes.COREF(path=('t', t_event3))),
    #         relation_classes.after(path=('t', t_event3)),
    #         notL(relation_classes.parent_child(path=('t', t_event3)))
    #     ))
    #
    # ifL(andL(relation_classes.after('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.COREF(path=('t', t_event2))),
    #     orL(
    #         notL(relation_classes.COREF(path=('t', t_event3))),
    #         relation_classes.after(path=('t', t_event3)),
    #         notL(relation_classes.parent_child(path=('t', t_event3)))
    #     ))


    # transitive_table["after"]["equal"] = transitive_table["equal"]["after"] = case2
    ifL(andL(relation_classes.EQUAL('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.after(path=('t', t_event2))),
        relation_classes.after(path=('t', t_event3)))

    ifL(andL(relation_classes.after('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.EQUAL(path=('t', t_event2))),
        relation_classes.after(path=('t', t_event3)))
    # transitive_table["after"]["after"] = case2
    ifL(andL(relation_classes.after('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.after(path=('t', t_event2))),
        relation_classes.after(path=('t', t_event3)))
    # transitive_table["child_parent"]["equal"] = case2
    # ifL(andL(relation_classes.child_parent('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.EQUAL(path=('t', t_event2))),
    #     orL(
    #         notL(relation_classes.COREF(path=('t', t_event3))),
    #         relation_classes.after(path=('t', t_event3)),
    #         notL(relation_classes.parent_child(path=('t', t_event3)))
    #     ))
    #
    # transitive_table["parent_child"]["parent_child"] = ("parent_child", "not after")
    # ifL(andL(relation_classes.parent_child('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.parent_child(path=('t', t_event2))),
    #     orL(
    #         notL(relation_classes.after(path=('t', t_event3))),
    #         relation_classes.parent_child(path=('t', t_event3))
    #     ))
    # transitive_table["parent_child"]["coref"] = \
    #     transitive_table["coref"]["parent_child"] = ("parent_child", "not after")
    # ifL(andL(relation_classes.parent_child('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.COREF(path=('t', t_event2))),
    #     orL(
    #         notL(relation_classes.after(path=('t', t_event3))),
    #         relation_classes.parent_child(path=('t', t_event3))
    #     ))
    # ifL(andL(relation_classes.COREF('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.parent_child(path=('t', t_event2))),
    #     orL(
    #         notL(relation_classes.after(path=('t', t_event3))),
    #         relation_classes.parent_child(path=('t', t_event3))
    #     ))
    # transitive_table["parent_child"]["norel"] = \
    #     transitive_table["norel"]["parent_child"] = ("not child_parent", "not coref")
    # ifL(andL(relation_classes.parent_child('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.NOREL(path=('t', t_event2))),
    #     orL(
    #         notL(relation_classes.child_parent(path=('t', t_event3))),
    #         notL(relation_classes.COREF(path=('t', t_event3)))
    #     ))
    # ifL(andL(relation_classes.NOREL('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.parent_child(path=('t', t_event2))),
    #     orL(
    #         notL(relation_classes.child_parent(path=('t', t_event3))),
    #         notL(relation_classes.COREF(path=('t', t_event3)))
    #     ))
    #
    # transitive_table["child_parent"]["child_parent"] = ("child_parent", "not before")
    # ifL(andL(relation_classes.child_parent('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.child_parent(path=('t', t_event2))),
    #     orL(
    #         relation_classes.child_parent(path=('t', t_event3)),
    #         notL(relation_classes.before(path=('t', t_event3)))
    #     ))
    # # transitive_table["child_parent"]["coref"] = \
    # #     transitive_table["coref"]["child_parent"] = ("child_parent", "not before")
    # ifL(andL(relation_classes.child_parent('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.COREF(path=('t', t_event2))),
    #     orL(
    #         relation_classes.child_parent(path=('t', t_event3)),
    #         notL(relation_classes.before(path=('t', t_event3)))
    #     ))
    # ifL(andL(relation_classes.COREF('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.child_parent(path=('t', t_event2))),
    #     orL(
    #         relation_classes.child_parent(path=('t', t_event3)),
    #         notL(relation_classes.before(path=('t', t_event3)))
    #     ))
    # transitive_table["child_parent"]["norel"] = \
    #     transitive_table["norel"]["child_parent"] = ("not parent_child", "not coref")
    # ifL(andL(relation_classes.child_parent('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.NOREL(path=('t', t_event2))),
    #     orL(
    #         notL(relation_classes.parent_child(path=('t', t_event3))),
    #         notL(relation_classes.COREF(path=('t', t_event3)))
    #     ))
    # ifL(andL(relation_classes.NOREL('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.child_parent(path=('t', t_event2))),
    #     orL(
    #         notL(relation_classes.parent_child(path=('t', t_event3))),
    #         notL(relation_classes.COREF(path=('t', t_event3)))
    #     ))
    #
    # transitive_table["coref"]["coref"] = ("coref", "equal")
    # ifL(andL(relation_classes.COREF('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.COREF(path=('t', t_event2))),
    #     orL(
    #         relation_classes.COREF(path=('t', t_event3)),
    #         relation_classes.EQUAL(path=('t', t_event3))
    #     ))
    # # transitive_table["coref"]["norel"] = transitive_table["norel"]["coref"] = ("norel",)
    # ifL(andL(relation_classes.COREF('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.NOREL(path=('t', t_event2))),
    #     relation_classes.NOREL(path=('t', t_event3))
    #     )
    # ifL(andL(relation_classes.NOREL('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.COREF(path=('t', t_event2))),
    #     relation_classes.NOREL(path=('t', t_event3))
    #     )
    # # transitive_table["coref"]["equal"] = transitive_table["equal"]["coref"] = ("equal",)
    # ifL(andL(relation_classes.COREF('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.EQUAL(path=('t', t_event2))),
    #     relation_classes.EQUAL(path=('t', t_event3))
    #     )
    # ifL(andL(relation_classes.EQUAL('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.COREF(path=('t', t_event2))),
    #     relation_classes.EQUAL(path=('t', t_event3))
    #     )
    # # transitive_table["coref"]["vague"] = transitive_table["vague"]["coref"] = ("vague",)
    # ifL(andL(relation_classes.COREF('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.VAGUE(path=('t', t_event2))),
    #     relation_classes.EQUAL(path=('t', t_event3))
    #     )
    # ifL(andL(relation_classes.VAGUE('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.COREF(path=('t', t_event2))),
    #     relation_classes.EQUAL(path=('t', t_event3))
    #     )
    #
    # transitive_table["before"]["vague"] = transitive_table["vague"]["before"] = ("not after", "not equal")
    ifL(andL(relation_classes.before('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.VAGUE(path=('t', t_event2))),
        orL(
            notL(relation_classes.after(path=('t', t_event3))),
            notL(relation_classes.EQUAL(path=('t', t_event3)))
        )
        )
    ifL(andL(relation_classes.VAGUE('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.before(path=('t', t_event2))),
        orL(
            notL(relation_classes.after(path=('t', t_event3))),
            notL(relation_classes.EQUAL(path=('t', t_event3)))
        )
        )
    # transitive_table["after"]["vague"] = transitive_table["vague"]["after"] = ("not before", "not equal")
    ifL(andL(relation_classes.after('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.VAGUE(path=('t', t_event2))),
        orL(
            notL(relation_classes.before(path=('t', t_event3))),
            notL(relation_classes.EQUAL(path=('t', t_event3)))
        )
        )
    ifL(andL(relation_classes.VAGUE('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.after(path=('t', t_event2))),
        orL(
            notL(relation_classes.before(path=('t', t_event3))),
            notL(relation_classes.EQUAL(path=('t', t_event3)))
        )
        )
    #
    # transitive_table["equal"]["parent_child"] = ("not after",)
    # ifL(andL(relation_classes.EQUAL('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.parent_child(path=('t', t_event2))),
    #     notL(relation_classes.after(path=('t', t_event3)))
    #     )
    # # transitive_table["equal"]["child_parent"] = ("not before",)
    # ifL(andL(relation_classes.EQUAL('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.child_parent(path=('t', t_event2))),
    #     notL(relation_classes.before(path=('t', t_event3)))
    #     )
    # transitive_table["equal"]["vague"] = transitive_table["vague"]["equal"] = ("not coref", "vague")
    ifL(andL(relation_classes.EQUAL('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.VAGUE(path=('t', t_event2))),
        relation_classes.VAGUE(path=('t', t_event3))
        )
    ifL(andL(relation_classes.VAGUE('x'), existsL(transitive("t", path=('x', transitive))), relation_classes.EQUAL(path=('t', t_event2))),
        relation_classes.VAGUE(path=('t', t_event3))
        )

