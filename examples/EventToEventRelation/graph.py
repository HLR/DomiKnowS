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
    relation_classes = event_relation(name="relation_classes", ConceptClass=EnumConcept, values=["parent_child", "child_parent",
                                                                                "coref", "norel", "before", "after",
                                                                                         "EQUAL", "VAGUE"])

    # Only one of the labels to be true
    ifL(event_relation, exactL(
        relation_classes.parent_child, relation_classes.child_parent, relation_classes.coref, relation_classes.norel,
        relation_classes.before, relation_classes.after, relation_classes.EQUAL, relation_classes.VAGUE))

    ifL(event_relation, atMostL(
        relation_classes.parent_child, relation_classes.child_parent, relation_classes.coref, relation_classes.norel,
        relation_classes.before, relation_classes.after, relation_classes.EQUAL, relation_classes.VAGUE))

    # Symmetric Constrain
    symmetric = Concept("symmetric")
    s_event1, s_event2 = symmetric.has_a(arg1=event_relation, arg2=event_relation)
    # TODO: Symmetric of subevent relation

    # Before(e1, e2) <=> After(e2, e1)
    ifL(event_relation.before('x'), event_relation.after(path=('x', symmetric, s_event2)))
    ifL(event_relation.after('x'), event_relation.before(path=('x', symmetric, s_event2)))

    # Equal(e1, e2) <=> Vague(e2, e1)
    ifL(event_relation.EQUAL('x'), event_relation.VAGUE(path=('x', symmetric, s_event2)))
    ifL(event_relation.VAGUE('x'), event_relation.EQUAL(path=('x', symmetric, s_event2)))

    # # TODO: Transitive Constrains
    # transitive_table = {
    #     relation.after: {}, relation.before: {}, relation.vague: {}, relation.equal: {},
    #     relation.parent_child: {}, relation.child_parent: {}, relation.coref: {}, relation.noref: {}
    # }
    transitive = Concept("transitive")
    t_event1, t_event2, t_event3 = transitive.has_a(arg11=event_relation, arg22=event_relation, arg33=event_relation)
