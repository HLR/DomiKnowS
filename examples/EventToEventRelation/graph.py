from regr.graph import Graph, Concept, Relation, EnumConcept
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, V, nandL, exactL

Graph.clear()
Concept.clear()
Relation.clear()
with Graph('Useful_arg') as graph:

    # Group of sentence
    paragraph = Concept(name="paragraph")
    event = Concept(name="event")
    paragraph_contain, = paragraph.contains(event)

    # each relation could be one the following four concepts:
    # sub-event relation consists of four relations, parent child, child parent, coref, and noref
    sub_relation = event(name="sub_relation", ConceptClass=EnumConcept, values=["parent_child", "child_parent",
                                                                                "coref", "noref"])

    temp_relation = event(name="temp_relation", ConceptClass=EnumConcept, values=["before", "after",
                                                                                 "equal", "vague"])

    # Only one of the labels to be true
    ifL(event, exactL(
        sub_relation.parent_child, sub_relation.child_parent, sub_relation.coref, sub_relation.noref,
        temp_relation.before, temp_relation.after, temp_relation.equal, temp_relation.vague))

    ifL(event, atMostL(
        sub_relation.parent_child, sub_relation.child_parent, sub_relation.coref, sub_relation.noref,
        temp_relation.before, temp_relation.after, temp_relation.equal, temp_relation.vague))

    # Symmetric Constrain
    symmetric = Concept("symmetric")
    s_event1, s_event2 = symmetric.has_a(arg1=event, arg2=event)
    # TODO: Symmetric of subevent relation

    # Before(e1, e2) <=> After(e2, e1)
    ifL(temp_relation.before('x'), temp_relation.after(path=('x', symmetric, s_event2)))
    ifL(temp_relation.after('x'), temp_relation.before(path=('x', symmetric, s_event2)))

    # Equal(e1, e2) <=> Vague(e2, e1)
    ifL(temp_relation.equal('x'), temp_relation.vague(path=('x', symmetric, s_event2)))
    ifL(temp_relation.vague('x'), temp_relation.equal(path=('x', symmetric, s_event2)))

    # TODO: Transitive Constrains
    transitive_table = {
        temp_relation.after: {}, temp_relation.before: {}, temp_relation.vague: {}, temp_relation.equal: {},
        sub_relation.parent_child: {}, sub_relation.child_parent: {}, sub_relation.coref: {}, sub_relation.noref: {}
    }
