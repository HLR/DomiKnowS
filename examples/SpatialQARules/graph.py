from regr.graph import Graph, Concept, Relation, EnumConcept
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, V, nandL, exactL

CONSTRAIN_ACTIVE = True

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('event_to_event') as graph:
    # Group of sentence
    story = Concept(name="story")
    question = Concept(name="question")
    story_contain, = story.contains(question)

    answer = question(name="answer", ConceptClass=EnumConcept,
                                      values=["yes", "no"])

    # Symmetric Constrain
    symmetric = Concept("symmetric")
    s_quest1, s_quest2 = symmetric.has_a(arg1=question, arg2=question)
    ifL(andL(answer.yes('x'), existsL(symmetric('s', path=('x', symmetric)))),
        answer.yes(path=('x', symmetric, s_quest2)), active=CONSTRAIN_ACTIVE)

    reverse = Concept("reverse")
    r_quest1, r_quest2 = reverse.has_a(arg1=question, arg2=question)
    ifL(andL(answer.yes('x'), existsL(reverse('s', path=('x', reverse)))),
        answer.no(path=('x', reverse, s_quest2)), active=CONSTRAIN_ACTIVE)

    transitive = Concept("transitive")
    t_quest1, t_quest2, t_quest3 = transitive.has_a(arg1=question, arg2=question, arg3=question)
    ifL(andL(answer.yes('x'), existsL(transitive("t", path=('x', transitive))), answer.yes(path=('t', t_quest2))),
        answer.yes(path=('t', t_quest3)), active=CONSTRAIN_ACTIVE)

    transitive_topo = Concept("transitive_topo")
    tt_quest1, tt_quest2, tt_quest3, tt_quest4 = transitive_topo.has_a(arg1=question, arg2=question, arg3=question, arg4=question)
    ifL(andL(answer.yes('x'), existsL(transitive("t", path=('x', transitive_topo))),
             answer.yes(path=('t', tt_quest2)), answer.yes(path=('t', tt_quest3))),
        answer.yes(path=('t', tt_quest4)), active=CONSTRAIN_ACTIVE)