from regr.graph import Graph, Concept, Relation, EnumConcept
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, V, nandL, exactL

CONSTRAIN_ACTIVE = True

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('spatial_QA_rule') as graph:
    # Group of sentence
    story = Concept(name="story")
    question = Concept(name="question")
    story_contain, = story.contains(question)

    answer_class = question(name="answer_class", ConceptClass=EnumConcept,
                            values=["yes", "no", "dk"])

    # Symmetric Constrain
    symmetric = Concept(name="symmetric")
    s_quest1, s_quest2 = symmetric.has_a(arg1=question, arg2=question)
    ifL(andL(answer_class.yes('x'), existsL(symmetric('s', path=('x', symmetric)))),
        answer_class.yes(path=('s', s_quest2)))

    reverse = Concept(name="reverse")
    r_quest1, r_quest2 = reverse.has_a(arg10=question, arg20=question)
    ifL(andL(answer_class.yes('x'), existsL(reverse('r', path=('x', reverse)))),
        answer_class.no(path=('r', r_quest2)))

    transitive = Concept(name="transitive")
    t_quest1, t_quest2, t_quest3 = transitive.has_a(arg11=question, arg22=question, arg33=question)
    ifL(andL(answer_class.yes('x'), existsL(transitive("t", path=('x', transitive))), answer_class.yes(path=('t', t_quest2))),
        answer_class.yes(path=('t', t_quest3)))

    transitive_topo = Concept(name="transitive_topo")
    tt_quest1, tt_quest2, tt_quest3, tt_quest4 = transitive_topo.has_a(arg111=question, arg222=question,
                                                                       arg333=question, arg444=question)
    ifL(andL(answer_class.yes('x'), existsL(transitive("t", path=('x', transitive_topo))),
             answer_class.yes(path=('t', tt_quest2)), answer_class.yes(path=('t', tt_quest3))),
        answer_class.yes(path=('t', tt_quest4)))
