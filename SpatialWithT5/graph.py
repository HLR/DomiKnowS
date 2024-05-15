from domiknows.graph import Graph, Concept, Relation, EnumConcept
from domiknows.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, V, nandL, exactL
from domiknows.graph.concept import EnumConcept

CONSTRAIN_ACTIVE = True

Graph.clear()
Concept.clear()
Relation.clear()
with Graph('spatialQArule') as graph:
    context = Concept(name="context")
    question = Concept(name="question")
    answer = Concept(name="answer")
    rel_context_contain_question, = context.contains(question)
    rel_question_contain_answer, = question.contains(answer)

    answer_relations = answer(name="answer_relations", ConceptClass=EnumConcept,
                              values=["left", "right", "above", "below", "behind", "front",
                               "near", "far", "disconnected", "touch", "overlap", "coveredby",
                               "inside", "cover", "contain", "<pad>", "<eos>"])
    
    inverse = Concept(name="inverse")
    inv_question1, inv_question2 = inverse.has_a(arg1=question, arg2=question)

    # # List of opposite relations
    # above <-> below, left <-> right, front <-> behind
    inverse_list1 = [('above', 'below'), ('left', 'right'), ('front', 'behind'), ('coveredby', 'cover'),
                      ('inside', 'contain')]
    
    for ans1_str, ans2_str in inverse_list1:

        post_fix="_"+ans1_str+"_"+ans2_str
        ans1=answer_relations.__getattr__(ans1_str)
        ans2=answer_relations.__getattr__(ans2_str)

        ifL(inverse('aconnect'+post_fix),
            ifL(andL(question("q1"+post_fix,path=('aconnect'+post_fix,inv_question1)),question("q2"+post_fix,path=('aconnect'+post_fix,inv_question2))),
                ifL(andL(
                        ans1(path=('q1'+post_fix,rel_question_contain_answer)),
                        notL(existsL(ans2(path=('q2'+post_fix,rel_question_contain_answer))))),
                )
            )
        )
    #
    # # Similar to Inverse relation
    # inverse_list2 = [(near, near), (far, far), (touch, touch), (disconnected, disconnected), (overlap, overlap),
    #                  (coveredby, inside), (cover, contain)]
    # for ans1, ans2 in inverse_list2:
    #     ifL(andL(ans1('x'), existsL(inverse('s', path=('x', inverse)))),
    #         ans2(path=('s', inv_question2)))
    #

    # # Transitive constrains
    # transitive = Concept(name="transitive")
    # Consists of three relation e.g. if question 1 with rel1 and question 2 with rel2, question 3 should have rel3
    # tran_quest1, tran_quest2, tran_quest3 = transitive.has_a(arg11=question, arg22=question, arg33=question)
    #
    # # if question 1 with rel1 and question 2 with rel1, question 3 should have rel1
    # transitive_1 = [left, right, above, below, behind, front, inside, contain]
    # for rel in transitive_1:
    #     ifL(andL(rel('x'),
    #              existsL(transitive("t", path=('x', transitive))),
    #              rel(path=('t', tran_quest2))),
    #         rel(path=('t', tran_quest3)))

    # if question 1 with rel2 and question 2 with rel1, question 3 should have rel2
    # # Transitive of cover and contain
    # transitive_2 = [(coveredby, inside), (cover, contain)]
    # for rel1, rel2 in transitive_2:
    #     ifL(andL(rel2('x'),
    #              existsL(transitive("t", path=('x', transitive))),
    #              rel1(path=('t', tran_quest2))),
    #         rel2(path=('t', tran_quest3)))
    #

    # if question 1 with rel1 and question 2 with rel2, question 3 should have rel2
    # # Transitive of inside/cover with position
    # transitive_3_1 = [inside, coveredby]
    # transitive_3_2 = [left, right, above, below, behind, front, near, far, disconnected]
    # for rel1 in transitive_3_1:
    #     for rel2 in transitive_3_2:
    #         ifL(andL(rel1('x'),
    #                  existsL(transitive("t", path=('x', transitive))),
    #                  rel2(path=('t', tran_quest2))),
    #             rel2(path=('t', tran_quest3)))
    #

    # # Transitive + topo constrains
    # tran_topo = Concept(name="transitive_topo")
    # tran_topo_quest1, tran_topo_quest2, tran_topo_quest3, tran_topo_quest4 = transitive.has_a(arg111=question,
    #                                                                                           arg222=question,
    #                                                                                           arg333=question,
    #                                                                                           arg444=question)
    # # (x inside y) + (h inside z) + (y direction z) => (x direction h)

    # tran_topo_2_1 = [inside, coveredby]
    # tran_topo_2_2 = [left, right, above, below, behind, front, near, far, disconnected]
    # for rel1 in tran_topo_2_1:
    #     for rel2 in tran_topo_2_2:
    #         ifL(andL(rel1('x'),
    #                  existsL(tran_topo('to', path=('x', tran_topo))),
    #                  rel1(path=('to', tran_topo_quest2)),
    #                  rel2(path=('to', tran_topo_quest3))
    #                  ),
    #             rel2(path=('to', tran_topo_quest4)))
