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
                                      "inside", "cover", "contain", "<eos>", "<pad>"])

    inverse = Concept(name="inverse")
    inv_question1, inv_question2 = inverse.has_a(arg1=question, arg2=question)

    # # List of opposite relations
    # above <-> below, left <-> right, front <-> behind
    inverse_list1 = [('above', 'below'), ('left', 'right'), ('front', 'behind'), ('coveredby', 'cover'),
                     ('inside', 'contain')]

    inverse_list1 += [(x, y) for y, x in inverse_list1]


    def create_symmetric(ans1, ans2, post_fix):
        ifL(inverse('aconnect' + post_fix),
            ifL(andL(question("q1" + post_fix, path=('aconnect' + post_fix, inv_question1)),
                     question("q2" + post_fix, path=('aconnect' + post_fix, inv_question2))),
                ifL(
                    existsL(ans1(path=('q1' + post_fix, rel_question_contain_answer))),
                    existsL(ans2(path=('q2' + post_fix, rel_question_contain_answer))),
                ),
                # existsL(ans1(path=('q1'+post_fix,rel_question_contain_answer)))
                ),
            name="if_inverse(%s)" % post_fix
            )

    def create_transitivity_relation(ans1, ans2, ans3, post_fix):
        ifL(transitive('atrans' + post_fix),
            ifL(andL(
                question("q1" + post_fix, path=('atrans' + post_fix, tran_quest1)),
                question("q2" + post_fix, path=('atrans' + post_fix, tran_quest2)),
                question("q3" + post_fix, path=('atrans' + post_fix, tran_quest3))
            ),
                ifL(andL(
                    existsL(ans1(path=('q1' + post_fix, rel_question_contain_answer))),
                    existsL(ans2(path=('q2' + post_fix, rel_question_contain_answer)))),
                    existsL(ans3(path=('q3' + post_fix, rel_question_contain_answer))),
                )
            )
            )


    for ans1_str, ans2_str in inverse_list1:
        post_fix = "_" + ans1_str + "_" + ans2_str
        ans1 = answer_relations.__getattr__(ans1_str)
        ans2 = answer_relations.__getattr__(ans2_str)
        create_symmetric(ans1, ans2, post_fix)

    #
    # # Similar to Inverse relation
    inverse_list2 = [('near', 'near'), ('far', 'far'), ('touch', 'touch'), ('disconnected', 'disconnected'),
                     ('overlap', 'overlap')]

    for ans1_str, ans2_str in inverse_list2:
        post_fix = "_" + ans1_str + "_" + ans2_str
        ans1 = answer_relations.__getattr__(ans1_str)
        ans2 = answer_relations.__getattr__(ans2_str)
        create_symmetric(ans1, ans2, post_fix)

    # # Transitive constrains
    transitive = Concept(name="transitive")
    # Consists of three relation e.g. if question 1 with rel1 and question 2 with rel2, question 3 should have rel3
    tran_quest1, tran_quest2, tran_quest3 = transitive.has_a(arg11=question, arg22=question, arg33=question)


    # Transitivity 1
    transitive_1 = ['left', 'right', 'above', 'below', 'behind', 'front', 'inside', 'contain']
    for answer_str in transitive_1:
        post_fix = "_" + answer_str + "_" + answer_str + "_" + answer_str + "_transitivity1"
        ans1 = answer_relations.__getattr__(answer_str)
        ans2 = answer_relations.__getattr__(answer_str)
        ans3 = answer_relations.__getattr__(answer_str)
        create_transitivity_relation(ans1, ans2, ans3, post_fix)

    # Transitivity 1
    transitive_2 = [('coveredby', 'inside'), ('cover', 'contain')]
    for answer1_str, answer2_str in transitive_2:
        post_fix = "_" + answer2_str + "_" + answer1_str + "_" + answer2_str + "_transitivity2"
        ans1 = answer_relations.__getattr__(answer2_str)
        ans2 = answer_relations.__getattr__(answer1_str)
        ans3 = answer_relations.__getattr__(answer2_str)
        create_transitivity_relation(ans1, ans2, ans3, post_fix)

    transitive_3_1 = ['inside', 'coveredby']
    transitive_3_2 = ['left', 'right', 'above', 'below', 'behind', 'front', 'near', 'far', 'disconnected']
    for answer1_str in transitive_3_1:
        for answer2_str in transitive_3_2:
            post_fix = "_" + answer1_str + "_" + answer2_str + "_" + answer2_str + "_transitivity3"
            ans1 = answer_relations.__getattr__(answer1_str)
            ans2 = answer_relations.__getattr__(answer2_str)
            ans3 = answer_relations.__getattr__(answer2_str)
            create_transitivity_relation(ans1, ans2, ans3, post_fix)

        # # Transitive constrains

    # Transitive + topo constrains
    tran_topo = Concept(name="transitive_topo")
    tran_topo_quest1, tran_topo_quest2, tran_topo_quest3, tran_topo_quest4 = tran_topo.has_a(arg111=question,
                                                                                              arg222=question,
                                                                                              arg333=question,
                                                                                              arg444=question)
    # (x inside y) + (h inside z) + (y direction z) => (x direction h)

    tran_topo_2_1 = ['inside', 'coveredby']
    tran_topo_2_2 = ['left', 'right', 'above', 'below', 'behind', 'front', 'near', 'far', 'disconnected']
    for answer1_str in tran_topo_2_1:
        for answer2_str in tran_topo_2_2:
            post_fix = "_" + answer1_str + "_" + answer1_str + "_" + answer2_str + "_" + answer1_str + "_transitivity_topo"
            ans1 = answer_relations.__getattr__(answer1_str)
            ans2 = answer_relations.__getattr__(answer1_str)
            ans3 = answer_relations.__getattr__(answer2_str)
            ans4 = answer_relations.__getattr__(answer1_str)
            ifL(tran_topo('atranstopo' + post_fix),
                ifL(andL(
                    question("q1" + post_fix, path=('atranstopo' + post_fix, tran_topo_quest1)),
                    question("q2" + post_fix, path=('atranstopo' + post_fix, tran_topo_quest2)),
                    question("q3" + post_fix, path=('atranstopo' + post_fix, tran_topo_quest3)),
                    question("q4" + post_fix, path=('atranstopo' + post_fix, tran_topo_quest4)),
                ),
                    ifL(andL(
                        existsL(ans1(path=('q1' + post_fix, rel_question_contain_answer))),
                            existsL(ans2(path=('q2' + post_fix, rel_question_contain_answer))),
                            existsL(ans3(path=('q3' + post_fix, rel_question_contain_answer)))
                        ),
                        existsL(ans4(path=('q3' + post_fix, rel_question_contain_answer))),
                    )
                )
                )
