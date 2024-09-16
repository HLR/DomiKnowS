from domiknows.graph import Graph, Concept, Relation, EnumConcept
from domiknows.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, V, nandL, exactL

CONSTRAIN_ACTIVE = True

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('spatial_QA_rule') as graph:
    # Group of sentence
    story = Concept(name="story")
    question = Concept(name="question")
    story_contain, = story.contains(question)

    left = question(name="left")
    right = question(name="right")
    above = question(name="above")
    below = question(name="below")
    behind = question(name="behind")
    front = question(name="front")
    near = question(name="near")
    far = question(name="far")
    disconnected = question(name="disconnected")
    touch = question(name="touch")
    overlap = question(name="overlap")
    coveredby = question(name="coveredby")
    inside = question(name="inside")
    cover = question(name="cover")
    contain = question(name="contain")
    output_for_loss = question(name="output_for_loss")

    # Only one label of opposite concepts
    exactL(left, right)
    exactL(above, below)
    exactL(behind, front)
    exactL(near, far)
    exactL(disconnected, touch)

    # Inverse Constrains
    inverse = Concept(name="inverse")
    inv_question1, inv_question2 = inverse.has_a(arg1=question, arg2=question)

    # First inverse relation, allow inverse back and forth
    inverse_list1 = [(above, below), (left, right), (front, behind), (coveredby, cover),
                    (inside, contain)]

    for ans1, ans2 in inverse_list1:
        ifL(andL(ans1('x'), existsL(inverse('s', path=('x', inverse)))),
            ans2(path=('s', inv_question2)))

        ifL(andL(ans2('x'), existsL(inverse('s', path=('x', inverse)))),
            ans1(path=('s', inv_question2)))

    # 2 PMD : = entropy + beta * constraint_loss ( Train with no-constraint first then working on)
    # symmetric
    inverse_list2 = [(near, near), (far, far), (touch, touch), (disconnected, disconnected), (overlap, overlap)]
    for ans1, ans2 in inverse_list2:
        ifL(andL(ans1('x'), existsL(inverse('s', path=('x', inverse)))),
            ans2(path=('s', inv_question2)))

    # Transitive constrains
    transitive = Concept(name="transitive")
    tran_quest1, tran_quest2, tran_quest3 = transitive.has_a(arg11=question, arg22=question, arg33=question)

    transitive_1 = [left, right, above, below, behind, front, inside, contain]

    for rel in transitive_1:
        ifL(andL(rel('x'),
                 existsL(transitive("t", path=('x', transitive))),
                 rel(path=('t', tran_quest2))),
            rel(path=('t', tran_quest3)))
    # Transitive of cover and contain
    transitive_2 = [(coveredby, inside), (cover, contain)]
    for rel1, rel2 in transitive_2:
        ifL(andL(rel2('x'),
                 existsL(transitive("t", path=('x', transitive))),
                 rel1(path=('t', tran_quest2))),
            rel2(path=('t', tran_quest3)))

    # Transitive of inside/cover with position
    transitive_3_1 = [inside, coveredby]
    transitive_3_2 = [left, right, above, below, behind, front, near, far, disconnected]
    for rel1 in transitive_3_1:
        for rel2 in transitive_3_2:
            ifL(andL(rel1('x'),
                     existsL(transitive("t", path=('x', transitive))),
                     rel2(path=('t', tran_quest2))),
                rel2(path=('t', tran_quest3)))

    # Transitive + topo constrains
    tran_topo = Concept(name="transitive_topo")
    tran_topo_quest1, tran_topo_quest2, tran_topo_quest3, tran_topo_quest4 = transitive.has_a(arg111=question,
                                                                                              arg222=question,
                                                                                              arg333=question,
                                                                                              arg444=question)
    # (x inside y) + (h inside z) + (y direction z) => (x direction h)
    tran_topo_2_1 = [inside, coveredby]
    tran_topo_2_2 = [left, right, above, below, behind, front, near, far, disconnected]
    for rel1 in tran_topo_2_1:
        for rel2 in tran_topo_2_2:
            ifL(andL(rel1('x'),
                     existsL(tran_topo('to', path=('x', tran_topo))),
                     rel1(path=('to', tran_topo_quest2)),
                     rel2(path=('to', tran_topo_quest3))
                     ),
                rel2(path=('to', tran_topo_quest4)))

    tran_topo_3_1 = [left, right, above, below, behind, front, near, far, disconnected]
    tran_topo_3_2 = [contain, cover]
    for rel1 in tran_topo_3_1:
        for rel2 in tran_topo_3_2:
            ifL(andL(rel1('x'),
                     existsL(tran_topo('to', path=('x', tran_topo))),
                     rel1(path=('to', tran_topo_quest2)),
                     rel2(path=('to', tran_topo_quest3))),
                rel1(path=('to', tran_topo_quest4)))