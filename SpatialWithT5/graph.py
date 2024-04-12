from domiknows.graph import Graph, Concept, Relation, EnumConcept
from domiknows.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, V, nandL, exactL
from domiknows.graph.concept import EnumConcept

CONSTRAIN_ACTIVE = True

Graph.clear()
Concept.clear()
Relation.clear()
with Graph('spatial_QA_rule') as graph:
    context = Concept(name="context")
    question = Concept(name="question")
    answer = Concept(name="answer")
    rel_context_contain_question, = context.contains(question)
    rel_question_contain_answer, = question.contains(answer)

    # relations = answer(name="relation", ConceptClass=EnumConcept,
    #                    values=["left", "right", "above", "below", "behind", "front",
    #                            "near", "far", "disconnected", "touch", "overlap", "covered by",
    #                            "inside", "cover", "contain"])

    relations = answer(name="relation")
    # Two issues:
    # 1. End of Sentence (Ignore word after EoS) to limit the scope of sentence


    # Example of Old relations
    # [
    # [rel1, rel2],
    # [rel2, rel3]
    # ]
    # ifL(andL(getattr(relations, 'left')('x'), existsL(inverse('s', path=('x', inverse)))),
    #         getattr(relations, 'right')(path=('s', inv_question2)))

