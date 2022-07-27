import sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")

from regex import B
from regr.graph import Graph, Concept, Relation, EnumConcept
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, V, nandL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('bio_graph') as graph:

    sentence = Concept(name="sentence")
    word = Concept(name="word")

    (sent_word_contains,) = sentence.contains(word)


    i_LOC = word("I-LOC")
    b_LOC = word("B-LOC")   ### how to insert b_LOC.before.i_LOC

    ### BIO constraints
    before = Concept(name="before")
    (b_prefix, i_prefix) = before.has_a(word,word)
    # before  -> B-, I-

    ## bio constraints start
    ifL(
        before('x'), 
        ifL(
           b_LOC(path=('x', b_prefix)),
           notL(
                i_LOC(path=('x', b_prefix))
                ),
            ),
        active = True
        )
    ### bio constraints end

graph.detach()







    ### hossein sent it to me in the chatbox
    # ifL(
    #     before('x'), 
    #     ifL(
    #         b(path=(‘x’, arg2)), 
    #         notL(b(path=(‘x’, arg1)))
	# 	)
    #     active = True
    # )


    #### my previous logic writing, maybe wrong, comment
    # ifL(
    #     before('x'), 
    #     andL(b_LOC(path=('x', bio_1)), i_LOC(path=('x', bio_2))), 
    #     active = True
    # )














# from regex import B
# from regr.graph import Graph, Concept, Relation, EnumConcept
# from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, V, nandL

# Graph.clear()
# Concept.clear()
# Relation.clear()

# with Graph('bio_graph') as graph:

#     sentence = Concept(name="sentence")
#     word = Concept(name="word")

#     (sent_word_contains,) = sentence.contains(word)

#     # before.has_a(word, word)
    
#     # ### 11 classes
#     # label_list = ['<pad>', '<bos>', '<eos>', 'O', 'I-PER', 'I-ORG', 'I-LOC', 'I-MISC', 'B-MISC', 'B-ORG', 'B-LOC']
#     # label = word(name="tag", ConceptClass=EnumConcept, values=label_list)
#     ### ['<pad>', '<bos>', '<eos>', 'O', 'I-PER', 'I-ORG', 'I-LOC', 'I-MISC', 'B-MISC', 'B-ORG', 'B-LOC']
#     pad = word("<pad>")
#     bos = word("<bos>")
#     eos = word("<eos>")
#     out_bio = word("O")
#     i_PER = word("I-PER")
#     i_ORG = word("I-ORG")
#     i_LOC = word("I-LOC")
#     i_MISC = word("I-MISC")
#     b_MISC = word("B-MISC") ### how to insert b_MISC.before.i_MISC
#     b_ORG = word("B-ORG")   ### how to insert b_ORG.before.i_ORG
#     b_LOC = word("B-LOC")   ### how to insert b_LOC.before.i_LOC

#     ### BIO constraints
#     before = Concept(name="before")
#     (b_prefix, i_prefix) = before.has_a(word,word)

#     # before  -> B-, I-

#     # ifL(
#     #     before('x'), 
#     #     ifL(
#     #         b(path=(‘x’, arg2)), 
#     #         notL(b(path=(‘x’, arg1)))
# 	# 	)
#     #     active = True
#     # )

#     ## bio constraints start
#     ifL(
#         before('x'), 
#         ifL(
#            b_LOC(path=('x', b_prefix)),
#            notL(
#                 i_LOC(path=('x', b_prefix),)
#                 ),
#             ),
#         active = True
#         )

#     ifL(
#         before('x'), 
#         ifL(
#            b_ORG(path=('x', b_prefix)),
#            notL(
#                 i_ORG(path=('x', b_prefix),)
#                 ),
#             ),
#         active = True
#         )

#     ifL(
#         before('x'), 
#         ifL(
#            b_MISC(path=('x', b_prefix)),
#            notL(
#                 i_MISC(path=('x', b_prefix),)
#                 ),
#         ),
#         active = True
#         )
#     ### bio constraints end


#     # ifL(
#     #     before('x'), 
#     #     andL(b_LOC(path=('x', bio_1)), i_LOC(path=('x', bio_2))), 
#     #     active = True
#     # )

#     # ifL(
#     #     before('x'), 
#     #     andL(b_ORG(path=('x', bio_1)), i_ORG(path=('x', bio_2))), 
#     #     active = True
#     # )

#     # ifL(
#     #     before('x'), 
#     #     andL(b_MISC(path=('x', bio_1)), i_MISC(path=('x', bio_2))), 
#     #     active = True
#     # )
 


#     nandL(pad, bos)
#     nandL(pad, eos)
#     nandL(pad, out_bio)
#     nandL(pad, i_PER)
#     nandL(pad, i_ORG)
#     nandL(pad, i_LOC)
#     nandL(pad, i_MISC)
#     nandL(pad, b_MISC)
#     nandL(pad, b_ORG)
#     nandL(pad, b_LOC)

#     nandL(bos, eos)
#     nandL(bos, out_bio)
#     nandL(bos, i_PER)
#     nandL(bos, i_ORG)
#     nandL(bos, i_LOC)
#     nandL(bos, i_MISC)
#     nandL(bos, b_MISC)
#     nandL(bos, b_ORG)
#     nandL(bos, b_LOC)

#     nandL(eos, out_bio)
#     nandL(eos, i_PER)
#     nandL(eos, i_ORG)
#     nandL(eos, i_LOC)
#     nandL(eos, i_MISC)
#     nandL(eos, b_MISC)
#     nandL(eos, b_ORG)
#     nandL(eos, b_LOC)

#     nandL(out_bio, i_PER)
#     nandL(out_bio, i_ORG)
#     nandL(out_bio, i_LOC)
#     nandL(out_bio, i_MISC)
#     nandL(out_bio, b_MISC)
#     nandL(out_bio, b_ORG)
#     nandL(out_bio, b_LOC)

#     nandL(i_PER, i_ORG)
#     nandL(i_PER, i_LOC)
#     nandL(i_PER, i_MISC)
#     nandL(i_PER, b_MISC)
#     nandL(i_PER, b_ORG)
#     nandL(i_PER, b_LOC)

#     nandL(i_ORG, i_LOC)
#     nandL(i_ORG, i_MISC)
#     nandL(i_ORG, b_MISC)
#     nandL(i_ORG, b_ORG)
#     nandL(i_ORG, b_LOC)

#     nandL(i_LOC, i_MISC)
#     nandL(i_LOC, b_MISC)
#     nandL(i_LOC, b_ORG)
#     nandL(i_LOC, b_LOC)

#     nandL(i_MISC, b_MISC)
#     nandL(i_MISC, b_ORG)
#     nandL(i_MISC, b_LOC)

#     nandL(b_MISC, b_ORG)
#     nandL(b_MISC, b_LOC)

#     nandL(b_ORG, b_LOC)

#     orL(pad, bos, eos, out_bio, i_PER, i_ORG, i_LOC, i_MISC, b_MISC, b_ORG, b_LOC)


# # 'Paris Hilton lives in New York .'
# # Paris       B-LOC
# # Hilton      O
# # lives       O
# # in          O
# # New         B-LOC
# # York        I-LOC
# # .           O






