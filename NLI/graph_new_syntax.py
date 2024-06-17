from domiknows.graph import Graph, Concept, Relation, EnumConcept
from domiknows.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, V, nandL, fixedL, exactL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('NLI_Graph') as graph:
    pair = Concept(name='pair')
    premise = Concept(name='premise')
    hypothesis = Concept(name='hypothesis')
    (rel_pair_premise, rel_pair_hypothesis) = pair.has_a(arg1=premise, arg2=hypothesis)

    with Graph('task') as ngraph:
        # 3 classes in a multi-class setting
        nli_class = pair(name="nli_class", ConceptClass=EnumConcept,
                            values=["entailment", "neutral", "contradiction"])
        
        ifL(pair('x'), exactL(nli_class.entailment(path=('x',)), nli_class.neutral(path=('x',)), nli_class.contradiction(path=('x',))))

        ### Ent(X1, X2) => !CON(X2, X1)
        ### New Syntax
        ifL(
            andL(
                nli_class.entailment('x', 'y'),
                existsL(pair('y', 'x'))
            ),
            notL(
                nli_class.contradiction('y', 'x')
            ),
            active= True
        )
        
        ### old syntax
        # Ent(X1, X2) => !CON(X2, X1)
        ifL(
            andL(
                nli_class.entailment('x'), 
                existsL(
                    pair('y', 
                        path=(
                            ('x', rel_pair_premise, rel_pair_hypothesis.reversed), ### the original premise is a hypothesis in this new pair
                            ('x', rel_pair_hypothesis, rel_pair_premise.reversed) ### the hypothesis is a premise in this new pair
                        )
                    )
                )
            ),
            andL(
                notL(
                    nli_class.contradiction(
                        path=(
                            ('x', rel_pair_premise, rel_pair_hypothesis.reversed), ### the original premise is a hypothesis in this new pair
                            ('x', rel_pair_hypothesis, rel_pair_premise.reversed) ### the hypothesis is a premise in this new pair
                        )
                    )
                )
            ),
            active= False
        )
        
        ### corrected old syntax
        # Ent(X1, X2) => !CON(X2, X1)
        ifL(
            andL(
                nli_class.entailment('x'), 
                existsL(
                    pair('y', 
                        path=(
                            ('x', rel_pair_premise, rel_pair_premise.reversed), ### the original premise is a hypothesis in this new pair
                            ('x', rel_pair_hypothesis, rel_pair_hypothesis.reversed ) ### the hypothesis is a premise in this new pair
                        )
                    )
                )
            ),
            andL(
                notL(
                    nli_class.contradiction(
                        path=(
                            ('x', rel_pair_premise, rel_pair_premise.reversed ), ### the original premise is a hypothesis in this new pair
                            ('x', rel_pair_hypothesis, rel_pair_hypothesis.reversed) ### the hypothesis is a premise in this new pair
                        )
                    )
                )
            ),
            active= False
        )
        
        '''

        ### Neu(X1, X2) => !CON(X2, X1)
        ### New Syntax
        ifL(
            andL(
                nli_class.neutral('x', 'y'),
                existsL(pair('y', 'x'))
            ),
            notL(
                nli_class.contradiction('y', 'x')
            )
        )
        ### old syntax
        ifL(
            andL(
                nli_class.neutral('x'),
                existsL(
                    pair('y', 
                        path=(
                            ('x', rel_pair_premise, rel_pair_hypothesis.reversed), ### the original premise is a hypothesis in this new pair
                            ('x', rel_pair_hypothesis, rel_pair_premise.reversed) ### the hypothesis is a premise in this new pair
                        )
                    )
                )
            ),
            andL(
                notL(
                    nli_class.contradiction(
                        path=(
                            ('x', rel_pair_premise, rel_pair_hypothesis.reversed), ### the original premise is a hypothesis in this new pair
                            ('x', rel_pair_hypothesis, rel_pair_premise.reversed) ### the hypothesis is a premise in this new pair
                        )
                    )
                )
            )
        )

        ### CON(X1, X2) => CON(X2, X1)
        ### New Syntax
        ifL(
            andL(
                nli_class.contradiction('x', 'y'),
                existsL(pair('y', 'x'))
            ),
            nli_class.contradiction('y', 'x')
        )
        ### Old syntax
        ifL(
            andL(
                nli_class.contradiction('x'),
                existsL(
                    pair('y', 
                        path=(
                            ('x', rel_pair_premise, rel_pair_hypothesis.reversed), ### the original premise is a hypothesis in this new pair
                            ('x', rel_pair_hypothesis, rel_pair_premise.reversed) ### the hypothesis is a premise in this new pair
                        )
                    )
                )
            ),
            nli_class.contradiction(
                path=(
                    ('x', rel_pair_premise, rel_pair_hypothesis.reversed), ### the original premise is a hypothesis in this new pair
                    ('x', rel_pair_hypothesis, rel_pair_premise.reversed) ### the hypothesis is a premise in this new pair
                )
            )
        )

        ### Ent(X1, X2) + Ent(X2, X3) => Ent(X1, X3)
        ### New Syntax
        ifL(
            andL(
                nli_class.entailment('x', 'y'),
                andL(
                    existsL(
                        pair('y', 'z')
                    )   
                )   
            ),
            ifL(
                nli_class.entailment('y', 'z'),
                nli_class.entailment('x', 'z'),
            )
        )
        ### Old syntax
        ifL(
            andL(
                nli_class.entailment('x'),
                existsL(
                    andL(
                        pair('y',
                            path=(
                                ('x', rel_pair_hypothesis, rel_pair_premise.reversed) ### the hypothesis of the first pair is a premise in this new pair
                            )
                        ), 
                        pair('z',
                            path=(
                                ('y', rel_pair_hypothesis, rel_pair_hypothesis.reversed), ### the hypothesis of the second pair is a hypothesis in this new pair
                                ('x', rel_pair_premise, rel_pair_premise.reversed) ### the premise of the first pair is a premise in this new pair
                            )
                        ),
                    ) 
                ),
            ),
            ifL(
                andL(
                    pair('y',
                        path=(
                            ('x', rel_pair_hypothesis, rel_pair_premise.reversed) ### the hypothesis of the first pair is a premise in this new pair
                        )
                    ), 
                    pair('z',
                        path=(
                            ('y', rel_pair_hypothesis, rel_pair_hypothesis.reversed), ### the hypothesis of the second pair is a hypothesis in this new pair
                            ('x', rel_pair_premise, rel_pair_premise.reversed) ### the premise of the first pair is a premise in this new pair
                        )
                    ),
                    nli_class.entailment(path=('y')),
                ),
                nli_class.entailment(path=('z')) 
            )
        )

        ### To introduce the symmetric and transitive constraints, we first define the two new concepts to related multiple pairs to each other
        ### The symmetric concept is used to relate two pairs to each other
        symmetric = Concept(name="symmetric")
        s_pair1, s_pair2 = symmetric.has_a(arg1=pair, arg2=pair)

        ### The transitive concept is used to relate three pairs to each other
        transitive = Concept("transitive")
        t_pair1, t_pair2, t_pair3 = transitive.has_a(arg11=pair, arg22=pair, arg33=pair)

        ### Symmetric constraint
        #### Ent(X1, X2) => !CON(X2, X1)
        ifL(
            andL(
                nli_class.entailment('x'),
                existsL(
                    symmetric('s', path=('x', symmetric.reversed))
                )
            ),
            andL(
                notL(
                    nli_class.contradiction(path=('x', symmetric.reversed, s_pair2))
                )
            )
        )

        #### Neu(X1, X2) => !CON(X2, X1)
        ifL(
            andL(
                nli_class.neutral('x'),
                existsL(
                    symmetric('s', path=('x', symmetric.reversed))
                )
            ),
            andL(
                notL(
                    nli_class.contradiction(path=('x', symmetric.reversed, s_pair2))
                )
            )
        )

        #### CON(X1, X2) => CON(X2, X1)
        ifL(
            andL(
                nli_class.contradiction('x'),
                existsL(
                    symmetric('s', path=('x', symmetric.reversed))
                )
            ),
            nli_class.contradiction(path=('x', symmetric.reversed, s_pair2))
        )

        ### Transitive constraint
        #### Ent(X1, X2) + Ent(X2, X3) => Ent(X1, X3)
        ifL(
            andL(
                nli_class.entailment('x'),
                existsL(
                    transitive("t", path=('x', transitive.reversed))
                ),
            ),
            ifL(
                nli_class.entailment(path=('t', t_pair2)),
                nli_class.entailment(path=('t', t_pair3))
            )
        )
    '''
