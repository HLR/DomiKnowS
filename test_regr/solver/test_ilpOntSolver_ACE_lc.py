import pytest

# Testing ILP solver with ACE graph based on constrains specified using logical constrains
@pytest.fixture()
def emr_input(request):
    import numpy as np
    from regr.graph import Graph, Concept, andL, nandL, notL, ifL, existsL, orL

    with Graph('global') as graph:
        #graph.ontology = ('http://ontology.ihmc.us/ML/ACE.owl', './')
    
        with Graph('linguistic') as ling_graph:
            word = Concept('word')
            phrase = Concept('phrase')
            sentence = Concept('sentence')
            (rel_sentence_contains_phrase, ) = sentence.contains(phrase)
            (rel_sentence_contains_word, ) = sentence.contains(word)
            (rel_phrase_contains_word, ) = phrase.contains(word)
            pair = Concept('pair')
            (rel_pair_phrase1, rel_pair_phrase2, ) = pair.has_a(phrase, phrase)
    
        with Graph('application') as app_graph:
            FAC = phrase(name='FAC') # Facility
            GPE = phrase(name='GPE') # Geo-Political Entity
            LOC = phrase(name='LOC') # Location
            ORG = phrase(name='ORG') # Organization
            PER = phrase(name='PER') # Person
            VEH = phrase(name='VEH') # Vehicle
            WEA = phrase(name='WEA') # Weapon
    
            FAC.not_a(GPE, LOC, VEH, WEA, PER, ORG)
            GPE.not_a(FAC, LOC, VEH, WEA, PER, ORG)
            LOC.not_a(FAC, GPE, VEH, WEA, PER, ORG)
            VEH.not_a(FAC, GPE, LOC, WEA, PER, ORG)
            WEA.not_a(FAC, GPE, LOC, VEH, PER, ORG)
            PER.not_a(FAC, GPE, LOC, VEH, WEA, ORG)
            ORG.not_a(FAC, GPE, LOC, VEH, WEA, PER)
            
            ART = pair(name="ART")
            GEN_AFF = pair(name="GEN-AFF")
            METONYMY = pair(name="METONYMY")
            ORG_AFF = pair(name="ORG-AFF")
            PART_WHOLE = pair(name="PART-WHOLE") # Part-Whole
            PER_SOC = pair(name="PER-SOC")
            PHYS = pair(name="PHYS")
    
            # -------------- Example 1----------

            # 'PHYS': ['PER', 'LOC'], ['PER', 'FAC'], ['PER', 'GPE']
            ifL(PHYS, ('x', 'y'), ifL(PER, ('x', ), orL(LOC, FAC, GPE), ('y',)))
            
            # 'PHYS': ['FAC', 'LOC'], ['FAC', 'FAC'], ['FAC', 'GPE']
            ifL(PHYS, ('x', 'y'), ifL(FAC, ('x', ), orL(LOC, FAC, GPE), ('y',)))
    
            # 'PHYS': ['LOC', 'LOC'], ['LOC', 'GPE'], ['LOC', 'FAC'],  ['LOC', 'PER']
            ifL(PHYS, ('x', 'y'), ifL(LOC, ('x', ), orL(LOC, GPE, FAC, PER), ('y',)))
    
            # 'PHYS': ['GPE', 'LOC'], ['GPE', 'GPE'], ['GPE', 'FAC'], ['GPE', 'PER']
            ifL(PHYS, ('x', 'y'), ifL(GPE, ('x', ), orL(LOC, GPE, FAC, PER), ('y',)))
    
            # 'PHYS': ['ORG', 'GPE'], ['ORG', 'FAC']
            ifL(PHYS, ('x', 'y'), ifL(ORG, ('x', ), orL(GPE, FAC), ('y',)))
    
            # -------------- Example 2 ---------
            
            # 'GEN-AFF': ['PER', 'GPE'], ['PER', 'PER'], ['PER', 'LOC'], ['PER', 'ORG']
            ifL(GEN_AFF, ('x', 'y'), ifL(PER, ('x', ), orL(GPE, PER, LOC, ORG), ('y',)))

            # 'GEN-AFF': ['ORG', 'LOC'], ['ORG', 'GPE'], ['ORG', 'ORG']
            ifL(GEN_AFF, ('x', 'y'), ifL(ORG, ('x', ), orL(LOC, GPE, ORG), ('y',)))

            # 'GEN-AFF':['GPE', 'PER']
            ifL(GEN_AFF, ('x', 'y'), ifL(GPE, ('x', ), PER, ('y',)))
            
    test_phrase = [("John", "NNP"), ("works", "VBN"), ("for", "IN"), ("IBM", "NNP")]
    conceptNamesList = ['PER', 'LOC', 'FAC', 'GPE', 'ORG']
    relationNamesList = ['PHYS', 'GEN-AFF']

    # tokens
    test_graphResultsForPhraseToken = {}
    #                                                  John  works for  IBM
    test_graphResultsForPhraseToken['PER'] = np.array([0.7, 0.1, 0.02, 0.6])
    test_graphResultsForPhraseToken['LOC'] = np.array([0.5, 0.2, 0.03, 0.91])
    test_graphResultsForPhraseToken['FAC'] = np.array([0.3, 0.6, 0.05, 0.5])
    test_graphResultsForPhraseToken['GPE'] = np.array([0.3, 0.4, 0.1 , 0.3])
    test_graphResultsForPhraseToken['ORG'] = np.array([0.1, 0.9, 0.9 , 0.1])
    
    test_graphResultsForPhraseRelation = dict()
    # PHYS
    #                                    John  works for   IBM
    PHYS_relation_table = np.array([[0.40, 0.20, 0.20, 0.63],  # John
                                        [0.00, 0.00, 0.40, 0.30],  # works
                                        [0.02, 0.03, 0.05, 0.10],  # for
                                        [0.65, 0.20, 0.10, 0.30],  # IBM
                                        ])
    test_graphResultsForPhraseRelation['PHYS'] = PHYS_relation_table

    # GEN-AFF
    #                                   John  works for   IBM
    GEN_AFF_relation_table = np.array([[0.10, 0.20, 0.20, 0.06],  # John
                                       [0.00, 0.00, 0.20, 0.10],  # works
                                       [0.02, 0.03, 0.05, 0.10],  # for
                                       [0.10, 0.20, 0.10, 0.00],  # IBM
                                       ])
    test_graphResultsForPhraseRelation['GEN-AFF'] = GEN_AFF_relation_table


    yield app_graph, test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation

@pytest.mark.gurobi
def test_main_emr_owl(emr_input):
    from regr.solver.ilpOntSolverFactory import ilpOntSolverFactory
    from regr.solver.ilpOntSolver import ilpOntSolver

    app_graph, test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation = emr_input

    # ------Call solver -------
    myilpOntSolver = ilpOntSolverFactory.getOntSolverInstance(app_graph)
    tokenResult, relationsResult, _ = myilpOntSolver.calculateILPSelection(test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation, None)

    # -------Evaluate results
    # token results:
    # expect John[PER] works[o] for[o] IBM[LOC]
    assert tokenResult['PER'].sum() == 1
    assert tokenResult['LOC'].sum() == 1
    assert tokenResult['FAC'].sum() == 0
    assert tokenResult['GPE'].sum() == 0
    assert tokenResult['ORG'].sum() == 2

    assert tokenResult['PER'][0] == 1  # John
    assert tokenResult['ORG'][1] == 1  # works
    assert tokenResult['ORG'][2] == 1  # for
    assert tokenResult['LOC'][3] == 1  # IBM

    # relation results:
    # (John, IBM)[PHYS]
    assert relationsResult['PHYS'].sum().sum() == 1
    #assert relationsResult['PHYS'][0][3] == 1 # John - IBM
    assert relationsResult['GEN-AFF'].sum().sum() == 0
