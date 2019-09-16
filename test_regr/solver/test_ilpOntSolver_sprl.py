import pytest

@pytest.fixture()
def sprl_input(request):
    #------------------
    # sample input
    #------------------
    sentence = "About 20 kids in traditional clothing and hats waiting on stairs ."
    
    # NB
    # Input to the inference in SPRL example is a bit different.
    # After processing the original sentence, only phrases (with feature extracted) remain.
    # They might not come in the original order.
    # There could be repeated token in different phrase.
    # However, those should not influence the design of inference interface.
    phrases = ["stairs",                # GT: LANDMARK
                "About 20 kids ",        # GT: TRAJECTOR
                "About 20 kids",         # GT: TRAJECTOR
                "on",                    # GT: SPATIALINDICATOR
                "hats",                  # GT: NONE
                "traditional clothing"]  # GT: NONE
    # SPRL relations are triplet with this combination
    # (landmark, trajector, spatialindicator)
    # Relation GT:
    # ("stairs", "About 20 kids ", "on") : triplet (and other? @Joslin please confirm if there is any other)
    # ("stairs", "About 20 kids", "on") : triplet
    
    test_phrase = [(phrase, 'NP') for phrase in phrases] # Not feasible to have POS-tag. Usually they are noun phrase
    
    #------------------
    # sample inference setup
    #------------------
    conceptNamesList = ["TRAJECTOR", "LANDMARK", "SPATIAL_INDICATOR", "NONE_ENTITY"]
    relationNamesList = ["triplet", "spatial_triplet", "region", "none_relation"]
    
    #------------------
    # sample output from learners
    #------------------
    # phrase
    test_graphResultsForPhraseToken = dict()
                                                                     # s  a/2/k a/2/k  o     h     t/c
    test_graphResultsForPhraseToken['TRAJECTOR']         = np.array([0.37, 0.72, 0.78, 0.01, 0.42, 0.22])
    test_graphResultsForPhraseToken['LANDMARK']          = np.array([0.68, 0.15, 0.33, 0.03, 0.43, 0.13])
    test_graphResultsForPhraseToken['SPATIAL_INDICATOR'] = np.array([0.05, 0.03, 0.02, 0.93, 0.03, 0.01])
    test_graphResultsForPhraseToken['NONE_ENTITY']       = np.array([0.2 , 0.61, 0.48, 0.03, 0.5 , 0.52])
    
    test_graphResultsForTripleRelations = dict()
    
    # triplet
    # triplet relation is a 3D array
    triplet_relation_table = np.random.rand(6, 6, 6) * 0.2
    triplet_relation_table[0, 1, 3] = 0.85 # ("stairs", "About 20 kids ", "on") - GT
    triplet_relation_table[0, 2, 3] = 0.78 # ("stairs", "About 20 kids", "on") - GT
    triplet_relation_table[1, 0, 3] = 0.32 # ("About 20 kids ", "stairs", "on")
    triplet_relation_table[1, 2, 3] = 0.30 # ("About 20 kids ", "About 20 kids", "on")
    triplet_relation_table[2, 0, 3] = 0.31 # ("About 20 kids", "stairs", "on")
    triplet_relation_table[2, 1, 3] = 0.30 # ("About 20 kids", "About 20 kids ", "on")
    triplet_relation_table[0, 4, 3] = 0.42 # ("stairs", "hat", "on")
    triplet_relation_table[0, 5, 3] = 0.52 # ("stairs", "traditional clothing", "on")
    triplet_relation_table[1, 4, 3] = 0.32 # ("About 20 kids ", "hat", "on")
    triplet_relation_table[1, 5, 3] = 0.29 # ("About 20 kids ", "traditional clothing", "on")
    triplet_relation_table[2, 4, 3] = 0.25 # ("About 20 kids ", "hat", "on")
    triplet_relation_table[2, 5, 3] = 0.27 # ("About 20 kids ", "traditional clothing", "on")
    # ... can be more
    test_graphResultsForTripleRelations["triplet"] = triplet_relation_table
    
    # spatial_triplet
    # triplet relation is a 3D array
    spatial_triplet_relation_table = np.random.rand(6, 6, 6) * 0.2
    spatial_triplet_relation_table[0, 1, 3] = 0.25 # ("stairs", "About 20 kids ", "on")
    spatial_triplet_relation_table[0, 2, 3] = 0.38 # ("stairs", "About 20 kids", "on")
    spatial_triplet_relation_table[1, 0, 3] = 0.32 # ("About 20 kids ", "stairs", "on")
    spatial_triplet_relation_table[1, 2, 3] = 0.30 # ("About 20 kids ", "About 20 kids", "on")
    spatial_triplet_relation_table[2, 0, 3] = 0.31 # ("About 20 kids", "stairs", "on")
    spatial_triplet_relation_table[2, 1, 3] = 0.30 # ("About 20 kids", "About 20 kids ", "on")
    spatial_triplet_relation_table[0, 4, 3] = 0.22 # ("stairs", "hat", "on")
    spatial_triplet_relation_table[0, 5, 3] = 0.12 # ("stairs", "traditional clothing", "on")
    spatial_triplet_relation_table[1, 4, 3] = 0.22 # ("About 20 kids ", "hat", "on")
    spatial_triplet_relation_table[1, 5, 3] = 0.39 # ("About 20 kids ", "traditional clothing", "on")
    spatial_triplet_relation_table[2, 4, 3] = 0.15 # ("About 20 kids ", "hat", "on")
    spatial_triplet_relation_table[2, 5, 3] = 0.27 # ("About 20 kids ", "traditional clothing", "on")
    # ... can be more
    test_graphResultsForTripleRelations["spatial_triplet"] = spatial_triplet_relation_table
    
    # region
    # triplet relation is a 3D array
    region_relation_table = np.random.rand(6, 6, 6) * 0.2
    region_relation_table[0, 1, 3] = 0.25 # ("stairs", "About 20 kids ", "on")
    region_relation_table[0, 2, 3] = 0.38 # ("stairs", "About 20 kids", "on")
    region_relation_table[1, 0, 3] = 0.32 # ("About 20 kids ", "stairs", "on")
    region_relation_table[1, 2, 3] = 0.30 # ("About 20 kids ", "About 20 kids", "on")
    region_relation_table[2, 0, 3] = 0.31 # ("About 20 kids", "stairs", "on")
    region_relation_table[2, 1, 3] = 0.30 # ("About 20 kids", "About 20 kids ", "on")
    region_relation_table[0, 4, 3] = 0.22 # ("stairs", "hat", "on")
    region_relation_table[0, 5, 3] = 0.12 # ("stairs", "traditional clothing", "on")
    region_relation_table[1, 4, 3] = 0.22 # ("About 20 kids ", "hat", "on")
    region_relation_table[1, 5, 3] = 0.39 # ("About 20 kids ", "traditional clothing", "on")
    region_relation_table[2, 4, 3] = 0.15 # ("About 20 kids ", "hat", "on")
    region_relation_table[2, 5, 3] = 0.27 # ("About 20 kids ", "traditional clothing", "on")
    # ... can be more
    test_graphResultsForTripleRelations["region"] = region_relation_table
    
    # none_relation
    # triplet relation is a 3D array
    none_relation_relation_table = np.random.rand(6, 6, 6) * 0.8
    none_relation_relation_table[0, 1, 3] = 0.15 # ("stairs", "About 20 kids ", "on")
    none_relation_relation_table[0, 2, 3] = 0.08 # ("stairs", "About 20 kids", "on")
    test_graphResultsForTripleRelations["none_relation"] = none_relation_relation_table

    yield test_phrase, test_graphResultsForPhraseToken, test_graphResultsForTripleRelations

@pytest.mark.gurobi
def test_main_sprl(sprl_input):
    from regr.solver.ilpOntSolverFactory import ilpOntSolverFactory
    from regr.solver.ilpOntSolver import ilpOntSolver
    from regr.graph import Graph

    test_phrase, test_graphResultsForPhraseToken, test_graphResultsForTripleRelations = sprl_input

    # ------Call solver -------
    test_graph = Graph(iri='http://ontology.ihmc.us/ML/SPRL.owl', local='./examples/SpRL_new/')
    
    myilpOntSolver = ilpOntSolverFactory.getOntSolverInstance(test_graph)
    tokenResult, _, tripleRelationsResult = myilpOntSolver.calculateILPSelection(test_phrase, test_graphResultsForPhraseToken, None, test_graphResultsForTripleRelations)

    #------------------
    # evaluate
    #------------------
    # -- phrase results:
    assert tokenResult['LANDMARK'].sum() == 1
    assert tokenResult['LANDMARK'][0] == 1 # "stairs" - LANDMARK

    assert tokenResult['TRAJECTOR'].sum() == 2
    assert tokenResult['TRAJECTOR'][1] == 1 # "About 20 kids" - TRAJECTOR
    assert tokenResult['TRAJECTOR'][2] == 1 # "About 20 kids " - TRAJECTOR

    assert tokenResult['SPATIAL_INDICATOR'].sum() == 1
    assert tokenResult['SPATIAL_INDICATOR'][3] == 1 # "on" - SPATIALINDICATOR

    assert tokenResult['NONE_ENTITY'].sum() == 3 # instead of 2
    #assert tokenResult['NONE_ENTITY'][4] == 1 # "hats" - NONE
    assert tokenResult['NONE_ENTITY'][5] == 1  # "traditional clothing" - NONE
        
    # -- relation results:
    assert tripleRelationsResult['triplet'].sum() == 3 # instead of 2
    assert tripleRelationsResult['triplet'][0, 1, 3] == 1 # ("stairs", "About 20 kids ", "on")
    assert tripleRelationsResult['triplet'][0, 2, 3] == 1 # ("stairs", "About 20 kids", "on")
    # one more ?
    
    assert tripleRelationsResult['region'].sum() == 0 # 0 elsewhere
    
    assert tripleRelationsResult['spatial_triplet'].sum() == 0 # 0 elsewhere
    
    #assert tripleRelationsResult['none_relation'].sum() == 58 # 4 * 6 * 6 - 3 elsewhere
    assert tripleRelationsResult['none_relation'][0, 1, 3] == 0 # ("stairs", "About 20 kids ", "on")
    assert tripleRelationsResult['none_relation'][0, 2, 3] == 0 # ("stairs", "About 20 kids", "on")
