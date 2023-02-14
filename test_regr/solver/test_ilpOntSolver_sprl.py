import pytest


@pytest.fixture()
def sprl_input(request):
    import numpy as np

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
    test_graphResultsForPhraseToken['NONE_ENTITY']       = np.array([0.2 , 0.61, 0.48, 0.03, 0.51, 0.52])
    
    test_graphResultsForTripleRelations = dict()
    
    # triplet
    # triplet relation is a 3D array
    triplet_relation_table = np.random.rand(6, 6, 6) * 0.2
    triplet_relation_table[0, 1, 3] = 0.85 # ("stairs", "About 20 kids ", "on") - GT
    triplet_relation_table[0, 2, 3] = 0.78 # ("stairs", "About 20 kids", "on") - GT
    # ... can be more
    test_graphResultsForTripleRelations["triplet"] = triplet_relation_table
    
    # spatial_triplet
    # triplet relation is a 3D array
    spatial_triplet_relation_table = np.random.rand(6, 6, 6) * 0.2
    spatial_triplet_relation_table[0, 1, 3] = 0.74 # ("stairs", "About 20 kids ", "on")
    spatial_triplet_relation_table[0, 2, 3] = 0.83 # ("stairs", "About 20 kids", "on")
    # ... can be more
    test_graphResultsForTripleRelations["spatial_triplet"] = spatial_triplet_relation_table
    
    # region
    # triplet relation is a 3D array
    region_relation_table = np.random.rand(6, 6, 6) * 0.2
    region_relation_table[0, 1, 3] = 0.65 # ("stairs", "About 20 kids ", "on")
    region_relation_table[0, 2, 3] = 0.88 # ("stairs", "About 20 kids", "on")
    # ... can be more
    test_graphResultsForTripleRelations["region"] = region_relation_table


    yield test_phrase, test_graphResultsForPhraseToken, test_graphResultsForTripleRelations

@pytest.mark.skip(reason="define model building datanode")
@pytest.mark.gurobi
def test_main_sprl(sprl_input):
    from domiknows.solver.ilpOntSolverFactory import ilpOntSolverFactory
    from domiknows.solver.ilpOntSolver import ilpOntSolver
    from domiknows.graph import Graph

    test_phrase, test_graphResultsForPhraseToken, test_graphResultsForTripleRelations = sprl_input

    # ------Call solver -------
    test_graph = Graph(iri='http://ontology.ihmc.us/ML/SPRL.owl', local='./examples/SpRL/')
    
    myilpOntSolver = ilpOntSolverFactory.getOntSolverInstance(test_graph)
    tokenResult, _, tripleRelationsResult = myilpOntSolver.calculateILPSelection(test_phrase, test_graphResultsForPhraseToken, None, test_graphResultsForTripleRelations)

    #------------------
    # evaluate
    #------------------
    # -- phrase results:
    assert tokenResult['LANDMARK'][0] == 1 # "stairs" - LANDMARK
    assert tokenResult['LANDMARK'].sum() == 1

    assert tokenResult['TRAJECTOR'][1] == 1 # "About 20 kids" - TRAJECTOR
    assert tokenResult['TRAJECTOR'][2] == 1 # "About 20 kids " - TRAJECTOR
    assert tokenResult['TRAJECTOR'].sum() == 2

    assert tokenResult['SPATIAL_INDICATOR'][3] == 1 # "on" - SPATIALINDICATOR
    assert tokenResult['SPATIAL_INDICATOR'].sum() == 1

    assert tokenResult['NONE_ENTITY'][4] == 1 # "hats" - NONE
    assert tokenResult['NONE_ENTITY'][5] == 1  # "traditional clothing" - NONE
    assert tokenResult['NONE_ENTITY'].sum() == 2

    # -- relation results:
    assert tripleRelationsResult['triplet'][0, 1, 3] == 1 # ("stairs", "About 20 kids ", "on")
    assert tripleRelationsResult['triplet'][0, 2, 3] == 1 # ("stairs", "About 20 kids", "on")
    assert tripleRelationsResult['triplet'].sum() == 2

    assert tripleRelationsResult['region'][0, 1, 3] == 1 # ("stairs", "About 20 kids ", "on")
    assert tripleRelationsResult['region'][0, 2, 3] == 1 # ("stairs", "About 20 kids", "on")
    assert tripleRelationsResult['region'].sum() == 2 # 0 elsewhere

    assert tripleRelationsResult['spatial_triplet'][0, 1, 3] == 1 # ("stairs", "About 20 kids ", "on")
    assert tripleRelationsResult['spatial_triplet'][0, 2, 3] == 1 # ("stairs", "About 20 kids", "on")
    assert tripleRelationsResult['spatial_triplet'].sum() == 2 # 0 elsewhere
