import pytest

@pytest.fixture()
def sprl_input(request):
    import numpy as np
    from regr.graph import Graph
    from regr.graph import DataGraph

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
    # create graphs
    #------------------
    test_ont_graph = Graph(iri='http://ontology.ihmc.us/ML/SPRL.owl', local='./examples/SpRL_new/')
    test_data_graph = DataGraph(test_phrase, test_ont_graph)
   
    #------------------
    # sample output from learners
    #------------------
    # phrase
    test_data_graph.setPredictionResult("Learned", 0, 'TRAJECTOR',         prediction=0.37)
    test_data_graph.setPredictionResult("Learned", 0, 'LANDMARK',          prediction=0.68)
    test_data_graph.setPredictionResult("Learned", 0, 'SPATIAL_INDICATOR', prediction=0.05)
    test_data_graph.setPredictionResult("Learned", 0, 'NONE_ENTITY',       prediction=0.20)

    test_data_graph.setPredictionResult("Learned", 1, 'TRAJECTOR',         prediction=0.72)
    test_data_graph.setPredictionResult("Learned", 1, 'LANDMARK',          prediction=0.15)
    test_data_graph.setPredictionResult("Learned", 1, 'SPATIAL_INDICATOR', prediction=0.03)
    test_data_graph.setPredictionResult("Learned", 1, 'NONE_ENTITY',       prediction=0.61)
    
    test_data_graph.setPredictionResult("Learned", 2, 'TRAJECTOR',         prediction=0.78)
    test_data_graph.setPredictionResult("Learned", 2, 'LANDMARK',          prediction=0.33)
    test_data_graph.setPredictionResult("Learned", 2, 'SPATIAL_INDICATOR', prediction=0.02)
    test_data_graph.setPredictionResult("Learned", 2, 'NONE_ENTITY',       prediction=0.48)
    
    test_data_graph.setPredictionResult("Learned", 3, 'TRAJECTOR',         prediction=0.01)
    test_data_graph.setPredictionResult("Learned", 3, 'LANDMARK',          prediction=0.03)
    test_data_graph.setPredictionResult("Learned", 3, 'SPATIAL_INDICATOR', prediction=0.93)
    test_data_graph.setPredictionResult("Learned", 3, 'NONE_ENTITY',       prediction=0.03)
    
    test_data_graph.setPredictionResult("Learned", 4, 'TRAJECTOR',         prediction=0.42)
    test_data_graph.setPredictionResult("Learned", 4, 'LANDMARK',          prediction=0.43)
    test_data_graph.setPredictionResult("Learned", 4, 'SPATIAL_INDICATOR', prediction=0.03)
    test_data_graph.setPredictionResult("Learned", 4, 'NONE_ENTITY',       prediction=0.05)
    
    test_data_graph.setPredictionResult("Learned", 5, 'TRAJECTOR',         prediction=0.22)
    test_data_graph.setPredictionResult("Learned", 5, 'LANDMARK',          prediction=0.13)
    test_data_graph.setPredictionResult("Learned", 5, 'SPATIAL_INDICATOR', prediction=0.01)
    test_data_graph.setPredictionResult("Learned", 5, 'NONE_ENTITY',       prediction=0.52)
    
    # triplet relation    
    test_data_graph.setPredictionResult("Learned", 0, "triplet", 1, 3, prediction=0.85) # ("stairs", "About 20 kids ", "on") - GT
    test_data_graph.setPredictionResult("Learned", 0, "triplet", 2, 3, prediction=0.78) # ("stairs", "About 20 kids", "on") - GT
    test_data_graph.setPredictionResult("Learned", 0, "triplet", 4, 3, prediction=0.42) # ("stairs", "hat", "on")
    test_data_graph.setPredictionResult("Learned", 0, "triplet", 5, 3, prediction=0.52) # ("stairs", "traditional clothing", "on")
    
    test_data_graph.setPredictionResult("Learned", 1, "triplet", 0, 3, prediction=0.32) # ("About 20 kids ", "stairs", "on")
    test_data_graph.setPredictionResult("Learned", 1, "triplet", 2, 3, prediction=0.30) # ("About 20 kids ", "About 20 kids", "on")
    test_data_graph.setPredictionResult("Learned", 1, "triplet", 4, 3, prediction=0.32) # ("About 20 kids ", "hat", "on")
    test_data_graph.setPredictionResult("Learned", 1, "triplet", 5, 3, prediction=0.29) # ("About 20 kids ", "traditional clothing", "on")
    
    test_data_graph.setPredictionResult("Learned", 2, "triplet", 0, 3, prediction=0.31) # ("About 20 kids", "stairs", "on")
    test_data_graph.setPredictionResult("Learned", 2, "triplet", 1, 3, prediction=0.30) # ("About 20 kids", "About 20 kids ", "on")
    test_data_graph.setPredictionResult("Learned", 2, "triplet", 4, 3, prediction=0.25) # ("About 20 kids ", "hat", "on")
    test_data_graph.setPredictionResult("Learned", 2, "triplet", 5, 3, prediction=0.27) # ("About 20 kids ", "traditional clothing", "on")
    
    # spatial_triplet relation
    test_data_graph.setPredictionResult("Learned", 0, "spatial_triplet", 1, 3, prediction=0.25) # ("stairs", "About 20 kids ", "on") - GT
    test_data_graph.setPredictionResult("Learned", 0, "spatial_triplet", 2, 3, prediction=0.38) # ("stairs", "About 20 kids", "on") - GT
    test_data_graph.setPredictionResult("Learned", 0, "spatial_triplet", 4, 3, prediction=0.22) # ("stairs", "hat", "on")
    test_data_graph.setPredictionResult("Learned", 0, "spatial_triplet", 5, 3, prediction=0.39) # ("stairs", "traditional clothing", "on")
    
    test_data_graph.setPredictionResult("Learned", 1, "spatial_triplet", 0, 3, prediction=0.32) # ("About 20 kids ", "stairs", "on")
    test_data_graph.setPredictionResult("Learned", 1, "spatial_triplet", 2, 3, prediction=0.30) # ("About 20 kids ", "About 20 kids", "on")
    test_data_graph.setPredictionResult("Learned", 1, "spatial_triplet", 4, 3, prediction=0.22) # ("About 20 kids ", "hat", "on")
    test_data_graph.setPredictionResult("Learned", 1, "spatial_triplet", 5, 3, prediction=0.39) # ("About 20 kids ", "traditional clothing", "on")
   
    test_data_graph.setPredictionResult("Learned", 2, "spatial_triplet", 0, 3, prediction=0.31) # ("About 20 kids", "stairs", "on")
    test_data_graph.setPredictionResult("Learned", 2, "spatial_triplet", 1, 3, prediction=0.30) # ("About 20 kids", "About 20 kids ", "on")
    test_data_graph.setPredictionResult("Learned", 2, "spatial_triplet", 4, 3, prediction=0.15) # ("About 20 kids ", "hat", "on")
    test_data_graph.setPredictionResult("Learned", 2, "spatial_triplet", 5, 3, prediction=0.27) # ("About 20 kids ", "traditional clothing", "on")
    
    # region
    test_data_graph.setPredictionResult("Learned", 0, "region", 1, 3, prediction=0.25) # ("stairs", "About 20 kids ", "on") - GT
    test_data_graph.setPredictionResult("Learned", 0, "region", 2, 3, prediction=0.38) # ("stairs", "About 20 kids", "on") - GT
    test_data_graph.setPredictionResult("Learned", 0, "region", 4, 3, prediction=0.22) # ("stairs", "hat", "on")
    test_data_graph.setPredictionResult("Learned", 0, "region", 5, 3, prediction=0.12) # ("stairs", "traditional clothing", "on")
    
    test_data_graph.setPredictionResult("Learned", 1, "region", 0, 3, prediction=0.32) # ("About 20 kids ", "stairs", "on")
    test_data_graph.setPredictionResult("Learned", 1, "region", 2, 3, prediction=0.30) # ("About 20 kids ", "About 20 kids", "on")
    test_data_graph.setPredictionResult("Learned", 1, "region", 4, 3, prediction=0.22) # ("About 20 kids ", "hat", "on")
    test_data_graph.setPredictionResult("Learned", 1, "region", 5, 3, prediction=0.39) # ("About 20 kids ", "traditional clothing", "on")
    
    test_data_graph.setPredictionResult("Learned", 2, "region", 0, 3, prediction=0.31) # ("About 20 kids", "stairs", "on")
    test_data_graph.setPredictionResult("Learned", 2, "region", 1, 3, prediction=0.30) # ("About 20 kids", "About 20 kids ", "on")
    test_data_graph.setPredictionResult("Learned", 2, "region", 4, 3, prediction=0.15) # ("About 20 kids ", "hat", "on")
    test_data_graph.setPredictionResult("Learned", 2, "region", 5, 3, prediction=0.27) # ("About 20 kids ", "traditional clothing", "on")

    # none_relation
    test_data_graph.setPredictionResult("Learned", 0, "none_relation", 1, 3, prediction=0.15) # ("stairs", "About 20 kids ", "on")
    test_data_graph.setPredictionResult("Learned", 0, "none_relation", 2, 3, prediction=0.08) # ("stairs", "About 20 kids", "on")
    
    yield test_data_graph

@pytest.mark.gurobi
def test_main_sprl(sprl_input):
    from regr.solver.ilpOntSolverFactory import ilpOntSolverFactory
    from regr.solver.ilpOntSolver import ilpOntSolver
    from regr.graph import DataGraph

    test_data_graph = sprl_input
    
     # ------Call solver -------

    myilpSolver = ilpOntSolverFactory.getOntSolverInstance(test_data_graph.getOntologyGraph())
    test_data_graph_updated = myilpSolver.calculateILPSelection(instance = test_phrase, ontologyGraph = test_data_graph)

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
