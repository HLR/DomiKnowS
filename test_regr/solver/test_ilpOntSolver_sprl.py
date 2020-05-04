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
    phrases = ["0_stairs",                # GT: LANDMARK
                "1_About 20 kids ",        # GT: TRAJECTOR
                "2_About 20 kids",         # GT: TRAJECTOR
                "3_on",                    # GT: SPATIALINDICATOR
                "4_hats",                  # GT: NONE
                "5_traditional clothing"]  # GT: NONE
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
    test_graphResultsForPhraseToken['LANDMARK']          = np.array([0.68, 0.15, 0.01, 0.03, 0.43, 0.13])
    test_graphResultsForPhraseToken['TRAJECTOR']         = np.array([0.37, 0.78, 0.78, 0.01, 0.49, 0.22])
    test_graphResultsForPhraseToken['SPATIAL_INDICATOR'] = np.array([0.05, 0.03, 0.02, 0.93, 0.03, 0.48])
    test_graphResultsForPhraseToken['NONE_ENTITY']       = np.array([0.2 , 0.61, 0.48, 0.03, 0.51, 0.51])
    
    test_graphResultsForTripleRelations = dict()
    
    # triplet
    # triplet relation is a 3D array
    triplet_relation_table = np.random.rand(6, 6, 6) * 0.2
    triplet_relation_table[0, 1, 3] = 0.85 # ("stairs", "About 20 kids ", "on") - GT
    triplet_relation_table[0, 2, 3] = 0.78 # ("stairs", "About 20 kids", "on") - GT
    triplet_relation_table[1, 2, 3] = 0.99 # JUST FOR FUNCTIONAL TEST
    # ... can be more
    test_graphResultsForTripleRelations["triplet"] = triplet_relation_table
    
    # spatial_triplet
    # triplet relation is a 3D array
    spatial_triplet_relation_table = np.random.rand(6, 6, 6) * 0.2
    spatial_triplet_relation_table[0, 1, 3] = 0.74 # ("stairs", "About 20 kids ", "on")
    spatial_triplet_relation_table[0, 2, 3] = 0.23 # ("stairs", "About 20 kids", "on")
    spatial_triplet_relation_table[0, 1, 5] = 0.50 # TEST: lm - 0.68, tr - 0.78, sp - 0.48
    spatial_triplet_relation_table[0, 4, 5] = 0.50 # TEST: lm - 0.68, tr - 0.49, sp - 0.48
    # ... can be more
    test_graphResultsForTripleRelations["spatial_triplet"] = spatial_triplet_relation_table
    
    # region
    # triplet relation is a 3D array
    region_relation_table = np.random.rand(6, 6, 6) * 0.2
    region_relation_table[0, 1, 3] = 0.94 # ("stairs", "About 20 kids ", "on")
    region_relation_table[0, 2, 3] = 0.88 # ("stairs", "About 20 kids", "on")
    region_relation_table[1, 2, 3] = 0.51 # TEST
    region_relation_table[0, 1, 5] = 0.52 # TEST: 5_sp:0.48(-2), 5_ne:0.51(-1), <st:0.5>
    region_relation_table[0, 4, 5] = 0.53 # TEST: 4_tr:0.49(-1), 4_ne:0.51(-1), <5_sp:0.48(-2), st:0.5>
    # ... can be more
    test_graphResultsForTripleRelations["region"] = region_relation_table

    # distance
    # triplet relation is a 3D array
    distance_relation_table = np.random.rand(6, 6, 6) * 0.2
    distance_relation_table[0, 1, 3] = 0.65 # ("stairs", "About 20 kids ", "on")
    distance_relation_table[0, 2, 3] = 0.88 # ("stairs", "About 20 kids", "on")
    # ... can be more
    test_graphResultsForTripleRelations["distance"] = distance_relation_table

    # direction
    # triplet relation is a 3D array
    direction_relation_table = np.random.rand(6, 6, 6) * 0.2
    direction_relation_table[0, 1, 3] = 0.65 # ("stairs", "About 20 kids ", "on")
    direction_relation_table[0, 2, 3] = 0.88 # ("stairs", "About 20 kids", "on")
    # ... can be more
    test_graphResultsForTripleRelations["direction"] = direction_relation_table

    yield test_phrase, test_graphResultsForPhraseToken, test_graphResultsForTripleRelations


@pytest.mark.gurobi
@pytest.fixture()
def sprl_solution(sprl_input):
    from regr.solver.ilpOntSolverFactory import ilpOntSolverFactory
    from regr.solver.ilpOntSolver import ilpOntSolver
    from regr.graph import Graph

    test_phrase, test_graphResultsForPhraseToken, test_graphResultsForTripleRelations = sprl_input

    # ------Call solver -------
    test_graph = Graph(iri='http://ontology.ihmc.us/ML/SPRL.owl', local='./examples/SpRL/')
    
    myilpOntSolver = ilpOntSolverFactory.getOntSolverInstance(test_graph)
    tokenResult, _, tripleRelationsResult = myilpOntSolver.calculateILPSelection(test_phrase, test_graphResultsForPhraseToken, None, test_graphResultsForTripleRelations)
    return tokenResult, tripleRelationsResult


@pytest.mark.gurobi
@pytest.fixture()
def sprl_solution_sp_rg(sprl_input):
    from regr.solver.ilpOntSolverFactory import ilpOntSolverFactory
    from regr.solver.ilpOntSolver import ilpOntSolver
    from regr.graph import Graph

    test_phrase, test_graphResultsForPhraseToken, test_graphResultsForTripleRelations = sprl_input
    del test_graphResultsForTripleRelations['triplet']
    del test_graphResultsForTripleRelations['distance']
    del test_graphResultsForTripleRelations['direction']

    # ------Call solver -------
    test_graph = Graph(iri='http://ontology.ihmc.us/ML/SPRL.owl', local='./examples/SpRL/')
    
    myilpOntSolver = ilpOntSolverFactory.getOntSolverInstance(test_graph)
    tokenResult, _, tripleRelationsResult = myilpOntSolver.calculateILPSelection(test_phrase, test_graphResultsForPhraseToken, None, test_graphResultsForTripleRelations)
    return tokenResult, tripleRelationsResult


def test_sprl_entities(sprl_solution):
    tokenResult, _ = sprl_solution

    # -- phrase results:
    assert tokenResult['LANDMARK'][0] == 1 # "stairs" - LANDMARK
    assert tokenResult['LANDMARK'][1] == 0, 'triplet[1,2,3]=0.99' # FIXME: error due to triplet [1,2,3]
    assert tokenResult['LANDMARK'].sum() == 1

    assert tokenResult['TRAJECTOR'][1] == 1 # "About 20 kids" - TRAJECTOR
    assert tokenResult['TRAJECTOR'][2] == 1 # "About 20 kids " - TRAJECTOR
    assert tokenResult['TRAJECTOR'].sum() == 2

    assert tokenResult['SPATIAL_INDICATOR'][3] == 1 # "on" - SPATIALINDICATOR
    assert tokenResult['SPATIAL_INDICATOR'].sum() == 1

    assert tokenResult['NONE_ENTITY'][4] == 1 # "hats" - NONE
    assert tokenResult['NONE_ENTITY'][5] == 1  # "traditional clothing" - NONE
    assert tokenResult['NONE_ENTITY'].sum() == 2


def test_sprl_relation_triplet(sprl_solution):
    _, tripleRelationsResult = sprl_solution
    assert tripleRelationsResult['triplet'][1, 2, 3] == 1


def test_sprl_relation_region(sprl_solution):
    _, tripleRelationsResult = sprl_solution
    assert tripleRelationsResult['region'][0, 1, 5] == 0
    assert tripleRelationsResult['region'][0, 4, 5] == 0


def test_sprl_relation_composition(sprl_solution):
    from itertools import product
    tokenResult, tripleRelationsResult = sprl_solution
    relations = ['spatial_triplet', 'region', 'distance', 'direction']
    for relation in relations:
        if relation in tripleRelationsResult:
            for lm, tr, sp in product(range(6), repeat=3):
                assert tripleRelationsResult[relation][lm, tr, sp] <= tokenResult['LANDMARK'][lm]
                assert tripleRelationsResult[relation][lm, tr, sp] <= tokenResult['TRAJECTOR'][tr]
                assert tripleRelationsResult[relation][lm, tr, sp] <= tokenResult['SPATIAL_INDICATOR'][sp]

def test_sprl_relation_composition_sp_rg(sprl_solution_sp_rg):
    from itertools import product
    tokenResult, tripleRelationsResult = sprl_solution_sp_rg
    relations = ['spatial_triplet', 'region', 'distance', 'direction']
    for relation in relations:
        if relation in tripleRelationsResult:
            for lm, tr, sp in product(range(6), repeat=3):
                assert tripleRelationsResult[relation][lm, tr, sp] <= tokenResult['LANDMARK'][lm]
                assert tripleRelationsResult[relation][lm, tr, sp] <= tokenResult['TRAJECTOR'][tr]
                assert tripleRelationsResult[relation][lm, tr, sp] <= tokenResult['SPATIAL_INDICATOR'][sp]


def test_sprl_relation_constraint_subtype(sprl_solution):
    from itertools import product
    _, tripleRelationsResult = sprl_solution
    relations = ['region', 'distance', 'direction']
    for relation in relations:
        if relation in tripleRelationsResult:
            for lm, tr, sp in product(range(6), repeat=3):
                assert tripleRelationsResult['spatial_triplet'][lm, tr, sp] >= tripleRelationsResult[relation][lm, tr, sp], 'sub-type constraint not working: spatial_triplet{} >= {}{}'.format([lm, tr, sp], relation, [lm, tr, sp])

def test_sprl_relation_constraint_subtype_sp_rg(sprl_solution_sp_rg):
    from itertools import product
    _, tripleRelationsResult = sprl_solution_sp_rg
    relations = ['region', 'distance', 'direction']
    for relation in relations:
        if relation in tripleRelationsResult:
            for lm, tr, sp in product(range(6), repeat=3):
                assert tripleRelationsResult['spatial_triplet'][lm, tr, sp] >= tripleRelationsResult[relation][lm, tr, sp], 'sub-type constraint not working: spatial_triplet{} >= {}{}'.format([lm, tr, sp], relation, [lm, tr, sp])


def test_sprl_relation_constraint_disjoint(sprl_solution):
    from itertools import product
    _, tripleRelationsResult = sprl_solution

    if 'distance' in tripleRelationsResult and 'direction' in tripleRelationsResult:
        # direction-distance disjoint
        for lm, tr, sp in product(range(6), repeat=3):
            assert tripleRelationsResult['distance'][lm, tr, sp] + tripleRelationsResult['direction'][lm, tr, sp] <= 1, 'disjoint constraint not working: distance{} + direction{} <= 1'.format([lm, tr, sp], [lm, tr, sp])
