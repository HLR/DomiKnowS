import pytest

@pytest.fixture()
def emr_input(request):
    import numpy as np
    import pandas as pd

    tokenList = ["0_John", "1_works", "2_for", "3_IBM"]
    conceptNamesList = ["people", "organization", "other", "location", "O"]
    relationNamesList = ["work_for", "live_in", "located_in"]

    # tokens
    test_graphResultsForPhraseToken = {}
    #                                                  John  works for   IBM
    test_graphResultsForPhraseToken["people"] =       [0.7 , 0.  , 0.02, 0.92]
    test_graphResultsForPhraseToken["organization"] = [0.98, 0.5 , 0.03, 0.93]
    test_graphResultsForPhraseToken["other"] =        [0.95, 0.4 , 0.05, 0.93]
    test_graphResultsForPhraseToken["location"] =     [0.02, 0.6 , 0.1 , 0.9 ]
    test_graphResultsForPhraseToken["O"] =            [0.  , 0.9 , 0.9 , 0.  ]
    
    test_graphResultsForPhraseRelation = dict()
    # work_for
    #                                    John  works for   IBM
    work_for_relation_table = np.array([[0.50, 0.20, 0.20, 0.63],  # John
                                        [0.00, 0.00, 0.40, 0.30],  # works
                                        [0.02, 0.03, 0.05, 0.10],  # for
                                        [0.26, 0.20, 0.10, 0.90],  # IBM
                                        ])
    test_graphResultsForPhraseRelation["work_for"] = work_for_relation_table

    # live_in
    #                                   John  works for   IBM
    live_in_relation_table = np.array([[0.10, 0.20, 0.20, 0.06],  # John
                                       [0.00, 0.00, 0.20, 0.10],  # works
                                       [0.02, 0.03, 0.05, 0.10],  # for
                                       [0.10, 0.20, 0.10, 0.00],  # IBM
                                       ])
    test_graphResultsForPhraseRelation["live_in"] = live_in_relation_table

    # located_in
    #                                      John  works for   IBM
    located_in_relation_table = np.array([[0.10, 0.20, 0.20, 0.06],  # John
                                          [0.00, 0.00, 0.00, 0.00],  # works
                                          [0.02, 0.03, 0.05, 0.10],  # for
                                          [0.03, 0.20, 0.10, 0.00],  # IBM
                                          ])
    test_graphResultsForPhraseRelation["located_in"] = located_in_relation_table

    yield tokenList, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation


@pytest.mark.gurobi
def test_main_emr_owl(emr_input):
    from regr.solver.ilpSelectClassification import ilpOntSolver
    from regr.graph import Graph

    tokenList, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation = emr_input

    # ------Call solver -------
    test_graph = Graph(iri='http://ontology.ihmc.us/ML/EMR.owl', local='./examples/emr/')
    
    myilpOntSolver = ilpOntSolver.getInstance(test_graph)
    tokenResult, relationsResult, _ = myilpOntSolver.calculateILPSelection(
        tokenList, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation)

    # -------Evaluate results
    # token results:
    # expect John[people] works[o] for[o] IBM[organization]
    assert tokenResult.sum()['people'] == 1
    assert tokenResult.sum()['organization'] == 1
    assert tokenResult.sum()['other'] == 0
    assert tokenResult.sum()['location'] == 0
    assert tokenResult.sum()['O'] == 2

    assert tokenResult['people'][0] == 1  # John
    assert tokenResult['O'][1] == 1  # works
    assert tokenResult['O'][2] == 1  # for
    assert tokenResult['organization'][3] == 1  # IBM

    # relation results:
    # (John, IBM)[work_for]
    assert relationsResult['work_for'].sum().sum() == 1
    assert relationsResult['work_for']['3_IBM']['0_John'] == 1
    assert relationsResult['live_in'].sum().sum() == 0
    assert relationsResult['located_in'].sum().sum() == 0

    assert relationsResult['work_for'][tokenList[3]][tokenList[0]] == 1  # IBM - John


@pytest.mark.gurobi
def test_main_ont(emr_input):
    from regr.solver.ilpSelectClassification import ilpOntSolver
    from regr.graph import Graph

    tokenList, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation = emr_input

    # ------Call solver -------
    test_graph = Graph(iri='http://trips.ihmc.us/ont', local='./examples/emr/')
    
    myilpOntSolver = ilpOntSolver.getInstance(test_graph)
    tokenResult, relationsResult, _ = myilpOntSolver.calculateILPSelection(
        tokenList, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation)

    # -------Evaluate results
    # token results:
    # expect John[people] works[o] for[o] IBM[organization]
    assert tokenResult.sum()['people'] == 1
    assert tokenResult.sum()['organization'] == 1
    assert tokenResult.sum()['other'] == 0
    assert tokenResult.sum()['location'] == 0
    assert tokenResult.sum()['O'] == 2

    assert tokenResult['people'][0] == 1  # John
    assert tokenResult['O'][1] == 1  # works
    assert tokenResult['O'][2] == 1  # for
    assert tokenResult['organization'][3] == 1  # IBM

    # relation results:
    # (John, IBM)[work_for]
    assert relationsResult['work_for'].sum().sum() == 1
    assert relationsResult['live_in'].sum().sum() == 0
    assert relationsResult['located_in'].sum().sum() == 0

    assert relationsResult['work_for'][tokenList[3]][tokenList[0]] == 1  # IBM - John
