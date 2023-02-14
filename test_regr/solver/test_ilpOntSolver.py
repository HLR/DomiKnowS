import pytest

@pytest.fixture()
def emr_input(request):
    import numpy as np

    test_phrase = [("John", "NNP"), ("works", "VBN"), ("for", "IN"), ("IBM", "NNP")]
    conceptNamesList = ["people", "organization", "other", "location", "O"]
    relationNamesList = ["work_for", "live_in", "located_in"]

    # tokens
    test_graphResultsForPhraseToken = {}
    #                                                           John  works for  IBM
    test_graphResultsForPhraseToken["people"] =       np.array([0.7, 0.1, 0.02, 0.6])
    test_graphResultsForPhraseToken["organization"] = np.array([0.5, 0.2, 0.03, 0.91])
    test_graphResultsForPhraseToken["other"] =        np.array([0.3, 0.6, 0.05, 0.5])
    test_graphResultsForPhraseToken["location"] =     np.array([0.3, 0.4, 0.1 , 0.3])
    test_graphResultsForPhraseToken["O"] =            np.array([0.1, 0.9, 0.9 , 0.1])
    
    test_graphResultsForPhraseToken["people"] =       np.array([[0.3, 0.9, 0.08, 0.4],
                                                               [0.7, 0.1, 0.02, 0.6]])
        
    test_graphResultsForPhraseToken["people"] = np.swapaxes(test_graphResultsForPhraseToken["people"], 1, 0)
    
    test_graphResultsForPhraseToken["organization"] = np.array([[0.5, 0.8, 0.07, 0.09],
                                                               [0.5, 0.2, 0.03, 0.91]])
    
    test_graphResultsForPhraseToken["organization"] = np.swapaxes(test_graphResultsForPhraseToken["organization"], 1, 0)
    
    test_graphResultsForPhraseToken["other"] =        np.array([[0.7, 0.4, 0.05, 0.5],
                                                               [0.3, 0.6, 0.05, 0.5]])
    
    test_graphResultsForPhraseToken["other"] = np.swapaxes(test_graphResultsForPhraseToken["other"], 1, 0)

    test_graphResultsForPhraseToken["location"] =     np.array([[0.7, 0.6, 0.9 , 0.7],
                                                               [0.3, 0.4, 0.1 , 0.3]])
    
    test_graphResultsForPhraseToken["location"] = np.swapaxes(test_graphResultsForPhraseToken["location"], 1, 0)

    test_graphResultsForPhraseToken["O"] =            np.array([[0.9, 0.1, 0.1 , 0.9],
                                                               [0.1, 0.9, 0.9 , 0.1]])
    
    test_graphResultsForPhraseToken["O"] = np.swapaxes(test_graphResultsForPhraseToken["O"], 1, 0)
    
    test_graphResultsForPhraseRelation = dict()
    # work_for
    #                                    John  works for   IBM
    work_for_relation_table = np.array([[0.40, 0.20, 0.20, 0.63],  # John
                                        [float("nan"), float("nan"), 0.40, 0.30],  # works
                                        [0.02, 0.03, 0.05, 0.10],  # for
                                        [0.65, 0.20, 0.10, 0.30],  # IBM
                                        ])
    test_graphResultsForPhraseRelation["work_for"] = work_for_relation_table

    # live_in
    #                                   John  works for   IBM
    live_in_relation_table = np.array([[0.10, 0.20, 0.20, 0.06],  # John
                                       [float("nan"), float("nan"), 0.20, 0.10],  # works
                                       [0.02, 0.03, 0.05, 0.10],  # for
                                       [0.10, 0.20, 0.10, float("nan")],  # IBM
                                       ])
    test_graphResultsForPhraseRelation["live_in"] = live_in_relation_table

    # located_in
    #                                      John  works for   IBM
    located_in_relation_table = np.array([[0.10, 0.20, 0.20, 0.06],  # John
                                          [float("nan"), float("nan"), float("nan"), float("nan")],  # works
                                          [0.02, 0.03, 0.05, 0.10],  # for
                                          [0.03, 0.20, 0.10, float("nan")],  # IBM
                                          ])
    test_graphResultsForPhraseRelation["located_in"] = located_in_relation_table

    yield test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation

@pytest.mark.skip(reason="define model building datanode")
@pytest.mark.gurobi
def test_main_emr_owl(emr_input):
    from domiknows.solver.ilpOntSolverFactory import ilpOntSolverFactory
    from domiknows.solver.ilpOntSolver import ilpOntSolver
    from domiknows.graph import Graph

    test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation = emr_input

    # ------Call solver -------
    test_graph = Graph(iri='http://ontology.ihmc.us/ML/EMR.owl', local='./examples/emr/')
    
    myilpOntSolver = ilpOntSolverFactory.getOntSolverInstance(test_graph)
    tokenResult, relationsResult, _ = myilpOntSolver.calculateILPSelection(test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation, None)

    # -------Evaluate results
    # token results:
    # expect John[people] works[o] for[o] IBM[organization]
    assert tokenResult['people'].sum() == 1
    assert tokenResult['organization'].sum() == 1
    assert tokenResult['other'].sum() == 0
    assert tokenResult['location'].sum() == 0
    assert tokenResult['O'].sum() == 2

    assert tokenResult['people'][0] == 1  # John
    assert tokenResult['O'][1] == 1  # works
    assert tokenResult['O'][2] == 1  # for
    assert tokenResult['organization'][3] == 1  # IBM

    # relation results:
    # (John, IBM)[work_for]
    assert relationsResult['work_for'].sum().sum() == 1
    assert relationsResult['work_for'][0][3] == 1 # John - IBM
    assert relationsResult['live_in'].sum().sum() == 0
    assert relationsResult['located_in'].sum().sum() == 0
